# Module Reference: VN Cross-Sectional Ranking System

What each module in `src/`, `config.py` and `kaggle_kernel/kernel.py` does and how the pieces connect. For setup, results and usage examples see [README.md](README.md). This file describes the code as it is now; where code and config disagree, that is called out under **Known issues**.

## Contents

1. [Data flow](#data-flow)
2. [config.py](#configpy)
3. [data_collect.py](#data_collectpy)
4. [features.py](#featurespy)
5. [alpha_mining.py](#alpha_miningpy)
6. [models.py](#modelspy)
7. [simulation.py](#simulationpy)
8. [evaluation.py](#evaluationpy)
9. [significance_test.py](#significance_testpy)
10. [inference.py](#inferencepy)
11. [deep_combiner.py](#deep_combinerpy)
12. [app.py](#apppy)
13. [kaggle_kernel/kernel.py](#kaggle_kernelkernelpy)
14. [gplearn factor mining (not recommended)](#gplearn-factor-mining-not-recommended)
15. [Known issues](#known-issues)

---

## Data flow

```
HF dataset ─► data_collect.py (pull, update, push) ─► market_data.parquet, vnindex_data.parquet
        │
        ▼
features.build_features()
  ├─ generate_target=True  ─► training frame  (target_magnitude, next_1m_ret, ...)
  └─ generate_target=False ─► feature frame including the latest date
        │
   ┌────┴──────────────────────────────┐
   ▼                                   ▼
models.walk_forward_cv()        inference.generate_paper_trade_signals()
   │ pred_score per test fold          │ (called from kaggle_kernel/kernel.py)
   ▼                                   ▼
simulation.OrderManager         today_signals.parquet ─► Hugging Face ─► app.py
evaluation.* / significance_test.*
```

The gplearn path (`use_gp=True`) is an optional side branch inside `walk_forward_cv` and is **not recommended**; see [its section](#gplearn-factor-mining-not-recommended).

---

## `config.py`

Constants plus one helper. Imported by `models.py`, `inference.py` and `kernel.py`.

| Name | Description |
|---|---|
| `candidate_features` | Feature pool passed to `walk_forward_cv` and `generate_paper_trade_signals` (**53** names): 5 returns, 6 volatility, 10 SMA/EMA distances, 4 volume, `RSI_14`, and **27** `WQ_Alpha_*` columns: 002, 006, 007, 013, 016, 024, 028, 040 (original set); 102–105 (monthly price/risk signals); 110–118 (slow-moving additions); 019, 035, 039, 043, 052, 055 (Alpha101 adaptations stretched to monthly windows). Every name is produced by `build_features`. `dist_52w_high`, `log_ret_skip1m`, `WQ_Alpha_201/202` and four structural features (`turnover_12m`, `limit_bias_60d`, `herding_dispersion`, `amihud_illiquidity`) are commented out. Of these, `herding_dispersion` is only a leftover name in `config.py`: `features.py` has no function that computes it. |
| `BASE_MODEL_PARAMS` | Default XGBoost parameters: `rank:ndcg`, `device='cuda'`, `tree_method='hist'`, 150 trees, depth 3, `min_child_weight=5`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.6`, `reg_lambda=5`, `reg_alpha=1`, `lambdarank_pair_method='topk'`, `lambdarank_num_pair_per_sample=20`, `ndcg_exp_gain=False`, `random_state=42`. |
| `FEATURE_GROUPS` | `{group: [features]}` for `returns`, `volatility`, `moving_average`, `volume`, `rsi`, `wq_features` (the 27 alphas). Together the groups cover exactly the features in `candidate_features`. The `price_structure` group is commented out. Correlation pruning only compares features **within** a group; any column outside every group (for example `GP_Alpha_*`) is a singleton and is never pruned. |
| `ALL_GROUP_NAMES` | `list(FEATURE_GROUPS)`. |
| `usedSymbols` | The 280 tickers that `kernel.py` filters the market data down to. |
| `groups_to_features(group_names)` | Flattens group names into one feature list. |

---

## `data_collect.py`

Builds the ticker universe and keeps `market_data.parquet` and `vnindex_data.parquet` current through `vnstock`. Running the file as a script (`python src/data_collect.py`) updates both files from `2016-01-01`; this is what `daily_update.yml` runs. The script requires `HF_TOKEN`: it first pulls both parquet files from the Hugging Face dataset (`HF_DATA_REPO`, default `PhongHPham/vn_cross_sectional_ranking_data_storage`), updates them, pushes only the files whose sha256 changed, and writes `changed=true/false` to `$GITHUB_OUTPUT` so the workflow can skip the Kaggle run. A missing file on HF (`EntryNotFoundError`) starts from scratch; any other download error aborts, so a transient failure cannot overwrite the full history. It also writes `data_collect.log`.

Helpers: `_file_hash(path)` (sha256 or `None`), `pull_from_hf(filename)`, `push_to_hf(filename)`.

- **`clean_symbols(symbol_list)`**: Strips, de-duplicates and sorts ticker strings, dropping nulls.
- **`get_tags(fetching=False)`**: Returns a hard-coded list of about 300 tickers. With `fetching=True` returns the live `VNALL` index members from `vnstock.Reference`.
- **`clean_ohlcv(df)`**: Normalises a raw OHLCV frame. Rows that break an OHLC rule are repaired by carrying the whole candle forward; volume is left alone and no rows are dropped.
- **`_fetch_quote(symbol, start_d, end_d)`**: Daily OHLCV for one symbol. Returns `None` on failure.
- **`update_market_data(file_path, symbols, start_date="2016-01-01", batch_size=1, base_sleep=7, filter_symbols=True)`**: Main entry point. Works out the last completed trading day (today counts after 14:45 Vietnam time), skips symbols whose stored data already reaches it, and for every other symbol re-fetches from `start_date` and merges by `(date, Symbol)`, new rows winning. Saves after each batch, sleeps `base_sleep` seconds per symbol, and retries once when the API reports a rate limit. `filter_symbols=True` purges stored symbols that are no longer in `symbols`.
- **`_drop_before_last_zero_volume(df)`**: For each symbol, drops everything up to and including its most recent zero-volume day (pre-listing and halt noise). A halt late in the history therefore removes all earlier history for that symbol.
- **`_merge_and_dedup(existing, new_chunks)`**: Concatenates, de-duplicates on `(date, Symbol)`, keeps dates from 2016-01-01 and applies the zero-volume trim.
- **`fetch_indicator_data(ind_symbol, start_date, file_path)`**: Index data (for example `VNINDEX`) to Parquet.
- **`_save(df, file_path)`**: Writes Parquet.

---

## `features.py`

Feature engineering, targets and market-regime columns. Contract for every per-symbol function: **row *i* (date *T*) uses data only up to the close of *T-1***. Each function lags its own inputs.

### Feature functions

| Function | Adds |
|---|---|
| `return_ln(df)` | `log_ret_1w/1m/3m/6m/1y` |
| `volatility(df)` | `volatility_1w/1m/3m/6m` (annualised) and `volatility_shock_monthly/weekly` |
| `MA(df)` | SMA/EMA at 9/21/50/100/200 and `dist_SMA_*`, `dist_EMA_*` |
| `volume(df)` | `vol_5d/1m/3m`, `volume_surge_weekly/monthly` |
| `rsi(df, window_length=14)` | `RSI_14` |
| `volume_quality(df)` | `obv_trend`, `price_vol_divergence` |
| `price_structure(df)` | `dist_52w_high`, `log_ret_skip1m`. Not called by `build_features` and not in `candidate_features`; call it yourself if you want these features |
| `turnover_ratio(df)`, `limit_bias(df, d=60)`, `amihud_illiquidity(df)` | Structural features. Computed by `build_features`, but their candidate entries in `config.py` are commented out |
| `adtv(df, window=21)` | 21-day average daily traded value (lagged), used for the liquidity mask |
| `market_breadth(df, liquidity_mask=None)` | `breadth_ema21`, `breadth_25pct_threshold`, `narrow_breadth` |
| `robust_market_regime_pipeline_monthly(df, return_horizon=21, rolling_window=63, num_buckets=4, liquidity_mask=None)` | The **P2** input. Takes each stock's lagged 21-day return, computes the cross-sectional MAD per date over liquid rows, smooths it with a 63-day rolling mean, converts it to a point-in-time percentile over a trailing 252-day window, and cuts it into quartiles. Outputs `regime_disp_raw_monthly`, `regime_percentile_monthly`, `regime_bucket_monthly` (`Q1` lowest dispersion to `Q4` highest) |

### Target and label functions

- **`build_targets(df)`**: `next_1m_ret = log(close[t+21]/close[t])` and `next_1w_ret` (5-day), then **drops rows where either is NaN**, i.e. the last ~21 sessions per symbol.
- **`make_magnitude_label(df, target_col="risk_adj_ret", scale_max=100, winsor_pct=0.0)`**: Per-date min-max scaling to `[0, 100]`, producing `target_magnitude`. This is the training and evaluation label for every model in `MODEL_REGISTRY` and for `alphaforge`.
- **`make_adaptive_bucket_label(...)`**: Per-date bucket labels (`target_bucket`) with a bucket count chosen to give ~20 stocks each. **Not called** by `build_features` any more.
- **`target_generating_ranking(df, freq="M")`**: Builds `risk_adj_ret`, quintile labels `target_quintile`, and `qid`. **Not called** by `build_features` and not needed by any model in `walk_forward_cv`.
- **`apply_cross_sectional_ranking(df, feature_cols)`**: Adds per-date percentile-rank columns `csr_<feature>`. Exploratory only.

### `build_features(df, min_stocks_per_date=50, adtv_limit=None, generate_target=True)`

Order of work: returns, volatility, MAs, volume, RSI, volume quality, structural features; then (if `generate_target`) `build_targets`, `risk_adj_ret = next_1m_ret / volatility_3m`, `make_magnitude_label`; then ADTV and the `is_liquid` mask; WQ alphas (27 columns); the regime pipeline; `market3m_ema63` and `market1m_ema21` (the **P3** inputs: EMA span 63 of the liquid-universe mean `log_ret_3m`, EMA span 21 of the mean `log_ret_1m`); breadth. Finally it drops illiquid rows (once), replaces ±inf with NaN, drops rows with any NaN in a computed column, and removes dates with fewer than `min_stocks_per_date` symbols.

- `generate_target=False` skips target construction, so the latest date survives. The regime and trend columns are produced either way.
- With `adtv_limit` set, per-symbol rolling windows still run on the full series; only cross-sectional statistics use the mask.
- The longest alpha lookbacks (`log_ret_1y`, `WQ_Alpha_105`, `WQ_Alpha_112` and others, ~253 days) set how much history a symbol needs before its first row survives.

---

## `alpha_mining.py`

WorldQuant-style alphas, plus gplearn helpers (legacy; see the [gplearn section](#gplearn-factor-mining-not-recommended)).

- **Operators:** `to_series`, `ts_delay`, `ts_delta`, `ts_mean`, `ts_min`, `ts_rank`, `ts_argmax` (per-symbol, rolling, `min_periods=d` so warm-up rows stay NaN) and `cs_mask`, `cs_rank`, `cs_mean` (per-date, liquidity-mask aware).
- **`class WorldQuantAlphas(df, liquidity_mask=None)`**: Builds a lagged OHLCV copy (`ldf`) so every alpha respects the T-1 rule. `generate_all()` returns **27** columns:
  - Original set, rewritten to be price-scale invariant: `002, 006, 007, 013, 016, 024, 028, 040`. Correlation/covariance windows are 21 days.
  - Monthly price and risk signals: `102` (price acceleration), `103` (52-week-high proximity, a momentum-type signal), `104` (BAB, negative 60-day beta), `105` (12-1m momentum).
  - Slow-moving additions: `110` (MAX effect), `111` (negative 63d skewness), `112` (months 7-12 momentum), `113` (risk-adjusted 6-1m momentum), `114` (information discreteness), `115` (overnight drift), `116` (close-location value), `117` (abnormal volume), `118` (Parkinson low-vol).
  - Alpha101 adaptations stretched to monthly windows: `019, 035, 039, 043, 052, 055`.

  `get_alpha_201` and `get_alpha_202` exist, but their calls in `generate_all()` are commented out.
- **`add_and_filter_alphas(gp_model, original_df, input_features)`** (legacy, gplearn): transforms the whole frame with a fitted `SymbolicTransformer` and appends the columns that survive an exact-duplicate check. Not used by `walk_forward_cv`.
- **`rank_ic_fitness(y_true, y_pred)`** (legacy, gplearn): pooled Spearman between a candidate and forward returns. It is not wrapped with gplearn's `make_fitness` and nothing calls it.

---

## `models.py`

Feature selection, walk-forward machinery and the model zoo.

### Feature selection (training window only)

- **`select_features_by_icir(df, candidate_features, icir_threshold=0.02, target_col="next_1m_ret")`**: Vectorised per-date Spearman IC for each feature, aggregated to IR = mean/std. Returns `(selected, ic_ir_map)` keeping features with |IC IR| above the threshold.
- **`_compute_icir_map(df, features, target_col)`**: Same computation without thresholding. Used for tie-breaking when `corr_prune=True` and `icir_filter=False`.
- **`prune_correlated_features(df, candidate_features, ic_ir_map, feature_groups=None, correlation_threshold=0.75)`**: Within each `FEATURE_GROUPS` group, for any pair with |Spearman| at or above the threshold, drops the one with the lower |IC IR|. Features in different groups, or in no group, never compete.
- **`test_train_spliter(df, test_start, features)`**: Single split helper for ad-hoc work.

### Walk-forward windows

All lengths are counts of **unique trading dates** in the data.

- **`class Fold`**: dataclass with `fold_number`, `train_df`, `test_df`, `test_start`, `test_end`.
- **`class WindowPolicy(initial_train_dates, gap_dates=0)`**: abstract base. `iter_slices(unique_dates)` yields `(train_dates, test_dates)` pairs. Raises if `gap_dates >= initial_train_dates`.
- **`class RollingWindow(initial_train_dates, train_percent, gap_dates=0)`**: fixed-length train window. Test length is `round((initial_train_dates - gap_dates) × (1 - train_percent) / train_percent)`, and the window advances by exactly the test length, so test blocks tile the timeline with no overlap and no holes. The last incomplete fold is dropped.
- **`class ExpandingWindow(initial_train_dates, test_len_dates=None, train_percent=None, gap_dates=0)`**: train starts at date 0 and grows by one test block per fold. Test length is fixed for the whole run, given directly or derived once from `train_percent`.
- **`generate_folds(df, window_policy, liquidity_filter=False, adequate_adtv=2_500_000)`**: Turns a policy's date slices into `Fold` objects and assigns a per-date `qid` to each train and test frame. With `liquidity_filter=True` each fold is filtered separately by `_filter_fold_by_liquidity`.
- **`_filter_fold_by_liquidity(...)`**: ADTV filter applied per fold using the `adtv` column (rows with NaN ADTV are kept).
- **`_test_len_from_ratio(eff_train_dates, train_percent)`**: shared test-length formula.

### Model zoo

Each `predict_*` takes a `Fold` and returns the test frame with a `pred_score` column.

- **`predict_xgboost_ndcg(fold, features, model_params)`**: **Production.** `XGBRanker` (`rank:ndcg`) fit on `target_magnitude` with `qid` groups. Defaults to `BASE_MODEL_PARAMS`.
- **`predict_xgboost_mse(...)`**: `XGBRegressor` with `reg:squarederror`, same label.
- **`predict_linear(...)`**: `StandardScaler` plus `LassoCV(cv=5)`, same label. (Called "linear" in the registry and "LassoCV" in the notebook.)
- **`predict_lstm_mse(...)`**: Small PyTorch LSTM over per-symbol windows, MSE loss. `model_params` may set `seq_len` (10), `hidden_size` (32), `num_layers` (1), `dropout` (0), `epochs` (20), `batch_size` (256), `lr` (1e-3), `device`. Symbols without enough history get the median training prediction.
- **`predict_alphaforge(fold, alpha_pool, combiner_kwargs=None, label_horizon=21, ret_col="target_magnitude")`**: Calls `combiner.precompute(...)` once per fold, then `fit_and_predict(...)` date by date; the Mega-Alpha is the score. For each test date `d` the combiner only uses history up to `label_horizon + 1` dates earlier, so every label it fits on (21-day forward return) has already realized by `d`. Default `max_active_factors` is 10 here. `fold.train_df` already has its own gap trimmed, so the first test dates of a fold see slightly staler history than strictly necessary. Takes an alpha pool instead of `(features, model_params)`, so it has its own branch in `walk_forward_cv`.
- **`class ModelSpec(name, predict_fn, eval_target_col)`** and **`MODEL_REGISTRY`**: `xgboost_ndcg`, `xgboost_mse`, `linear`, `lstm_mse`, all scored on `target_magnitude`.
- **`build_mega_combiner(...)`**: Constructs an `AlphaForgeCombiner` with given hyperparameters (default `max_active_factors=10`).
- **`mine_gp_factors(df, features, initial_train_end, n_components=15, target_col="target_magnitude")`**: **Not recommended**, see the [gplearn section](#gplearn-factor-mining-not-recommended). Fits a `SymbolicTransformer` (20 generations, population 1,000, Pearson metric, `max_samples=0.5`) on dates before `initial_train_end`, once, and adds `GP_Alpha_*` columns to the full frame (`walk_forward_cv(use_gp=True)`). The fit target is the per-date centered percentile rank of `target_col`. `walk_forward_cv` sets `initial_train_end` 21 dates before the first test date, so no fit *label* reaches into fold 1; that does not remove the selection bias described below.
- **`score_and_evaluate(...)`**, **`_daily_precision_at_k(...)`**, **`_daily_topk_excess_return(...)`**: Per-fold NDCG, precision@k (against the realised top-k by forward return) and top-k excess return against the universe.

### `walk_forward_cv(df, features, model="xgboost_ndcg", model_params=None, initial_train_dates=252, train_percent=0.7, gap_dates=21, window_mode="expanding", use_gp=False, icir_filter=False, icir_threshold=0.02, icir_target_col="next_1m_ret", corr_prune=False, corr_threshold=0.75, liquidity_filter=False, adequate_adtv=2_500_000, nonskip_features=None, top_k=20, alphaforge_kwargs=None)`

Per fold: optional IC/IR filter, then optional correlation pruning (both on `fold.train_df` only), append any `nonskip_features`, fit and score the chosen model, record NDCG. `window_mode` is `"expanding"` or `"rolling"`. Returns `(final_df, fold_ndcg_df)`: every fold's scored test rows with `pred_score`, and one row per fold (`fold_index`, `test_start`, `test_end`, `ndcg`). It also prints overall NDCG, precision@k and top-k excess return.

- `model="alphaforge"` uses every `WQ_Alpha_*` column in the frame (plus GP columns if `use_gp=True`) and ignores `features`, `icir_filter` and `corr_prune`. `alphaforge_kwargs` is forwarded to `AlphaForgeCombiner`.
- `use_gp=True` triggers the one-time gplearn mining step. Leave it off.

Example rolling setup: `window_mode="rolling", initial_train_dates=756, train_percent=0.8, gap_dates=21` gives 735 training dates, a 21-date gap and 184-date test blocks.

---

## `simulation.py`

Day-by-day backtest engine. Prices come from `df_raw`, scores from `df_predict`, and an optional third frame feeds the entry filter.

- **`class DataEngine(df)`**: Cursor-based access to a frame: `get_today`, `get_date`, `get_last(symbol)`, `get_prev(symbol)`, `get_upcoming(symbol)`, `update_idx`, `is_finished`, `reset`. `get_last` only returns data up to the cursor, which prevents look-ahead. `get_prev` returns the last row strictly before the cursor date, i.e. the reference price known when an evening order is placed.
- **`tick_size(price_vnd)`** and **`ceiling_price(ref, unit=1.0, limit=0.07)`**: HOSE tick size (10 / 50 / 100 VND below 10,000 / below 50,000 / above) and the next day's ceiling, `ref × 1.07` rounded down to the tick. `unit` is VND per price unit in the data (1000 if `close` is in thousand VND). The result is only exact when `ref` is an unadjusted close.
- **`class pack`** and **`class Stock`**: Share lots that are still settling. `add_pack`, `next_day`, `reduce_shares`.
- **`class Portfolio(initial)`**: `fee = 0.1%` per side, `tax = 0.1%` on sells, settlement lag `T = 3`. `execute_buy` and `execute_sell` (sale proceeds go to a pending-cash list), `next_day` (settles cash, marks to market, appends to `history` and `holdings_history`). Failed orders are logged in `skipped_orders`.
- **`class DataMismatch`**: raised when the frames do not cover the same dates.
- **Filters:**
  - `StepFilter(multiplier)`: constant multiplier.
  - `RampFilter(column, floor, threshold, min_value=0.0, max_value=1)`: linear ramp on a column value; NaN or missing column means no effect.
  - `FilterGroup(condition=None, transformations=None)`: evaluates `condition` (a pandas-style expression string, run with `eval`, so use trusted strings only) on a row. If false the multiplier is 1.0; if true it is the product of the transformations. **P1** is `FilterGroup("market1m_ema21-market3m_ema63<0 and regime_bucket_monthly=='Q1'", [StepFilter(0)])`.
- **`build_data_engines(df_raw, df_predict, df_condition=None)`**: Trims everything to the prediction date range, checks that the date sets are identical (else `DataMismatch`) and returns three `DataEngine`s. Build once, reuse via `reset()`.
- **`class OrderManager(initial, topk, df_raw, df_predict, df_condition=None, regime_filter=None, price_unit=1.0)`**: `run_strategy(allocation_strategy="equal" | "rank_weighted")` loops over days; on every 21st day it selects the top-`topk` by `pred_score` and sells names that left the set (or trims overweight ones); three days later it buys the shortfall to target weight in 100-share lots. The filter multiplier scales that buy shortfall on the buy day, so a multiplier of 0 blocks new purchases but does not liquidate existing holdings.

    **Buy-order model (`_buying_phase`).** Buys mimic an order placed the evening before (T-1) that fills at the ATC close of T. Since row T of the features only contains data up to T-1, the signal is available at that time.
  - *Sizing* uses order-time information only: the T-1 close as price and the T-1 NAV (`history[-2]`).
  - *Buying power* is checked at the ceiling price (T-1 close × 1.07, rounded down to the tick), as brokers typically reserve cash for price-less orders at the ceiling. Cash is snapshotted once before the loop and reserved cumulatively across the batch, so later symbols see less cash. Pass `reserve_at_ceiling=False` to `_buying_phase` to reserve at the T-1 close instead.
  - *Fill* happens at the T close, and cash is debited at that price.
  - *Checks:* an assert fires if the ceiling is below the reference price (usually a wrong `price_unit`), and a buy whose T close is above the ceiling is skipped and logged in `skipped_orders` as `close_above_ceiling` (it signals dividend-adjusted prices).
  - `price_unit` is the VND-per-unit of the `close` column (1.0 for full VND, 1000.0 for thousand VND).

  **Sell-order model (`_selling_phase`).** Same convention as buys: trims of names that stay in the target set are sized from the T-1 close and T-1 NAV and fill at the T close. Names that left the set are sold in full (available shares only) at the T close, so no sizing applies. Pending, unsettled shares are never sold.

  `get_holdings(start, end)` and `get_symbols_held(start, end)` report positions after a run. No slippage is modelled and ATC orders (buys and sells) are assumed to fill in full.

- **`class NullHypothesisTest(OrderManager)`**: same engine, but picks `topk` stocks uniformly at random each rebalance, for Monte Carlo baselines. It adds its own `reset()` to rewind all engines.

---

## `evaluation.py`

Plots and performance statistics. No orchestration lives here.

- **`plot_feature_importances(model, features)`**: XGBoost gain importances.
- **`compute_model_ic(test_df, pred_col="pred_score", ret_col="next_1m_ret")`**: Per-date Spearman IC; reports mean, std and IR.
- **`compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col="next_1m_ret")`**: Win rate of top-quantile picks against the market.
- **`plot_feature_ic`**, **`plot_feature_ir`**, **`plot_feature_rolling_ir(test_df, feature, target_col, window=6)`**: Feature diagnostics.
- **`plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None)`**: Mean forward return per predicted quintile.
- **`plot_equity_curves(*results, labels=None, normalize=False, regime_colors=None, show_regime=False)`**: Overlays equity curves. With `show_regime=True` it shades `regime_bucket_monthly` regimes taken from the first frame, which therefore needs that column.
- **`print_performance_report(result, initial_capital=None, rf_annual=0.045, trading_days_per_year=252, verbose=True)`**: Returns (and optionally prints) total return, CAGR, Sharpe, Sortino, max drawdown and Calmar from a NAV history.

---

## `significance_test.py`

Statistical comparison of ranking models and portfolios. Target values are 21-day forward returns, so adjacent days share 20 of 21 days and daily differences are strongly autocorrelated; every test therefore resamples **blocks** of consecutive days (default 21). Differences are in volatility-adjusted `target_magnitude` units, not percent.

- **Daily series:** `topk_daily_avg(df, k=20, ...)` (mean target of the top-k by `pred_score` each day), `universe_daily_avg(df, ...)`, `compute_delta_base_vs_model(...)` (base minus other, positive means base is better), `compute_delta_vs_universe(...)`.
- **Core tests:** `paired_permutation_test(delta, n_iter, alternative, block, seed)` (block sign-flip) and `paired_bootstrap_ci(delta, n_boot, ci, block, seed)` (moving-block bootstrap).
- **`run_comparison(name, delta, block=None, step=None, offset=0, ...)`**: p-value at the chosen block (`p_value`), the naive block=1 p-value (`p_naive`) and a 95% CI. `step=21` keeps every 21st date for a non-overlapping cross-check.
- **`add_holm_correction(results, alpha=0.05)`**: Holm-adjusted p-values within one family of tests.
- **Families:** `compare_base_vs_models(dfs, base_model="ndcg", ...)` and `compare_models_vs_universe(dfs, ...)`.
- **Diagnostics:** `autocorr_report`, `outlier_report`, `suggest_block`, `plot_autocorr`, `offset_sensitivity(dfs, ..., step=21)`.
- **Output:** `print_table`, `plot_base_vs_models`, `plot_vs_universe`.
- **`run_analysis(dfs, base_model="ndcg", mode="both" | "base" | "universe", k=20, block=None, step=None, offset=0, save_dir=None, show=True)`**: One-call entry point returning the result tables.
- **Sharpe difference:** `sharpe_ratio(returns, ...)` and `sharpe_diff_block_bootstrap(history_1, history_2, block=21, n_boot=10000, rf_annual=0.0, date_col="date", value_col="total_value", seed=42)`: paired moving-block bootstrap on two NAV histories, same blocks for both series. Note `rf_annual` defaults to 0 here, while `print_performance_report` uses 4.5%.

The notebook `notebooks/Significance Test for Ranking Models.ipynb` shows the intended workflow (autocorrelation check first, then tests).

---

## `inference.py`

### `generate_paper_trade_signals(df, current_portfolio, features, use_mega=False, model=None, buy_n=30, trend_filter_col="dist_SMA_100", trend_filter_threshold=1.0, target_col="target_magnitude", icir_filter=False, icir_threshold=0.02, icir_target_col="next_1m_ret", corr_prune=False, corr_threshold=0.75, nonskip_features=None)`

1. `latest_date = df["date"].max()`. Training rows are the `date < latest_date` rows from the most recent **735** trading dates that have a non-null `target_col`; inference rows are those on `latest_date`.
2. Optional IC/IR filter and correlation pruning on the training rows only. `nonskip_features` are then appended to the feature list unconditionally (only when a new model is trained).
3. If `model is None`, fits `xgb.XGBRanker(**BASE_MODEL_PARAMS)` on `target_col` with a `qid` built from `date` inside the function, and scores the inference rows into `live_score`. `use_mega=True` raises `NotImplementedError`. If a pretrained `model` (an `XGBRanker`) is passed instead, nothing is trained: its own feature names (`model.get_booster().feature_names`) replace `features`, and combining it with `icir_filter` or `corr_prune` raises `ValueError`, as does a model that has no feature names.
4. Ranks today's stocks (1 = best). Held symbols stay in `hold_list` while ranked within `buy_n`, otherwise go to `sell_list`; held symbols absent from today's data go to `not_in_universe_list`. New buys come from the top `buy_n`, optionally gated by `trend_filter_col > trend_filter_threshold` (pass `trend_filter_col=None` to disable, as production does).

Returns `(buy_list, hold_list, sell_list, not_in_universe_list, ranked_today)`; `ranked_today` has `Symbol`, `live_score`, `rank`. Note the defaults (`buy_n=30`, trend filter on via `dist_SMA_100`) differ from what production passes (`buy_n=20`, `trend_filter_col=None`, `corr_prune=True`); the default `target_col` is already `target_magnitude`, the same as production.

### `get_actionable_portfolio_lists(df, current_portfolio, features, **kwargs)`

Wrapper returning a dict with `BUY`, `HOLD`, `SELL`, `NOT_IN_UNIVERSE`, `Rankings`.

---

## `deep_combiner.py` (experimental)

- **`_rolling_rank_ic(factor_series, ret_series, window)`**: Legacy helper, no longer used by the combiner.
- **`class AlphaForgeCombiner(ic_window=40, ic_threshold=0.02, icir_threshold=0.2, max_active_factors=10, ridge_alpha=1.0)`**: Implements Algorithm 2 of AlphaForge (arXiv:2406.18394). Two-step API:
  1. `precompute(all_df, alpha_cols, ret_col)`, once per fold: ranks factors and the label within each date (percentile, centered later) and caches the per-date rank-IC table (dates × factors).
  2. `fit_and_predict(current_df, last_date)`, once per test date: reads rolling RankIC/ICIR from the cache over the last `ic_window` dates up to `last_date` (the last date whose label has realized), keeps factors with |RankIC| ≥ `ic_threshold` and |ICIR| ≥ `icir_threshold`, takes the top `max_active_factors` by |ICIR| (if none pass, falls back to the single factor with the highest |ICIR|), flips each factor's sign to its IC sign, and fits Ridge (no intercept) on the **per-date centered percentile ranks** of the factors and the label over the same window. The current date's factors are ranked the same way and combined into the "Mega-Alpha" score.

  `report()` prints the active factors and weights. The caller must pass only history whose labels have realized (`predict_alphaforge` does this). Earlier results that showed it underperforming the XGBoost ranker came from before this fix and should be rerun before being quoted.

---

## `app.py`

A deliberately small Streamlit page, written in English.

- **`_today_vn()`**: Today's date in Vietnam time, used as a cache key so the cache refreshes after local midnight.
- **`load_today_signals(_date_key)`**: Downloads `today_signals.parquet` from the Hugging Face dataset `PhongHPham/vn_cross_sectional_ranking_data_storage` (1-hour `st.cache_data` TTL), using `st.secrets["HF_TOKEN"]` if present. Shows a warning and returns `None` on failure.
- **Page:** browser title "Scientific Research"; the paper's title in English ("Application of a Machine Learning Ranking Model Combined with an Entry-Point Filter on the Vietnamese Stock Market") and a research-only caption; a caption with the data cutoff date (`signal_date`); then either a "Hold all cash, do not buy today / P1 filter is active" warning when `filter_active` is true, or a table of stock ticker and rank with a fixed "5%" capital allocation per stock.

It performs no model work; it trusts `kernel.py` to have applied P1 and Top-K already.

---

## `kaggle_kernel/kernel.py`

The production entry point, pushed to Kaggle by `precompute_model.yml` (`kernel-metadata.json`: id `phmhuyphongp/precompute-vn`, script kernel, GPU and internet on, private). The file ships with the placeholders `__HF_TOKEN__` and `__GH_PAT__`; `precompute_model.yml` substitutes the `HF_TOKEN` and `GH_PAT` GitHub secrets into them (and fails if a placeholder is missing or a secret is empty) just before `kaggle kernels push`. Never commit the file with real tokens.

0. Guard: if either value still starts with the placeholder prefix, exit with "Secrets were not injected by the workflow."
1. Clone `main` (source) using `GH_PAT`, then reset the remote URL so the token is not stored in the clone's git config; download `market_data.parquet` from the Hugging Face dataset into `repo/data/`; `pip install -r requirements-dev.txt`.
2. Load the data, keep `config.usedSymbols`, drop 2018-01-23 and 2018-01-24 (days where most stocks are missing).
3. Build features twice with `adtv_limit=2_500_000`: with targets (training) and with `generate_target=False` (keeps today).
4. Evaluate P1 on the latest date from the second frame: `regime_bucket_monthly == "Q1"` and `market1m_ema21 - market3m_ema63 < 0`.
   - **Active:** write a one-row cash record (`filter_active=True`, null `rank`/`Symbol`/`live_score`).
   - **Inactive:** sort the training frame by date and assign `qid`, append today's rows with `target_magnitude = NaN`, call `generate_paper_trade_signals(buy_n=20, trend_filter_col=None, target_col="target_magnitude", icir_filter=False, corr_prune=True)`, keep the top 20.
5. Save `today_signals.parquet` (`signal_date`, `filter_active`, and `rank`/`Symbol`/`live_score` when not in cash) and upload it to the Hugging Face dataset that `app.py` reads.

Because `generate_paper_trade_signals` trains on the last 735 labelled dates, the model never sees the ~21 most recent sessions (their targets do not exist yet). The production path uses no gplearn factors.

---

## gplearn factor mining (not recommended)

**Recommendation: do not use `walk_forward_cv(use_gp=True)`, `mine_gp_factors`, `add_and_filter_alphas` or `rank_ic_fitness`.** The code is kept for reference and reproducing old experiments only. Production does not use it, so any backtest that does cannot be matched by the live pipeline.

**Main reasons (the author's concerns):**

1. **Selection bias.** `SymbolicTransformer` evolves 20 generations of 1,000 programs and keeps the best `n_components` (15) by in-sample fitness. Choosing the winners out of ~20,000 candidate expressions is a large multiple-testing search with no deflation or independent validation, so the survivors are mostly the expressions that fit noise in the initial window.
2. **Fitted once at the start.** `mine_gp_factors` runs a single time on the initial training window (dates before `initial_train_end`) and the expressions are frozen for every later fold. They are never re-mined, so they go stale in a long walk-forward, unlike per-fold feature selection (`icir_filter`, `corr_prune`) and the date-by-date AlphaForge combiner.

**Further problems found in the code:**

3. **In-sample optimism feeds the models.** The GP factors score far higher inside the window they were fitted on than afterwards. In fold 1 (and the next folds, whose training windows still contain that window) XGBoost sees factors whose training-time quality is inflated and learns to over-trust them. With `model="alphaforge"` the pool is `WQ_Alpha_* + GP_Alpha_*`, and the rolling IC/ICIR of the GP columns is also inflated for early test dates, so they are over-weighted there.
4. **Objective mismatch.** `metric="pearson"` is pooled across all dates on raw, unnormalised features (different scales for returns, ratios, RSI, ...), while evaluation uses per-date rank ordering. The pooled fit can be driven by differences between dates rather than by the cross-section, and protected operators (division, log, sqrt) create discontinuities.
5. **Overlapping labels and row subsampling.** Daily rows carry 21-day forward targets that overlap by 20/21, so the effective sample is much smaller than the row count suggests. `max_samples=0.5` samples random rows, not date blocks, so neighbouring rows land on both sides of the split and the "out-of-bag" fitness is not truly out-of-sample.
6. **Pruning and selection ignore GP columns.** `GP_Alpha_*` are in no `FEATURE_GROUPS` group, so `corr_prune` treats each as a singleton and never removes near-duplicates among them or against base features. `icir_filter` can include them without any extra penalty.
7. **`rank_ic_fitness` is not wired up.** It is a plain function (not wrapped with gplearn's `make_fitness`), computes a pooled Spearman rather than the per-date IC the rest of the repo uses, and nothing calls it.
8. **`add_and_filter_alphas` is legacy.** It removes only exactly identical columns (`np.unique`), so near-duplicates survive; it fills NaN/inf with 0 before transforming; the printed message hard-codes "from 10 candidates"; and it names columns `Mega_Alpha_i`, a different scheme from `mine_gp_factors` (`GP_Alpha_i`) and easy to confuse with the AlphaForge `Mega_Alpha` score.
9. **NaN handling.** `mine_gp_factors` applies `np.nan_to_num` (NaN becomes 0) to raw features both for fitting and for transforming the full frame, silently turning missing values into a real feature value.
10. **Cost and reproducibility.** The search is expensive (population 1,000 × 20 generations on a multi-year panel), and results depend on `random_state` and the gplearn version, which makes them hard to audit.

If you want learned alphas, prefer fixed, documented formulas (as in `alpha_mining.WorldQuantAlphas`) or a procedure that re-mines inside every fold on that fold's training window only and judges candidates on a held-out later window.

---

## Known issues

1. **Notebooks use the previous walk-forward API** (`initial_train_months`, `test_months`), which `walk_forward_cv` and `generate_folds` no longer accept.
2. **Different Sharpe conventions:** `print_performance_report` uses `rf_annual=0.045`; `sharpe_diff_block_bootstrap` defaults to 0.
3. **Order assumptions in `simulation.py`:** ATC buys always fill in full, although in reality a close at the ceiling often leaves buyers unfilled. The same applies to sells: a close at the floor can leave sellers unfilled. Reserving cash at the ceiling is an assumption about broker behaviour and should be checked against the real broker. The ceiling needs unadjusted closes, so dividend-adjusted prices can trigger the `close_above_ceiling` skip.
4. **AlphaForge history staleness:** at the start of each test fold the combiner's history is about 21 dates staler than needed, because `fold.train_df` is already gap-trimmed before `predict_alphaforge` applies its own label-horizon cutoff. It costs a little signal freshness and does not leak.
5. **gplearn:** see the [dedicated section](#gplearn-factor-mining-not-recommended). Selection bias, a single fit at the start, in-sample optimism in early folds, and no pruning of GP columns.
6. **Stale comments in `config.py`:** the `WQ_Alpha_006` comment says a 10-day correlation and `WQ_Alpha_028` says a 5-day one, but both code paths use 21 days; the `WQ_Alpha_103` comment calls it a value proxy, while the code documents it as a 52-week-high (momentum-type) signal. The code is authoritative.
7. **`data_collect.py` date logic:** the "last completed trading day" roll-back is weekday-based. On Sunday it resolves to Saturday (not Friday) and Vietnamese market holidays are not considered, so some symbols may be re-fetched needlessly. One warning log in `_fetch_quote` has a format string with two placeholders but one argument, which prints a logging error instead of the message.