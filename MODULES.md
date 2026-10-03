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
14. [Known issues](#known-issues)

---

## Data flow

```
data_collect.py ─► market_data.parquet, vnindex_data.parquet
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

---

## `config.py`

Constants plus one helper. Imported by `models.py`, `inference.py` and `kernel.py`.

| Name | Description |
|---|---|
| `candidate_features` | Feature pool passed to `walk_forward_cv` and `generate_paper_trade_signals` (42 names): returns, volatility, SMA/EMA distances, volume, RSI, `dist_52w_high`, `log_ret_skip1m`, and 14 `WQ_Alpha_*` columns. Four structural features are commented out. See [Known issues](#known-issues): 7 names are not produced by `build_features`. |
| `final_features` | Legacy hand-picked list of 20 features. Not used. |
| `BASE_MODEL_PARAMS` | Default XGBoost parameters: `rank:ndcg`, `device='cuda'`, `tree_method='hist'`, 150 trees, depth 3, `min_child_weight=5`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.6`, `reg_lambda=5`, `reg_alpha=1`, `lambdarank_pair_method='topk'`, `lambdarank_num_pair_per_sample=20`, `ndcg_exp_gain=False`, `random_state=42`. |
| `FEATURE_GROUPS` | `{group: [features]}` for `returns`, `volatility`, `moving_average`, `volume`, `rsi`, `wq_features`, `price_structure`. Correlation pruning only compares features **within** a group. |
| `ALL_GROUP_NAMES` | `list(FEATURE_GROUPS)`. |
| `usedSymbols` | The 280 tickers that `kernel.py` filters the market data down to. |
| `groups_to_features(group_names)` | Flattens group names into one feature list. |

---

## `data_collect.py`

Builds the ticker universe and keeps `market_data.parquet` and `vnindex_data.parquet` current through `vnstock`. Running the file as a script (`python src/data_collect.py`) updates both files from `2016-01-01`; this is what `daily_update.yml` runs. It also writes `data_collect.log`.

- **`clean_symbols(symbol_list)`**: Strips, de-duplicates and sorts ticker strings, dropping nulls.
- **`get_tags(fetching=False)`**: Returns a hard-coded list of 300 tickers. With `fetching=True` returns the live `VNALL` index members from `vnstock.Reference`.
- **`clean_ohlcv(df)`**: Normalises a raw OHLCV frame. Rows that break an OHLC rule are repaired by carrying the whole candle forward; volume is left alone and no rows are dropped.
- **`_fetch_quote(symbol, start_d, end_d)`**: Daily OHLCV for one symbol. Returns `None` on failure.
- **`update_market_data(file_path, symbols, start_date="2016-01-01", batch_size=1, base_sleep=7, filter_symbols=True)`**: Main entry point. Works out the last completed trading day (today counts after 14:45 Vietnam time), skips symbols whose stored data already reaches it, and for every other symbol re-fetches from `start_date` and merges by `(date, Symbol)`, new rows winning. Saves after each batch, sleeps `base_sleep` seconds per symbol, and retries once when the API reports a rate limit. `filter_symbols=True` purges stored symbols that are no longer in `symbols`.
- **`_drop_before_last_zero_volume(df)`**: For each symbol, drops everything up to and including its most recent zero-volume day (pre-listing and halt noise).
- **`_merge_and_dedup(existing, new_chunks)`**: Concatenates and de-duplicates on `(date, Symbol)`.
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
| `price_structure(df)` | `dist_52w_high`, `log_ret_skip1m`. **Currently not called** by `build_features` (see Known issues) |
| `turnover_ratio(df)`, `limit_bias(df, d=60)`, `amihud_illiquidity(df)` | Structural features. Computed, but their candidate entries in `config.py` are commented out |
| `adtv(df, window=21)` | 21-day average daily traded value, used for the liquidity mask |
| `market_breadth(df, liquidity_mask=None)` | Breadth columns such as `breadth_ema21` |
| `robust_market_regime_pipeline_monthly(df, return_horizon=21, rolling_window=63, num_buckets=4, liquidity_mask=None)` | The **P2** input. Takes each stock's lagged 21-day return, computes the cross-sectional MAD per date over liquid rows, smooths it with a 63-day rolling mean, converts it to a point-in-time percentile over a trailing 252-day window, and cuts it into quartiles. Outputs `regime_disp_raw_monthly`, `regime_percentile_monthly`, `regime_bucket_monthly` (`Q1` lowest dispersion to `Q4` highest) |

### Target and label functions

- **`build_targets(df)`**: `next_1m_ret = log(close[t+21]/close[t])` and `next_1w_ret` (5-day), then **drops rows where either is NaN**, i.e. the last ~21 sessions per symbol.
- **`make_magnitude_label(df, target_col="risk_adj_ret", scale_max=100, winsor_pct=0.0)`**: Per-date min-max scaling to `[0, 100]`, producing `target_magnitude`. This is the training and evaluation label for every model in `MODEL_REGISTRY`.
- **`make_adaptive_bucket_label(...)`**: Per-date bucket labels (`target_bucket`) with a bucket count chosen to give ~20 stocks each. **Not called** by `build_features` any more.
- **`target_generating_ranking(df, freq="M")`**: Builds `risk_adj_ret`, quintile labels `target_quintile`, and `qid` (one id per date, required by `XGBRanker`). **Not called** by `build_features`; the `alphaforge` model path scores on `target_quintile`, so it needs this run first.
- **`apply_cross_sectional_ranking(df, feature_cols)`**: Adds per-date percentile-rank columns `csr_<feature>`. Exploratory only.

### `build_features(df, min_stocks_per_date=50, adtv_limit=None, generate_target=True)`

Order of work: returns, volatility, MAs, volume, RSI, volume quality, structural features; then (if `generate_target`) `build_targets`, `risk_adj_ret = next_1m_ret / volatility_3m`, `make_magnitude_label`; then ADTV and the `is_liquid` mask; WQ alphas; the regime pipeline; `market3m_ema63` and `market1m_ema21` (the **P3** inputs: EMA span 63 of the liquid-universe mean `log_ret_3m`, EMA span 21 of the mean `log_ret_1m`); breadth. Finally it drops illiquid rows (once), replaces ±inf with NaN, drops rows with any NaN feature, and removes dates with fewer than `min_stocks_per_date` symbols.

- `generate_target=False` skips target construction, so the latest date survives. The regime and trend columns are produced either way.
- With `adtv_limit` set, per-symbol rolling windows still run on the full series; only cross-sectional statistics use the mask.

---

## `alpha_mining.py`

WorldQuant-style alphas and gplearn helpers.

- **Operators:** `to_series`, `ts_delay`, `ts_delta`, `ts_mean`, `ts_min`, `ts_rank`, `ts_argmax` (per-symbol, rolling) and `cs_mask`, `cs_rank`, `cs_mean` (per-date, liquidity-mask aware).
- **`class WorldQuantAlphas(df, liquidity_mask=None)`**: Builds a lagged OHLCV copy so every alpha respects the T-1 rule. `generate_all()` currently returns `WQ_Alpha_002, 006, 007, 013, 016, 024, 028, 040, 102, 103, 104, 105`. Methods `get_alpha_201` and `get_alpha_202` exist, but their calls in `generate_all()` are commented out.
- **`add_and_filter_alphas(gp_model, original_df, input_features)`**: Takes a fitted `gplearn` `SymbolicTransformer`, generates candidate expressions, filters them and merges the survivors into `original_df`.
- **`rank_ic_fitness(y_true, y_pred)`**: Custom gplearn fitness, the Spearman rank IC between a candidate and forward returns.

---

## `models.py`

Feature selection, walk-forward machinery and the model zoo.

### Feature selection (training window only)

- **`select_features_by_icir(df, candidate_features, icir_threshold=0.02, target_col="next_1m_ret")`**: Vectorised per-date Spearman IC for each feature, aggregated to IR = mean/std. Returns `(selected, ic_ir_map)` keeping features with |IC IR| above the threshold.
- **`_compute_icir_map(df, features, target_col)`**: Same computation without thresholding. Used for tie-breaking when `corr_prune=True` and `icir_filter=False`.
- **`prune_correlated_features(df, candidate_features, ic_ir_map, feature_groups=None, correlation_threshold=0.75)`**: Within each `FEATURE_GROUPS` group, for any pair with |Spearman| at or above the threshold, drops the one with the lower |IC IR|. Features in different groups never compete.
- **`test_train_spliter(df, test_start, features)`**: Single split helper for ad-hoc work.

### Walk-forward windows

All lengths are counts of **unique trading dates** in the data.

- **`class Fold`**: dataclass with `fold_number`, `train_df`, `test_df`, `test_start`, `test_end`.
- **`class WindowPolicy(initial_train_dates, gap_dates=0)`**: abstract base. `iter_slices(unique_dates)` yields `(train_dates, test_dates)` pairs. Raises if `gap_dates >= initial_train_dates`.
- **`class RollingWindow(initial_train_dates, train_percent, gap_dates=0)`**: fixed-length train window. Test length is `round((initial_train_dates - gap_dates) × (1 - train_percent) / train_percent)`, and the window advances by exactly the test length, so test blocks tile the timeline with no overlap and no holes. The last incomplete fold is dropped.
- **`class ExpandingWindow(initial_train_dates, test_len_dates=None, train_percent=None, gap_dates=0)`**: train starts at date 0 and grows by one test block per fold. Test length is fixed for the whole run, given directly or derived once from `train_percent`.
- **`generate_folds(df, window_policy, liquidity_filter=False, adequate_adtv=2_500_000)`**: Turns a policy's date slices into `Fold` objects and assigns a per-date `qid` to each train and test frame. With `liquidity_filter=True` each fold is filtered separately by `_filter_fold_by_liquidity`.
- **`_filter_fold_by_liquidity(...)`**: ADTV filter applied per fold, using only history available at each date.
- **`_test_len_from_ratio(eff_train_dates, train_percent)`**: shared test-length formula.

### Model zoo

Each `predict_*` takes a `Fold` and returns the test frame with a `pred_score` column.

- **`predict_xgboost_ndcg(fold, features, model_params)`**: **Production.** `XGBRanker` (`rank:ndcg`) fit on `target_magnitude` with `qid` groups. Defaults to `BASE_MODEL_PARAMS`.
- **`predict_xgboost_mse(...)`**: `XGBRegressor` with `reg:squarederror`, same label.
- **`predict_linear(...)`**: `StandardScaler` plus `LassoCV(cv=5)`, same label. (Called "linear" in the registry and "LassoCV" in the notebook.)
- **`predict_lstm_mse(...)`**: Small PyTorch LSTM over per-symbol windows, MSE loss. `model_params` may set `seq_len` (10), `hidden_size` (32), `num_layers` (1), `dropout` (0), `epochs` (20), `batch_size` (256), `lr` (1e-3), `device`. Symbols without enough history get the median training prediction.
- **`predict_alphaforge(fold, alpha_pool, combiner_kwargs=None)`**: Runs `AlphaForgeCombiner` date by date; the Mega-Alpha is the score. Takes an alpha pool instead of `(features, model_params)`, so it has its own branch in `walk_forward_cv`.
- **`class ModelSpec(name, predict_fn, eval_target_col)`** and **`MODEL_REGISTRY`**: `xgboost_ndcg`, `xgboost_mse`, `linear`, `lstm_mse`, all scored on `target_magnitude`. (The comment above the registry still talks about `target_bucket`/`target_quintile`; ignore it.)
- **`build_mega_combiner(...)`**: Constructs an `AlphaForgeCombiner` with given hyperparameters.
- **`mine_gp_factors(df, features, initial_train_end)`**: Fits a `SymbolicTransformer` on the initial window and adds GP columns to the full frame (`walk_forward_cv(use_gp=True)`).
- **`score_and_evaluate(...)`**, **`_daily_precision_at_k(...)`**, **`_daily_topk_excess_return(...)`**: Per-fold NDCG, precision@k and top-k excess return against the universe.

### `walk_forward_cv(df, features, model="xgboost_ndcg", model_params=None, initial_train_dates=252, train_percent=0.7, gap_dates=21, window_mode="expanding", use_gp=False, icir_filter=False, icir_threshold=0.02, icir_target_col="next_1m_ret", corr_prune=False, corr_threshold=0.75, liquidity_filter=False, adequate_adtv=2_500_000, nonskip_features=None, top_k=20)`

Per fold: optional IC/IR filter, then optional correlation pruning (both on `fold.train_df` only), append any `nonskip_features`, fit and score the chosen model, record NDCG. `window_mode` is `"expanding"` or `"rolling"`. Returns `(final_df, fold_ndcg_df)`: every fold's scored test rows with `pred_score`, and one row per fold (`fold_index`, `test_start`, `test_end`, `ndcg`). It also prints overall NDCG, precision@k and top-k excess return.

Example rolling setup: `window_mode="rolling", initial_train_dates=756, train_percent=0.8, gap_dates=21` gives 735 training dates, a 21-date gap and 184-date test blocks.

---

## `simulation.py`

Day-by-day backtest engine. Prices come from `df_raw`, scores from `df_predict`, and an optional third frame feeds the entry filter.

- **`class DataEngine(df)`**: Cursor-based access to a frame: `get_today`, `get_date`, `get_last(symbol)`, `get_upcoming(symbol)`, `update_idx`, `is_finished`, `reset`. `get_last` only returns data up to the cursor, which prevents look-ahead.
- **`class pack`** and **`class Stock`**: Share lots that are still settling. `add_pack`, `next_day`, `reduce_shares`.
- **`class Portfolio(initial)`**: `fee = 0.1%` per side, `tax = 0.1%` on sells, settlement lag `T = 3`. `execute_buy` and `execute_sell` (sale proceeds go to a pending-cash list), `next_day` (settles cash, marks to market, appends to `history` and `holdings_history`). Failed orders are logged in `skipped_orders`.
- **`class DataMismatch`**: raised when the frames do not cover the same dates.
- **Filters:**
  - `StepFilter(multiplier)`: constant multiplier.
  - `RampFilter(column, floor, threshold, min_value=0.0, max_value=1)`: linear ramp on a column value; NaN or missing column means no effect.
  - `FilterGroup(condition=None, transformations=None)`: evaluates `condition` (a pandas-style expression string, run with `eval`, so use trusted strings only) on a row. If false the multiplier is 1.0; if true it is the product of the transformations. **P1** is `FilterGroup("market1m_ema21-market3m_ema63<0 and regime_bucket_monthly=='Q1'", [StepFilter(0)])`.
- **`build_data_engines(df_raw, df_predict, df_condition=None)`**: Trims everything to the prediction date range, checks that the date sets are identical (else `DataMismatch`) and returns three `DataEngine`s. Build once, reuse via `reset()`.
- **`class OrderManager(initial, topk, df_raw, df_predict, df_condition=None, regime_filter=None)`**: `run_strategy(allocation_strategy="equal" | "rank_weighted")` loops over days; on every 21st day it selects the top-`topk` by `pred_score` and sells names that left the set (or trims overweight ones); three days later it buys the shortfall to target weight in 100-share lots. The filter multiplier scales that buy shortfall on the buy day, so a multiplier of 0 blocks new purchases but does not liquidate existing holdings. `get_holdings(start, end)` and `get_symbols_held(start, end)` report positions after a run. No slippage is modelled.
- **`class NullHypothesisTest(OrderManager)`**: same engine, but picks `topk` stocks uniformly at random each rebalance, for Monte Carlo baselines.

---

## `evaluation.py`

Plots and performance statistics. No orchestration lives here.

- **`plot_feature_importances(model, features)`**: XGBoost gain importances.
- **`compute_model_ic(test_df, pred_col="pred_score", ret_col="next_1m_ret")`**: Per-date Spearman IC; reports mean, std and IR.
- **`compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col="next_1m_ret")`**: Win rate of top-quantile picks against the market.
- **`plot_feature_ic`**, **`plot_feature_ir`**, **`plot_feature_rolling_ir(test_df, feature, target_col, window=6)`**: Feature diagnostics.
- **`plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None)`**: Mean forward return per predicted quintile.
- **`plot_equity_curves(*results, labels=None, normalize=False, regime_colors=None, show_regime=False)`**: Overlays equity curves, optionally shading regime buckets.
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

### `generate_paper_trade_signals(df, current_portfolio, features, use_mega=False, model=None, buy_n=30, trend_filter_col="dist_SMA_100", trend_filter_threshold=1.0, target_col="target_quintile", icir_filter=False, icir_threshold=0.02, icir_target_col="next_1m_ret", corr_prune=False, corr_threshold=0.75)`

1. `latest_date = df["date"].max()`. Training rows are the `date < latest_date` rows from the most recent **735** trading dates that have a non-null `target_col`; inference rows are those on `latest_date`.
2. Optional IC/IR filter and correlation pruning on the training rows only.
3. If `model is None`, fits `xgb.XGBRanker(**BASE_MODEL_PARAMS)` on `target_col` with the `qid` column (so `df` needs `qid`) and scores the inference rows into `live_score`. `use_mega=True` raises `NotImplementedError`.
4. Ranks today's stocks (1 = best). Held symbols stay in `hold_list` while ranked within `buy_n`, otherwise go to `sell_list`; held symbols absent from today's data go to `not_in_universe_list`. New buys come from the top `buy_n`, optionally gated by `trend_filter_col > trend_filter_threshold` (pass `trend_filter_col=None` to disable, as production does).

Returns `(buy_list, hold_list, sell_list, not_in_universe_list, ranked_today)`; `ranked_today` has `Symbol`, `live_score`, `rank`. Note the defaults (`buy_n=30`, trend filter on, `target_quintile`) differ from what production passes (`buy_n=20`, no trend filter, `target_magnitude`).

### `get_actionable_portfolio_lists(df, current_portfolio, features, **kwargs)`

Wrapper returning a dict with `BUY`, `HOLD`, `SELL`, `NOT_IN_UNIVERSE`, `Rankings`.

---

## `deep_combiner.py` (experimental)

- **`_rolling_rank_ic(factor_series, ret_series, window)`**: Rolling cross-sectional Spearman IC and ICIR.
- **`class AlphaForgeCombiner(ic_window=40, ic_threshold=0.02, icir_threshold=0.2, max_active_factors=13, ridge_alpha=1.0)`**: Implements Algorithm 2 of AlphaForge (arXiv:2406.18394). `fit_and_predict(hist_df, current_df, alpha_cols, ret_col="next_1m_ret")` computes rolling RankIC/ICIR per factor over recent history, keeps factors above the thresholds, takes the top `max_active_factors` by |RankIC|, fits Ridge to get weights, and returns a combined "Mega-Alpha" score for the current date. `report()` prints the active factors and weights. Reported in the project as underperforming the XGBoost ranker; not used in production.

---

## `app.py`

A deliberately small Streamlit page.

- **`_today_vn()`**: Today's date in Vietnam time, used as a cache key so the cache refreshes after local midnight.
- **`load_today_signals(_date_key)`**: Downloads `today_signals.parquet` from the Hugging Face dataset `PhongHPham/vn_cross_sectional_ranking_data_storage` (1-hour `st.cache_data` TTL), using `st.secrets["HF_TOKEN"]` if present. Shows a warning and returns `None` on failure.
- **Page:** Vietnamese title (the paper's title) and a research-only disclaimer; a caption with the data cutoff date (`signal_date`); then either a "hold all cash, P1 active" warning when `filter_active` is true, or a table of rank and symbol with a fixed "5%" allocation per stock.

It performs no model work; it trusts `kernel.py` to have applied P1 and Top-K already.

---

## `kaggle_kernel/kernel.py`

The production entry point, pushed to Kaggle by `precompute_model.yml` (`kernel-metadata.json`: id `phmhuyphongp/precompute-vn`, script kernel, GPU and internet on, private). `GH_PAT` and `HF_TOKEN` are injected by the workflow with `sed`, then restored.

1. Clone `main` (source) and `data-storage` (data) with `GH_PAT`; copy `market_data.parquet` into `repo/data/`; `pip install -r requirements-dev.txt`.
2. Load the data, keep `config.usedSymbols`, drop 2018-01-23 and 2018-01-24 (days where most stocks are missing).
3. Build features twice with `adtv_limit=2_500_000`: with targets (training) and with `generate_target=False` (keeps today).
4. Evaluate P1 on the latest date from the second frame: `regime_bucket_monthly == "Q1"` and `market1m_ema21 - market3m_ema63 < 0`.
   - **Active:** write a one-row cash record (`filter_active=True`, null `rank`/`Symbol`/`live_score`).
   - **Inactive:** sort the training frame by date and assign `qid`, append today's rows with `target_magnitude = NaN`, call `generate_paper_trade_signals(buy_n=20, trend_filter_col=None, target_col="target_magnitude", icir_filter=False, corr_prune=True)`, keep the top 20.
5. Save `today_signals.parquet` (`signal_date`, `filter_active`, and `rank`/`Symbol`/`live_score` when not in cash) and upload it to the Hugging Face dataset that `app.py` reads.

Because `generate_paper_trade_signals` trains on the last 735 labelled dates, the model never sees the ~21 most recent sessions (their targets do not exist yet).

---

## Known issues

Also listed in the [README](README.md#known-issues).

1. **Feature mismatch.** `config.candidate_features` includes `dist_52w_high`, `log_ret_skip1m`, `WQ_Alpha_001`, `WQ_Alpha_101`, `WQ_Alpha_200`, `WQ_Alpha_201` and `WQ_Alpha_202`, which `build_features` does not produce (verified on synthetic data). `WQ_Alpha_102`, `104` and `105` are produced but not listed. Selecting columns from the `build_features` output with `candidate_features` raises `KeyError`. Fix by re-enabling `price_structure(df)` and the missing alphas, or by updating `candidate_features` and `FEATURE_GROUPS`. The comments in `config.py` for alpha 103 (monthly acceleration) and in `alpha_mining.py` (103 is 12-1m momentum, 102 is acceleration) also disagree.
2. **Notebooks use the previous walk-forward API** (`initial_train_months`, `test_months`), which `walk_forward_cv` and `generate_folds` no longer accept.
3. **`daily_update.yml` has no schedule**, only `workflow_dispatch`.
4. **Different Sharpe conventions:** `print_performance_report` uses `rf_annual=0.045`; `sharpe_diff_block_bootstrap` defaults to 0.