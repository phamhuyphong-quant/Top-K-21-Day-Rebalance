# Module Documentation — VN Cross-Sectional Ranking System

This document describes each module in `src/`, `config.py`, and `kaggle_kernel/kernel.py`: what it does, and how it's used. Intended for contributors, researchers, or anyone extending the pipeline.

---

## Table of Contents

1. [data_collect.py](#data_collectpy)
2. [features.py](#featurespy)
3. [alpha_mining.py](#alpha_miningpy)
4. [models.py](#modelspy)
5. [simulation.py](#simulationpy)
6. [evaluation.py](#evaluationpy)
7. [inference.py](#inferencepy)
8. [deep_combiner.py](#deep_combinerpy)
9. [app.py](#apppy)
10. [config.py](#configpy)
11. [kaggle_kernel/kernel.py](#kaggle_kernelkernelpy)
12. [Data Flow Summary](#data-flow-summary)

---

## `data_collect.py`

Builds the VN stock universe and keeps `market_data.parquet` / `vnindex_data.parquet` up to date via incremental fetches from `vnstock` (VCI source).

### Functions

- **`clean_symbols(symbol_list)`** — Normalises a raw list of ticker symbols (dedup, strip, uppercase-style cleanup).
- **`get_tags(fetching=False)`** — Returns the VN universe symbol list (VN30 + VNMidCap). Pass `fetching=True` to refresh from source rather than a cached list.
- **`clean_ohlcv(df)`** — Cleans a raw OHLCV DataFrame (type coercion, column normalisation) before it's merged into the store.
- **`_fetch_quote(symbol, start_d, end_d)`** — Fetches OHLCV history for one symbol over a date range; returns `None` on failure.
- **`update_market_data(...)`** — Main incremental-update entry point: for each symbol in the universe, fetches only the trading days missing since the last stored date, merges, and saves. This is what `daily_update.yml` runs.
- **`_drop_before_last_zero_volume(df)`** — Drops leading rows before a symbol's last zero-volume day (handles pre-listing / halted-trading noise).
- **`_merge_and_dedup(existing, new_chunks)`** — Merges newly fetched chunks into the existing store, de-duplicating on `(Symbol, date)`.
- **`fetch_indicator_data(...)`** — Fetches index-level data (e.g. VNINDEX) used for `vnindex_data.parquet`.
- **`_save(df, file_path)`** — Writes a DataFrame to Parquet.

---

## `features.py`

The feature-engineering and target-construction module. Every per-symbol feature function follows the same contract: **row *i* (date T) only uses data available up to and including close_{T-1}** — each function does its own internal lagging, so there's no risk of same-day leakage.

### Feature Functions

- **`rsi(df, window_length=14)`** — RSI computed from lagged close prices.
- **`volume(df)`** — Rolling volume averages (`vol_5d`, `vol_1m`, `vol_3m`) and surge ratios (`volume_surge_monthly`, `volume_surge_weekly`), all from lagged volume.
- **`return_ln(df)`** — Log returns over 1W/1M/3M/6M/1Y horizons, from lagged close.
- **`volatility(df)`** — Annualised rolling std of daily log returns over 1W/1M/3M/6M, plus shock ratios (`volatility_shock_monthly/weekly`).
- **`MA(df)`** — SMA/EMA at spans 9/21/50/100/200, plus `dist_SMA_*`/`dist_EMA_*` distance ratios (lagged close ÷ lagged MA).
- **`price_structure(df)`** — `dist_52w_high` (lagged close ÷ rolling 252-day max) and `log_ret_skip1m` (skip-1-month momentum).
- **`volume_quality(df)`** — `obv_trend` (sign-of-return-weighted volume, rolled 21 days, normalised by `vol_1m`) and `price_vol_divergence`.
- **`turnover_ratio(df)`**, **`limit_bias(df, d=60)`** — Structural/context features (liquidity level, up/down price-limit day imbalance). Used by XGBoost only, excluded from the deep combiner's alpha pool.
- **`amihud_illiquidity(df)`** — Amihud illiquidity proxy (log1p(|return| / volume)).
- **`adtv(df, window=21)`** — Average daily traded value, used to build the `is_liquid` mask when `adtv_limit` is set.
- **`robust_market_regime_pipeline_monthly(df, return_horizon=21, rolling_window=63, num_buckets=4, liquidity_mask=None)`** — Computes market-wide dispersion (MAD of cross-sectional returns) and buckets it into quartiles → **`regime_bucket_monthly`**. This is the **P2** component of the P1 entry filter (`Q1` = bottom-quartile dispersion regime).
- **`market_breadth(liquidity_mask=None)`** — Cross-sectional breadth features (`breadth_ema21`, `narrow_breadth`, etc.).
- Inside `build_features` (not standalone): **`market3m_ema63`** and **`market1m_ema21`** — EMA of market-wide 3-month and 1-month mean returns. `market1m_ema21 - market3m_ema63 < 0` is the **P3** (trend) component of the P1 filter.

### Target / Label Functions

- **`build_targets(df)`** — Computes `next_1m_ret` (`shift(-21)`) and `next_1w_ret` (`shift(-5)`) forward log returns, then **drops any row where these are NaN** — i.e. the most recent ~21 trading days per symbol, since there's no future price yet to compute a forward return from.
- **`make_adaptive_bucket_label(df, target_col='next_1m_ret', target_bucket_size=20, min_buckets=2, risk_adjust=True, vol_col='volatility_3m')`** — Assigns each stock a `target_bucket` label per date, with bucket count chosen adaptively so each bucket has ~`target_bucket_size` stocks. Risk-adjusts by `vol_col` when `risk_adjust=True`.
- **`make_magnitude_label(df, target_col='risk_adj_ret', scale_max=100.0, winsor_pct=0.0)`** — Min-max scales `risk_adj_ret` to `[0, scale_max]` **per date** → `target_magnitude`. This is the production model's actual training target (`eval_target_col` for `xgboost_ndcg` in `models.py`).
- **`target_generating_ranking(df, freq='M')`** — Computes `risk_adj_ret` (forward return ÷ volatility) and buckets it into 5 per-date quintiles → `target_quintile`; also assigns `qid` (XGBRanker group id, one per date). Drops rows where a quintile can't be formed (fewer than 5 valid values that day).

### `build_features(df, min_stocks_per_date=50, adtv_limit=None, generate_target=True)`

The main entry point — runs every feature function above in sequence, then (if `generate_target=True`) also builds targets/labels. Two important behaviors:

- **`generate_target` controls whether the most recent ~21 trading days survive.** With the default `generate_target=True`, `build_targets()` runs internally and drops any row lacking a forward return — necessary for **training** data, but it means the DataFrame's max date is always ~21 trading days behind the true latest date in the input. Pass **`generate_target=False`** to skip target construction entirely and keep the true latest date's features — this is what live **prediction** needs. `regime_bucket_monthly`, `market1m_ema21`, and `market3m_ema63` (the P1 filter inputs) are computed unconditionally either way, so they're available in both paths.
- **`adtv_limit`** — When set, computes ADTV and applies it as a liquidity mask (`is_liquid`) for all cross-sectional (rank/mean-per-date) computations, while per-symbol rolling windows still run on the full unfiltered series (to avoid gaps breaking long lookback windows like 100/252 days). Rows failing the mask are only dropped once, right before bucket/label formation.

### `apply_cross_sectional_ranking(df, feature_cols)`

Utility: adds `csr_<feature>` = per-date percentile rank for each feature in `feature_cols`. Used for exploratory analysis, not part of the live pipeline.

---

## `alpha_mining.py`

WorldQuant-style alpha factor library plus genetic-programming (GP) alpha search.

### Operator Functions

Time-series and cross-sectional primitives used to build alpha expressions: `ts_delay`, `ts_delta`, `ts_mean`, `ts_min`, `ts_rank`, `ts_argmax` (rolling, per-symbol), and `cs_rank`, `cs_mean`, `cs_mask` (cross-sectional, per-date, liquidity-mask-aware).

### `class WorldQuantAlphas`

Builds a lagged OHLCV frame internally (`self.ldf`) on construction so every alpha respects the same T-1 look-ahead-safe contract as `features.py`. `generate_all()` returns a DataFrame of all implemented alphas: **#001, #002, #006, #007, #013, #016, #024, #028, #040, #101, #103, #200, #201, #202** (see inline comments in `config.candidate_features` for what each one captures — momentum, volume-price co-movement, mean-reversion, low-beta, etc.). Accepts an optional `liquidity_mask` so cross-sectional pieces only aggregate over liquid names while per-symbol rolling pieces still run on the full continuous series.

### `add_and_filter_alphas(gp_model, original_df, input_features)`

Takes a fitted `gplearn` `SymbolicTransformer`, generates candidate alpha expressions, and filters them (e.g. by fitness/uniqueness) before merging the survivors back into `original_df`.

### `rank_ic_fitness(y_true, y_pred)`

Custom `gplearn` fitness function: Spearman rank IC between a candidate GP alpha and forward returns, used to guide the genetic search toward alphas with real predictive power rather than just low error.

---

## `models.py`

Feature selection, walk-forward cross-validation, and the model zoo.

### Feature Selection

- **`select_features_by_icir(df, candidate_features, icir_threshold=0.02, target_col='next_1m_ret')`** — Vectorised per-date IC/IR computation (Spearman rank correlation between each feature and forward return, aggregated as mean/std → IR). Returns `(selected_features, ic_ir_map)`; only training-window data should ever be passed in, so this is inherently look-ahead-safe when called per-fold.
- **`_compute_icir_map(df, features, target_col='next_1m_ret')`** — Same IC/IR computation without the threshold filter; used internally when `corr_prune=True` but `icir_filter=False` (still needs IC/IR values for tie-breaking).
- **`prune_correlated_features(df, candidate_features, ic_ir_map, feature_groups=None, correlation_threshold=0.75)`** — Within each `FEATURE_GROUPS` group, for every feature pair with `|Spearman corr| >= correlation_threshold`, drops whichever has the lower `|IC IR|`. Features outside any group are never pruned against each other.

### Walk-Forward Machinery

- **`class Fold`** — Dataclass: `fold_number`, `train_df`, `test_df`, `test_start`, `test_end`.
- **`generate_folds(df, initial_train_months=12, test_months=6, gap_days=21, liquidity_filter=False, adequate_adtv=2_500_000)`** — Produces the list of walk-forward `Fold`s. Model-agnostic. `gap_days` is the buffer between train cutoff and test start that prevents leakage from the ~21-day-forward target horizon.
- **`_filter_fold_by_liquidity(...)`** — Applies an ADTV-based liquidity filter independently to each fold's train/test data (used when `generate_folds(liquidity_filter=True)`).
- **`test_train_spliter(df, test_start, features)`** — Simpler, single-split (non-walk-forward) train/test helper used in ad-hoc research notebooks.

### Model Zoo

- **`predict_xgboost_ndcg(fold, features, model_params)`** — Production model: `XGBRanker` with `objective='rank:ndcg'`, trained/scored on `target_magnitude`.
- **`predict_xgboost_mse(...)`** — XGBoost regression baseline (`reg:squarederror`) for comparison.
- **`predict_linear(...)`** — Plain linear regression baseline.
- **`predict_lstm_mse(...)`** — LSTM regression head, explored as an alternative to tree-based models.
- **`predict_alphaforge(fold, alpha_pool, ...)`** — Routes to `AlphaForgeCombiner` (see `deep_combiner.py`) instead of a fold-trained ML model; takes an `alpha_pool` of WQ/GP columns instead of `(fold, features, model_params)`.
- **`class ModelSpec`** / **`MODEL_REGISTRY`** — Declares, per model name, which column (`eval_target_col`) NDCG/scoring should treat as ground truth. `xgboost_ndcg`, `xgboost_mse`, `linear`, and `lstm_mse` all score on `target_magnitude`; `alphaforge` is handled as its own branch in `walk_forward_cv` since its signature differs.
- **`build_mega_combiner(...)`** — Constructs an `AlphaForgeCombiner` instance with the given hyperparameters (no training — it recomputes dynamic weights at every rebalance date from rolling history).
- **`mine_gp_factors(df, features, initial_train_end)`** — One-time GP alpha mining: fits a `SymbolicTransformer` on the initial training window, then transforms the full `df`, adding new GP-derived alpha columns. Enabled via `walk_forward_cv(use_gp=True)`.
- **`score_and_evaluate(...)`**, **`_daily_precision_at_k(...)`**, **`_daily_topk_excess_return(...)`** — Per-fold scoring utilities: NDCG, precision@k, and buy-only-book excess return diagnostics.

### `walk_forward_cv(df, features, model='xgboost_ndcg', model_params=None, initial_train_months=12, test_months=6, gap_days=21, use_gp=False, icir_filter=False, icir_threshold=0.02, icir_target_col='next_1m_ret', corr_prune=False, corr_threshold=0.75, liquidity_filter=False, adequate_adtv=2_500_000, nonskip_features=None, top_k=20)`

The main orchestration function. For each fold: optionally re-runs IC/IR filtering and correlation pruning (both training-window-only, zero look-ahead), trains/scores the selected model, and records NDCG. Returns `(final_df, fold_ndcg_df)` — `final_df` is every fold's scored test set concatenated (with a `pred_score` column), `fold_ndcg_df` has one row per fold for plotting NDCG over time. This is what produced the paper's `df_predict_ndcg` (see `notebooks/NTH RESEARCH/Saving DataFrame.ipynb`, called with `icir_filter=False, corr_prune=True`).

---

## `simulation.py`

Object-oriented backtest engine — replaces an earlier monolithic `simulate_portfolio()` function. Used by the research notebooks (e.g. `Q1 and Q2.ipynb`) to backtest P1/P2/P3 and compare filter variants.

- **`class DataEngine(df)`** — Wraps a DataFrame for fast, cursor-based iteration: per-symbol arrays for `get_last`/`get_upcoming` lookups, plus `date_to_rows` for whole-date slices. `reset()` rewinds the cursor for reuse across many runs without rebuilding.
- **`class Stock`** — Tracks a single position's shares, including T+N pending (unsettled) packs via `add_pack`/`next_day`/`reduce_shares`.
- **`class Portfolio(initial)`** — Full portfolio state: `execute_buy`/`execute_sell` (with fee/tax and T+3 settlement via `illiquid_cash`), `next_day` (marks positions to market, settles pending cash, records NAV history and per-date holdings snapshots). `skipped_orders` is an audit trail of any order that failed (insufficient cash/shares/no position).
- **`class DataMismatch(Exception)`** — Raised when raw/predict/condition DataFrames don't cover the same set of dates.
- **`class StepFilter(multiplier)`** — Always returns a fixed exposure multiplier. This is what implements P1's cash state: `StepFilter(0)` = zero exposure when triggered.
- **`class RampFilter(column, floor, threshold, min_value=0.0, max_value=1.0)`** — Linear ramp between `min_value` and `max_value` based on a column's value between `floor` and `threshold` (used for P2-only partial-exposure-reduction variants, e.g. `StepFilter(0.5)`).
- **`class FilterGroup(condition=None, transformations=None)`** — Evaluates `condition` (a string expression, e.g. `"market1m_ema21-market3m_ema63<0 and regime_bucket_monthly=='Q1'"`) against a row; if true, chains the `transformations` (Step/RampFilter) to compute a combined exposure multiplier. This is exactly how **P1** is expressed in the research notebooks.
- **`build_data_engines(df_raw, df_predict, df_condition=None)`** — One-time setup: date-range-aligns `df_raw`/`df_predict`/`df_condition`, validates they cover matching date sets (raises `DataMismatch` otherwise), and wraps each in a `DataEngine`. Call once per dataset and reuse via `.reset()` across trials.
- **`class OrderManager(initial, topk, df_raw, df_predict, df_condition=None, regime_filter=None)`** — Runs a full strategy simulation: `run_strategy(allocation_strategy='equal')` steps day-by-day, rebalancing every 21 days into the top-`topk` predicted names (scaled by `regime_filter`'s multiplier when set), settling trades, and recording NAV. `get_holdings(start_date, end_date)` reconstructs `{date: {symbol: shares}}` after a run.
- **`class NullHypothesisTest(OrderManager)`** — Subclass that overrides stock selection with **fully random** top-`topk` picks, for Monte Carlo null-hypothesis significance testing against the real strategy's performance.

---

## `evaluation.py`

Plotting and performance-reporting utilities — no longer contains a training/backtest orchestration function (that moved to `simulation.py`'s `OrderManager`).

### Functions

- **`plot_feature_importances(model, features)`** — XGBoost gain-based feature importance bar chart.
- **`compute_model_ic(test_df, pred_col='pred_score', ret_col='next_1m_ret')`** — Per-date Spearman IC between predictions and realised returns; reports IC Mean, IC Std, IC IR.
- **`compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col='next_1m_ret')`** — Win rate of top-quantile picks vs. market baseline.
- **`plot_feature_ic(test_df, features, target_col='next_1m_ret')`** / **`plot_feature_ir(...)`** / **`plot_feature_rolling_ir(...)`** — Feature-level IC/IR diagnostics (aggregate, per-feature bar chart, and rolling monthly IR line chart).
- **`plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None)`** — Mean forward return by predicted quintile bucket.
- **`plot_equity_curves(*results, labels=None, normalize=False, regime_colors=None, show_regime=False)`** — Overlay one or more equity curves, optionally with regime-bucket background shading.
- **`print_performance_report(result, initial_capital=None, rf_annual=0.045, trading_days_per_year=252, verbose=True)`** — Computes and prints Total Return, CAGR, Sharpe, Sortino, Max Drawdown, and Calmar from a `Portfolio`-style NAV history.

---

## `inference.py`

Live daily signal generation.

### `generate_paper_trade_signals(df, current_portfolio, features, use_mega=False, model=None, buy_n=30, trend_filter_col='dist_SMA_100', trend_filter_threshold=1.0, target_col='target_quintile', icir_filter=False, icir_threshold=0.02, icir_target_col='next_1m_ret', corr_prune=False, corr_threshold=0.75)`

- Splits `df` into `train_df` (`date < latest_date` and `target_col` not null) and `inference_df` (`date == latest_date`).
- Optionally re-runs IC/IR filtering (`icir_filter`) and correlation pruning (`corr_prune`) on `train_df` only.
- If `model is None`, retrains an `XGBRanker` on `train_df` (single-shot, not walk-forward) and scores `inference_df` to get `live_score`; ranks all stocks by score for that date.
- Applies a grace band: currently held symbols stay in `hold_list` as long as they rank within the top `buy_n`; new entries are drawn from the top `buy_n` candidates, optionally gated by `trend_filter_col` (set to `None` to disable — the live P1 pipeline does this, since P1's only filter is the market-level regime filter, not a per-stock trend filter).
- Returns `(buy_list, hold_list, sell_list, not_in_universe_list, ranked_today)`, where `ranked_today` has `['Symbol', 'live_score', 'rank']`.

### `get_actionable_portfolio_lists(df, current_portfolio, features, **kwargs)`

Thin wrapper around `generate_paper_trade_signals` that returns a dict (`BUY`/`HOLD`/`SELL`/`NOT_IN_UNIVERSE`/`Rankings`) instead of a tuple — convenience for UI/notebook use.

---

## `deep_combiner.py`

### `_rolling_rank_ic(factor_series, ret_series, window)`

Cross-sectional Spearman IC per date, rolled over `window` periods, returning both the rolling IC and rolling ICIR (IC mean ÷ IC std).

### `class AlphaForgeCombiner`

Implementation of AlphaForge's Algorithm 2 (arXiv:2406.18394). At each rebalance date:

1. Compute rolling RankIC/ICIR per factor over `ic_window` past periods.
2. Gate: drop factors with `|RankIC| < ic_threshold` or `|ICIR| < icir_threshold`.
3. Keep the top `max_active_factors` survivors by `|RankIC|`.
4. Fit a Ridge regression of those factors against recent returns → dynamic per-factor weights.
5. Apply weights to current-date factor values → a single "Mega-Alpha" score per stock.

Chosen over a nonlinear (e.g. LSTM-attention) combiner for interpretability and overfitting resistance, per the source paper. **Experimental** — found to underperform the XGBoost baseline out-of-sample; retained for research only, not used in production inference.

---

## `app.py`

A deliberately minimal Streamlit page — not a dashboard. No backtesting, feature diagnostics, or AI-commentary UI; it exists purely to surface today's live signal.

### Structure

1. **`_today_vn()`** — Returns today's date in Vietnam time (UTC+7) as a string, used purely as a cache key so `@st.cache_data(ttl=3600)` refreshes after VN midnight.
2. **`load_today_signals(_date_key)`** — Fetches `today_signals.parquet` from the Hugging Face dataset (`PhongHPham/vn_cross_sectional_ranking_data_storage`) over HTTPS, optionally authenticated via `st.secrets["HF_TOKEN"]` (needed only if the dataset is private).
3. Renders:
   - Title ("Scientific Research") and a research-only disclaimer caption.
   - A caption showing the data cutoff date (`signal_date`).
   - If `filter_active` is `True` in the loaded data → a "hold cash" warning, no table.
   - Otherwise → a table of `rank` / `Symbol` / `live_score` for today's top-20 picks.

No sidebar, no portfolio input, no charts — the app trusts that `kaggle_kernel/kernel.py` already applied P1's filter and Top-K selection before publishing.

---

## `config.py`

Central configuration — no functions with side effects, just constants and one small helper.

- **`final_features`** — Legacy static 20-feature list, hand-selected in an earlier research pass. No longer the operative feature set; kept for reference.
- **`candidate_features`** — The current full feature pool (~46 features) passed into `walk_forward_cv` / `generate_paper_trade_signals`. Per-fold IC/IR filtering and correlation pruning select the active subset at runtime.
- **`BASE_MODEL_PARAMS`** — Default XGBoost hyperparameters for the production `rank:ndcg` model (tree depth 3, 150 estimators, LambdaRank pairwise sampling, GPU (`device='cuda'`) training).
- **`FEATURE_GROUPS`** — Dict of `{group_name: [features]}` (`returns`, `volatility`, `moving_average`, `volume`, `rsi`, `wq_features`, `price_structure`) — defines which features compete against each other during correlation pruning.
- **`ALL_GROUP_NAMES`** — `list(FEATURE_GROUPS.keys())`.
- **`usedSymbols`** — Fixed list of ~250 VN tickers that live inference (`kaggle_kernel/kernel.py`) filters `market_data.parquet` down to before running the pipeline.
- **`groups_to_features(group_names)`** — Flattens a list of `FEATURE_GROUPS` keys into a single feature-name list.

---

## `kaggle_kernel/kernel.py`

Not part of `src/` — this is the standalone script pushed to Kaggle daily by `.github/workflows/precompute_model.yml`. It's the live production entry point that ties `features.py` and `inference.py` together and publishes the result.

Steps:

1. **Clone** the repo's source branch (`main` or a feature branch, whichever `precompute_model.yml` currently checks out) for `src/`/`config.py`, and the `data-storage` branch for `market_data.parquet`.
2. **Load & filter** `market_data.parquet` to `config.usedSymbols`, coerce dates, drop a couple of known bad dates (`2018-01-23/24`).
3. **Build features twice**:
   - `df = build_features(df_raw, adtv_limit=2_500_000)` — target-bearing, used for **training** (tail-trimmed by ~21 trading days).
   - `df_raw = build_features(df_raw, adtv_limit=2_500_000, generate_target=False)` — feature-only, retains the **true latest date**.
4. **Evaluate the P1 filter** on the true latest date (from step 3's `df_raw`): `regime_bucket_monthly == 'Q1'` AND `market1m_ema21 - market3m_ema63 < 0`.
   - **If active** → publishes a one-row "cash" record (`filter_active=True`, no stock rows).
   - **If inactive** → assigns `qid` to `df` (required by `XGBRanker`, normally added by `target_generating_ranking`, which this pipeline doesn't call), appends the true-latest-date row(s) from `df_raw` with `target_magnitude` set to `NaN`, and calls `generate_paper_trade_signals(..., buy_n=20, trend_filter_col=None, target_col='target_magnitude', icir_filter=False, corr_prune=True)` on the combined frame. Because `generate_paper_trade_signals` internally splits on `date == latest_date` for inference and `target_col.notna()` for training, this trains on `df`'s full labeled history and predicts on today, in one call.
5. **Publishes** `today_signals.parquet` (columns: `signal_date`, `filter_active`, and — when not in cash — `rank`/`Symbol`/`live_score` for the top 20) to the `PhongHPham/vn_cross_sectional_ranking_data_storage` Hugging Face dataset, which `src/app.py` reads.

---

## Data Flow Summary

```
data_collect.py ──► market_data.parquet, vnindex_data.parquet   (daily_update.yml)
                                │
                                ▼
                    features.py: build_features()
                    ├── generate_target=True  → training frame (tail-trimmed ~21 trading days)
                    └── generate_target=False → true-latest-date feature frame
                                │
                    ┌───────────┴────────────┐
                    ▼                         ▼
        models.py: walk_forward_cv()   kernel.py: append latest row,
        (research / offline backtests   call inference.generate_paper_trade_signals()
         via simulation.py OrderManager  (single-shot retrain + P1 filter check)
         + FilterGroup/StepFilter)                │
                    │                              ▼
                    ▼                    today_signals.parquet → Hugging Face
        evaluation.py: plots/reports                │
        (notebooks/NTH RESEARCH/*.ipynb)             ▼
                                            app.py (Streamlit, reads HF dataset)
```