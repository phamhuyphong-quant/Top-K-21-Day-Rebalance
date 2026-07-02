# Module Documentation — VN Cross-Sectional Ranking System

This document describes each module in `src/`, what it does, and how to use it. Intended for contributors, researchers, or anyone extending the pipeline.

---

## Table of Contents

1. [data_collect.py](#data_collectpy)
2. [features.py](#featurespy)
3. [alpha_mining.py](#alpha_miningpy)
4. [feature_search.py](#feature_searchpy)
5. [models.py](#modelspy)
6. [evaluation.py](#evaluationpy)
7. [inference.py](#inferencepy)
8. [deep_combiner.py](#deep_combinerpy)
9. [app.py](#apppy)
10. [config.py](#configpy)
11. [Data Flow Summary](#data-flow-summary)

---

## `data_collect.py`

**Purpose:** Constructs the VN investable universe and fetches/updates OHLCV market data. Supports multiple data sources (`VCI`, `KBS`) with automatic fallback, OHLC sanity cleaning, incremental merging, and logging.

### Functions

---

#### `clean_symbols(symbol_list) → list[str]`

Sanitises a raw list of ticker symbols by stripping whitespace, removing NaN values, deduplicating, and returning a sorted list.

**Parameters:**
- `symbol_list` — Raw list of symbols from a data source (may contain NaN or whitespace)

**Returns:** Sorted, deduplicated list of clean symbol strings.

---

#### `get_tags(fetching=False) → list[str]`

Returns the VN universe.

- If `fetching=False` (default): returns a **hardcoded static list** of ~300 tickers — fast and offline-safe, used by default in most workflows.
- If `fetching=True`: queries the KBS data source live via `vnstock` to combine VN30 + VNMidCap dynamically.

**Returns:** Sorted list of ~300 ticker symbols forming the VN universe.

**Example:**
```python
from src.data_collect import get_tags

# Fast path — use hardcoded list (default)
symbols = get_tags()

# Live path — query KBS for current index constituents
symbols = get_tags(fetching=True)
# Logs: VN30=30  VNMID=70  VN300~=300
```

---

#### `clean_ohlcv(df) → DataFrame`

Applies OHLCV sanity rules on a per-row basis. Detects any OHLC violation (e.g. `high < low`, non-positive prices) and forward-fills all four OHLC columns for that row from the previous valid row. Volume is never modified. No rows are dropped.

---

#### `update_market_data(file_path, symbols, start_date, batch_size, base_sleep, sources, filter_symbols) → None`

Fetches full OHLCV history from `start_date` to today for every symbol and saves to a Parquet file. Tries each source in order; falls back to the next source on timeout, rate-limit, or any other error.

Existing data is preserved — freshly fetched rows overwrite on `(date, Symbol)` conflict. Saves incrementally every `batch_size` symbols so progress is not lost mid-run.

**Parameters:**
- `file_path` (`str`) — Path to the output `.parquet` file (e.g., `"data/market_data.parquet"`)
- `symbols` (`list[str]`) — List of ticker symbols to fetch
- `start_date` (`str`, default `"2016-01-01"`) — Earliest date for initial data fetch
- `batch_size` (`int`, default `5`) — Number of symbols fetched per batch before a batch save
- `base_sleep` (`float`, default `3.0`) — Seconds to sleep between each symbol request (rate-limit courtesy)
- `sources` (`list[str]`, default `("VCI", "KBS")`) — Data sources to try in order; falls back to next on failure
- `filter_symbols` (`bool`, default `True`) — If `True`, purges rows whose `Symbol` is no longer in the provided list when loading the existing file

**Notes:**
- Rate-limit errors with an explicit wait time (e.g. `"30 giây"`) are detected via regex; the same source is retried once after the wait before falling back.
- All data is written in Parquet format using PyArrow for fast I/O.
- All activity is logged to `data_collect.log` and stdout.

---

#### `fetch_indicator_data(ind_symbol, start_date, file_path, sources) → None`

Fetches market index data (e.g. `"VNINDEX"`) and saves to a Parquet file. Tries each source in order; falls back on any error.

**Parameters:**
- `ind_symbol` (`str`) — Index symbol (e.g. `"VNINDEX"`)
- `start_date` (`str`) — Earliest date to fetch
- `file_path` (`str`) — Output path for the Parquet file
- `sources` (`list[str]`, default `("VCI", "KBS")`) — Data sources to try in order

---

## `features.py`

**Purpose:** Transforms raw OHLCV data into the full feature matrix used for model training. Also generates the ranking target variable.

> **Important:** All feature columns are shifted by 1 day inside `build_features()` before returning, so every feature reflects data known at close of the *previous* trading day. This prevents any same-day look-ahead during training or inference.

> **Important:** `build_features()` calls `WorldQuantAlphas.generate_all()` internally. Do **not** call `WorldQuantAlphas` manually on `df_raw` before passing it to `build_features()` — this will compute the alpha columns twice.

### Functions

---

#### `rsi(df, window_length=14) → DataFrame`

Computes the Relative Strength Index for each symbol using exponential weighted moving averages of gains and losses (Wilder smoothing).

**Adds columns:** `RSI_14` (or `RSI_{window_length}`)

---

#### `volume(df) → DataFrame`

Computes rolling volume averages and surge ratios to capture abnormal trading activity.

**Adds columns:**
- `vol_5d` — 5-day rolling average volume
- `vol_1m` — 21-day rolling average volume
- `vol_3m` — 63-day rolling average volume
- `volume_surge_weekly` — `vol_5d / vol_1m`
- `volume_surge_monthly` — `vol_1m / vol_3m`

---

#### `return_ln(df) → DataFrame`

Computes multi-horizon log returns relative to the current close. The 1-day lag applied in `build_features()` ensures look-ahead safety.

**Adds columns:** `log_ret_1w`, `log_ret_1m`, `log_ret_3m`, `log_ret_6m`, `log_ret_1y`

---

#### `volatility(df) → DataFrame`

Computes rolling annualised volatility at multiple horizons and volatility shock ratios (recent vol / longer-term vol). Uses a temporary `log_ret_daily` column that is dropped before returning.

**Adds columns:**
- `volatility_1w`, `volatility_1m`, `volatility_3m`, `volatility_6m` — Annualised rolling std (multiplied by √252)
- `volatility_shock_weekly` — `volatility_1w / volatility_1m`
- `volatility_shock_monthly` — `volatility_1m / volatility_3m`

---

#### `MA(df) → DataFrame`

Computes Simple and Exponential Moving Averages and the price's distance from key MAs.

**Adds columns:**
- `SMA_9`, `SMA_21`, `SMA_50`, `SMA_100`, `SMA_200`
- `EMA_9`, `EMA_21`, `EMA_50`, `EMA_100`, `EMA_200`
- `dist_SMA_9`, `dist_SMA_21`, `dist_SMA_50`, `dist_SMA_100`, `dist_SMA_200` — `close / SMA_N` (values above 1.0 mean price is above the MA)
- `dist_EMA_9`, `dist_EMA_21`, `dist_EMA_50`, `dist_EMA_100`, `dist_EMA_200` — `close / EMA_N`

---

#### `price_structure(df) → DataFrame`

Computes features that capture longer-term price positioning.

**Adds columns:**
- `dist_52w_high` — `close / rolling_252_max_close` — how far the stock is below its 52-week high
- `log_ret_skip1m` — `log(close_shifted_21 / close_shifted_126)` — the 3M→1M momentum factor (skips the most recent month to avoid short-term reversal noise)

---

#### `volume_quality(df) → DataFrame`

Computes two derived volume features beyond raw surge ratios.

**Adds columns:**
- `obv_trend` — On-Balance Volume trend normalised by 21-day average volume: `sign(Δclose) × volume` rolled over 21 days, divided by `vol_1m`
- `price_vol_divergence` — `log_ret_1m / (volume_surge_monthly + 0.001)` — flags stocks where price is rising on declining volume

> **Note:** `volume_quality()` depends on `volume()` and `return_ln()` having been called first. `build_features()` handles this ordering automatically.

---

#### `build_features(df, min_stocks_per_date=50) → DataFrame`

Master pipeline that calls all feature functions in the correct order: `WorldQuantAlphas.generate_all()` → `return_ln` → `volatility` → `MA` → `volume` → `rsi` → `volume_quality` → `price_structure`.

Look-ahead is prevented by a 1-day shift applied inside each sub-function (e.g. `return_ln`, `volatility`, `MA`, etc.) before features are assembled. Rows with any `inf` or `NaN` in feature columns are then dropped. Dates where fewer than `min_stocks_per_date` symbols survive are also removed (logged as a warning).

**Returns:** Cleaned DataFrame with all features. Forward-return targets (`next_1m_ret`, `next_1w_ret`) are **not** added here — call `build_targets()` separately.

---

#### `build_targets(df) → DataFrame`

Adds the forward-return target columns used during evaluation:

- `next_1m_ret` — Log return over the next 21 trading days (the main target)
- `next_1w_ret` — Log return over the next 5 trading days

Must be called **after** `build_features()`.

---

#### `target_generating_ranking(df, freq='M') → DataFrame`

Constructs the ranking target variable used to train the XGBoost ranker.

**Process:**
1. Computes `risk_adj_ret = next_1m_ret / volatility_3m` (or `next_1w_ret / volatility_1m` for weekly)
2. Bins each stock into 5 quintiles *within each trading day* using `pd.qcut` on risk-adjusted returns (with rank-based tie-breaking)
3. Assigns a `target_quintile` label (0 = worst, 4 = best) per stock per day
4. Assigns a `qid` (query ID) per day for XGBoost's ranking format

**Parameters:**
- `freq` (`'M'` or `'W'`) — Monthly or weekly ranking target

**Returns:** DataFrame with `qid`, `risk_adj_ret`, and `target_quintile` columns added.

---

#### `apply_cross_sectional_ranking(df, feature_cols) → (DataFrame, list)`

Transforms raw feature values into cross-sectional percentile ranks within each trading day. This normalises each feature relative to the universe on that day, making features directly comparable across time.

**Returns:** `(df_ranked, csr_features)` where `csr_features` is the list of new `csr_{col}` column names.

---

## `alpha_mining.py`

**Purpose:** Implements 14 WorldQuant-style quantitative alpha factors and a genetic programming (GP) framework for discovering new ones.

### Operator Functions

Low-level time-series operators used internally by the `WorldQuantAlphas` class:

| Function | Description |
|---|---|
| `ts_delay(df, col, d)` | Shift series backward by `d` days (per symbol) |
| `ts_delta(df, col, d)` | First difference over `d` days |
| `ts_mean(df, col, d)` | Rolling mean over `d` days |
| `ts_min(df, col, d)` | Rolling minimum over `d` days |
| `ts_rank(df, col, d)` | Rolling percentile rank over `d` days |
| `ts_argmax(df, col, d)` | Position of maximum value in rolling `d`-day window |

---

### `class WorldQuantAlphas`

Computes 14 WorldQuant alpha factors. On initialisation the DataFrame is sorted by `['Symbol', 'date']`. Look-ahead is prevented upstream by the 1-day shift applied in `build_features()`.

**Constructor:**
```python
wq = WorldQuantAlphas(df)  # df must have: Symbol, date, open, high, low, close, volume
```

**Alpha Factors:**

| Method | Output Column | Formula Logic |
|---|---|---|
| `get_alpha_001()` | `WQ_Alpha_001` | 12-1M momentum: sign of 12M return × (-1M vol rank) |
| `get_alpha_002()` | `WQ_Alpha_002` | `-corr(rank(Δlog(volume), 2), rank((close-open)/open, 2), 6)` — volume vs intraday return correlation |
| `get_alpha_006()` | `WQ_Alpha_006` | `-corr(open, volume, 10)` — negative open-volume correlation |
| `get_alpha_007()` | `WQ_Alpha_007` | Conditional momentum that only fires on high-volume days |
| `get_alpha_013()` | `WQ_Alpha_013` | `-rank(cov(rank(close), rank(volume), 5))` — close-volume co-movement penalty |
| `get_alpha_016()` | `WQ_Alpha_016` | Same as 013 using `high` instead of `close` — breakout/blowoff top signal |
| `get_alpha_024()` | `WQ_Alpha_024` | Conditional mean-reversion: compares 100-day price trend direction |
| `get_alpha_028()` | `WQ_Alpha_028` | `corr(adv20, low, 5) + midprice - close` — volume-adjusted midpoint deviation |
| `get_alpha_040()` | `WQ_Alpha_040` | Penalises high-vol stocks with high volume-return correlation (avoids blow-off tops) |
| `get_alpha_101()` | `WQ_Alpha_101` | BAB factor: negative rolling beta to the equal-weighted market |
| `get_alpha_103()` | `WQ_Alpha_103` | Monthly price acceleration (21-day second derivative) |
| `get_alpha_200()` | `WQ_Alpha_200` | Value proxy: distance from 52-week high |
| `get_alpha_201()` | `WQ_Alpha_201` | Volume trend confirmation: price momentum × volume momentum |
| `get_alpha_202()` | `WQ_Alpha_202` | Residual momentum: 12-1M excess return vs equal-weighted market |

**`generate_all() → DataFrame`**

Computes all 14 alphas, replaces `inf`/`NaN` with `NaN`, and returns a DataFrame with columns `WQ_Alpha_001` through `WQ_Alpha_202`. This is called **automatically inside `build_features()`** — do not invoke it manually before calling `build_features()`.

> **Note:** Of the 14 alphas, `WQ_Alpha_024` and `WQ_Alpha_028` are among the features in `config.candidate_features` (the operative feature pool passed to `walk_forward_cv`). `config.final_features` is a static legacy reference and is no longer the operative list.

---

### `add_and_filter_alphas(gp_model, original_df, input_features) → (DataFrame, list)`

Applies a trained `gplearn` GP model to generate new alpha expressions from base features. Deduplicates the output (removes identical alpha signals) and appends unique ones as `Mega_Alpha_N` columns.

---

### `rank_ic_fitness(y_true, y_pred) → float`

Custom fitness function for GP optimisation. Returns the Spearman rank correlation (IC) between an alpha's predicted values and actual future returns. Higher IC = better alpha.

---

## `feature_search.py`

**Purpose:** Exhaustively searches all combinations of predefined feature groups to find which combination produces the highest ROI or Sharpe ratio in a walk-forward backtest.

### Feature Groups

Features are organised into seven named groups:

| Group | Features |
|---|---|
| `returns` | `log_ret_1w`, `log_ret_1m`, `log_ret_3m`, `log_ret_6m`, `log_ret_1y` |
| `volatility` | `volatility_1w/1m/3m/6m`, `volatility_shock_monthly/weekly` |
| `moving_average` | `dist_SMA_9/21/50/100/200`, `dist_EMA_9/21/50/100/200` |
| `volume` | `volume_surge_monthly/weekly`, `obv_trend`, `price_vol_divergence` |
| `rsi` | `RSI_14` |
| `wq_features` | `WQ_Alpha_001`, `WQ_Alpha_002`, `WQ_Alpha_006`, `WQ_Alpha_007`, `WQ_Alpha_013`, `WQ_Alpha_016`, `WQ_Alpha_024`, `WQ_Alpha_028`, `WQ_Alpha_040`, `WQ_Alpha_101`, `WQ_Alpha_103`, `WQ_Alpha_200`, `WQ_Alpha_201`, `WQ_Alpha_202` |
| `price_structure` | `dist_52w_high`, `log_ret_skip1m` |

With 7 groups, the search evaluates up to 127 non-empty combinations — feasible to run on Kaggle.

### Functions

---

#### `groups_to_features(group_names) → list[str]`

Helper that flattens a list of group names into the corresponding flat list of feature column names.

---

#### `find_best_feature_combo(df, *, initial_capital, walk_forward_kwargs, backtest_kwargs, model_params, min_groups, verbose) → (list[str], float, DataFrame)`

Searches all feature group combinations and returns the one that maximises **final portfolio ROI (%)**.

**Returns:** `(best_features, best_roi, results_df)`

---

#### `find_best_feature_combo_sharpe(df, *, risk_free_rate, ...) → (list[str], float, DataFrame)`

Same as above but optimises for **annualised Sharpe ratio** instead of raw ROI.

**Returns:** `(best_features, best_sharpe, results_df)`

---

#### `find_best_feature_combo_subset(df, candidate_groups, metric, **kwargs) → (list[str], float, DataFrame)`

Restricts the search to a subset of groups. Pass `metric='roi'` or `metric='sharpe'`.

---

#### `search_best_roi_and_sharpe(df, *, risk_free_rate, initial_capital, walk_forward_kwargs, backtest_kwargs, model_params, min_groups, verbose) → (dict, dict, DataFrame)`

Runs the search once and identifies the best combination for both ROI and Sharpe simultaneously, avoiding two separate searches.

**Returns:** `(best_roi_row, best_sharpe_row, results_df)`

**Example:**
```python
from src.feature_search import search_best_roi_and_sharpe

best_roi_row, best_sharpe_row, result_df = search_best_roi_and_sharpe(
    df,
    initial_capital=10_000,
    walk_forward_kwargs=dict(initial_train_months=24, test_months=6, gap_days=21),
    backtest_kwargs=dict(topk=10, trend_filter_col='dist_SMA_100'),
)
print(best_roi_row['combo_name'], best_roi_row['roi_%'])
```

---

## `models.py`

**Purpose:** Model training, walk-forward cross-validation, Optuna hyperparameter search, and the AlphaForge training loop.

### Functions

---

#### `walk_forward_cv(df, features, model_params=None, initial_train_months=12, test_months=6, gap_days=21, model='xgboost', use_gp=False, icir_filter=False, icir_threshold=0.02, icir_target_col='next_1m_ret', corr_prune=False, corr_threshold=0.75, liquidity_filter=False) → DataFrame`

The core training and evaluation function. Simulates live deployment by rolling through time:

1. Train on all data up to `train_cutoff` (= `test_start - gap_days`)
2. Evaluate on `test_start` → `test_start + test_months`
3. Advance the window and repeat

At each fold, the model is fitted on the training slice and predictions are stored as `pred_score` in the OOS test DataFrame. Per-fold NDCG is computed and printed.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `initial_train_months` | `12` | Months of data required before first test fold. **All published results use `24` — always pass this explicitly.** |
| `test_months` | `6` | Length of each test window |
| `gap_days` | `21` | Trading-day gap between train end and test start |
| `model` | `'xgboost'` | `'xgboost'` for XGBoost LambdaRank only; `'alphaforge'` for the AlphaForge dynamic factor combiner |
| `use_gp` | `False` | Enable GP alpha mining (adds `gplearn`-evolved alpha columns before training) |
| `icir_filter` | `False` | Enable per-fold IC/IR feature selection. At each fold, features are evaluated on the training window only and those with `\|IC IR\| <= icir_threshold` are dropped before the model sees any data. Eliminates look-ahead bias that would result from selecting features on the full dataset upfront (as in notebook 02). |
| `icir_threshold` | `0.02` | Minimum `\|IC IR\|` for a feature to be used in a given fold. Features that fall below this bar in a fold may recover and be included in later folds as more history accumulates. |
| `icir_target_col` | `'next_1m_ret'` | Forward-return column used as the IC target. Change to `'next_1w_ret'` for weekly-horizon experiments. |
| `corr_prune` | `False` | Enable per-fold Spearman correlation pruning. After the IC/IR filter (or directly on the full candidate pool if `icir_filter=False`), pairs of features within the same `FEATURE_GROUPS` group whose `\|Spearman corr\| ≥ corr_threshold` are compared and the one with the lower `\|IC IR\|` is dropped. Computed on `fold.train_df` only — zero look-ahead bias. |
| `corr_threshold` | `0.75` | Correlation threshold for `corr_prune`. Mirrors the value used in notebook 02. |
| `liquidity_filter` | `False` | Filter out illiquid stocks before training |

> **Note:** The function default of `initial_train_months=12` is a code fallback only. Always pass `initial_train_months=24` to reproduce the reported backtest results (63-month period, multiple folds).

> **Note on `icir_filter` / `corr_prune`:** Feature selection in notebook 02 is performed on the full dataset and is intentionally exploratory. Use `icir_filter=True, corr_prune=True` in `walk_forward_cv` whenever you want a fully bias-free pipeline — each fold selects and deduplicates its own feature subset using only the data it is allowed to see.

**Returns:** Concatenated OOS predictions DataFrame with a `pred_score` column. Pass directly to `simulate_portfolio()` or `compute_top_quantile_win_rate()`.

---

#### `optimize_xgboost_ranker(df, features, n_trials=50) → dict`

Uses Optuna to find the best XGBoost hyperparameters. Trains on data up to 2023-12-11 and validates on 2024, keeping 2025+ as a held-out test set. Maximises mean NDCG across the validation period.

**Returns:** `dict` of best hyperparameters, ready to pass as `model_params` to `walk_forward_cv`.

---

#### `build_mega_combiner(ic_window=40, ic_threshold=0.02, icir_threshold=0.2, max_active_factors=13, ridge_alpha=1.0) → AlphaForgeCombiner`

Constructs an `AlphaForgeCombiner` instance (AlphaForge Algorithm 2). No training happens here — the combiner recomputes dynamic weights at every `fit_and_predict()` call using rolling RankIC/ICIR gating and Ridge regression. Called internally by the `'alphaforge'` path in `walk_forward_cv`.

**Parameters:**
- `ic_window` — Number of past dates used to compute rolling RankIC (default `40`)
- `ic_threshold` — Minimum |RankIC| for a factor to remain active (default `0.02`)
- `icir_threshold` — Minimum |ICIR| for a factor to remain active (default `0.2`)
- `max_active_factors` — Top-N active factors passed to Ridge (default `13`)
- `ridge_alpha` — Ridge regularisation strength (default `1.0`)

---

#### `test_train_spliter(df, test_start, features) → tuple`

Simple utility for a single train/test split at a given date (with a 30-day gap). Used for one-off experiments outside the walk-forward loop.

**Returns:** `(X_train, y_train, qids_train, X_test, y_test, qids_test, test_df)`

---

## `evaluation.py`

**Purpose:** Backtest engine, portfolio performance metrics, model IC, and feature quality metrics on OOS predictions.

### Functions

---

#### `compute_model_ic(test_df, pred_col='pred_score', ret_col='next_1m_ret') → dict`

Computes the model-level Information Coefficient: the Spearman rank correlation between `pred_score` and actual `next_1m_ret`, computed daily then averaged.

**Returns:** Dictionary with keys `ic_mean`, `ic_std`, `ic_ir`, `daily_ic`.

---

#### `compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col='next_1m_ret') → dict`

Evaluates the ranker's practical trading performance.

Supports two modes:
- **Static mode:** Pass `X_test` and `ranker` to score predictions on the fly
- **Dynamic mode (preferred):** Leave `X_test`/`ranker` as `None` and read `pred_score` from `test_df` (output of `walk_forward_cv`)

**Computes:**
- **Top-quintile win rate** — % of top picks with positive `ret_col` over the horizon
- **Market baseline win rate** — Same, over the full universe
- **Win rate lift** — Difference between the two

**Returns:**
```python
{
    'top_win_rate': float,
    'market_win_rate': float,
    'lift': float,
    'top_picks_df': DataFrame
}
```

---

#### `plot_feature_ic(test_df, features, target_col='next_1m_ret') → DataFrame`

Plots and prints IC statistics for each feature: IC Mean and IC Std.

**Returns:** DataFrame with columns `IC Mean`, `IC Std` indexed by feature name.

---

#### `plot_feature_ir(test_df, features, target_col='next_1m_ret') → DataFrame`

Extends `plot_feature_ic` to also compute and plot **IC IR** (IC Mean / IC Std) — the signal-to-noise ratio of each feature. Used during the feature pruning step in notebook 02.

---

#### `plot_feature_rolling_ir(test_df, feature, target_col='next_1m_ret', window=6) → None`

Plots the rolling IC IR for a single feature over time (default: 6-month rolling window). Useful for diagnosing whether a feature's predictive power is stable or regime-dependent.

---

#### `plot_feature_importances(model, features) → None`

Plots a horizontal bar chart of XGBoost gain-based feature importance.

---

#### `plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None) → None`

Plots average forward return by predicted quintile (1 = worst to 5 = best). A well-calibrated ranker should show a strictly increasing return from quintile 1 to 5. Supports both static and dynamic modes.

---

#### `plot_equity_curves(*results, labels=None, normalize=False) → None`

Plots one or more equity curves on the same chart. Pass `normalize=True` to show cumulative return (%) normalised to a common start — useful for comparing XGBoost vs AlphaForge.

**Example:**
```python
ev.plot_equity_curves(result_basic, result_alphaforge,
                      labels=['Basic XGBoost', 'AlphaForge'],
                      normalize=True)
```

---

#### `simulate_portfolio(df, model, features, initial_capital, topk, time_of_rebalance, trend_filter_col, settlement_delay, vnindex_df, liquidity_filter, vol_lookback, vol_percentile, vol_window, adtv_participation, allocation) → DataFrame`

Simulates a realistic portfolio with VN-market timing conventions.

**Execution model:**
1. **Rebalance day (day 0):** Model scores stocks → ranks universe → hard-sells every stock NOT in top-N at today's price; sell proceeds enter `pending_cash` (available after T+`settlement_delay`)
2. **Settlement (day +3):** Pending cash becomes available → new buy orders execute at settlement-day prices, split across all buy targets according to the allocation strategy (`'equal'` or `'rank_weighted'`)
This correctly models VN T+3 settlement — you cannot buy with money from the same-day sell.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Trained XGBRanker; pass `None` to use existing `pred_score` column |
| `initial_capital` | `10_000_000` | Starting capital |
| `topk` | `10` | Number of top-ranked stocks to buy |
| `time_of_rebalance` | `'M'` | Rebalance frequency key: `'M'` = every 21 trading days, `'W'` = every 5 trading days. Matches the `shift(-21)` / `shift(-5)` horizon used in `build_targets()` so the holding period the model was trained on equals the holding period in the backtest. |
| `trend_filter_col` | `'dist_SMA_100'` | Stock must have this column > 1.0 to qualify as a new buy. Pass `None` to disable. |
| `settlement_delay` | `3` | Trading days between sell and cash availability |
| `vnindex_df` | `None` | Optional VNINDEX DataFrame for volatility regime filtering |
| `liquidity_filter` | `True` | Filter out low-liquidity stocks before ranking using ADTV thresholds |
| `vol_lookback` | `21` | Days for realised VNINDEX volatility calculation |
| `vol_percentile` | `0.80` | Vol percentile threshold; above this = high-vol regime → rebalance skipped |
| `vol_window` | `252` | Rolling window for computing the percentile benchmark |
| `adtv_participation` | `0.10` | Max fraction of ADTV a position may represent; stocks where the target size exceeds this threshold are excluded |
| `allocation` | `'equal'` | Cash allocation strategy per buy order: `'equal'` splits cash evenly across all stocks; `'rank_weighted'` gives more cash to higher-ranked stocks proportionally to their rank position |

> **Volatility Regime Filter:** When `vnindex_df` is provided, rebalance months where VNINDEX realised vol exceeds the `vol_percentile` of its own history are skipped entirely. The equity curve is still recorded continuously.

**Returns:** DataFrame with columns `date`, `total_value`, `cash`, `pending_cash`, `number_of_holdings`.

---

#### `print_performance_report(result, initial_capital=None, rf_annual=0.045) → dict`

Computes and prints a full risk-adjusted performance report from a backtest equity curve.

**Metrics computed:** Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Monthly Win Rate, Profit Factor.

**Parameters:**
- `result` — DataFrame returned by `simulate_portfolio`
- `initial_capital` — Starting capital; if `None`, uses `result['total_value'].iloc[0]`
- `rf_annual` — Annual risk-free rate (default `0.045` — approximate Vietnam T-bill rate)

**Returns:** `dict` with keys `total_return`, `cagr`, `sharpe`, `sortino`, `calmar`, `max_drawdown`, `win_rate`, `profit_factor`.

---

#### `pretrain_and_save_artifacts(df, selected_features, use_mega_alpha, output_dir, vnindex_df) → (str, str, str)`

Used by the GitHub Actions precompute workflow. Runs the full walk-forward CV and backtest pipeline (`initial_train_months=24`, `test_months=6`, `gap_days=21`) and saves three artifacts so the Streamlit app can load them instantly without retraining.

**Parameters:**
- `df` — Fully processed DataFrame (features + targets + `qid` must already exist)
- `selected_features` — List of feature column names to train on
- `use_mega_alpha` — Reserved for future use; currently always runs the XGBoost path
- `output_dir` (`str`, default `"data/pretrained/"`) — Directory where artifacts are saved
- `vnindex_df` — Reserved for future use; currently unused

**Returns:** `(predictions_path, equity_curve_path, final_model_path)` — paths to the three saved artifacts:
- `pretrained_predictions.parquet` — Full OOS predictions DataFrame
- `pretrained_equity_curve.parquet` — Equity curve from the backtest

---

## `inference.py`

**Purpose:** Generates live paper-trading signals for a given portfolio by training on all available history and scoring today's universe.

### Functions

---

#### `generate_paper_trade_signals(df, current_portfolio, features, use_mega=False, model=None, buy_n=30, trend_filter_col='dist_SMA_100', trend_filter_threshold=1.0, target_col='target_quintile', icir_filter=False, icir_threshold=0.02, icir_target_col='next_1m_ret', corr_prune=False, corr_threshold=0.75) → tuple`

The main inference function. If no pretrained `model` is passed, trains a fresh XGBoost ranker on all historical data up to (but not including) today's date, then scores today's VN universe.

**Signal generation logic:**

1. **Portfolio review** — For each currently held symbol:
   - If not in today's VN → `NOT_IN_UNIVERSE`
   - If rank > `buy_n` → `SELL`
   - Otherwise → `HOLD`

2. **New buy candidates** — From the top `buy_n` ranked stocks today:
   - Skip if already held
   - Skip if `trend_filter_col` value ≤ `trend_filter_threshold`
   - Otherwise → `BUY`

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `buy_n` | `30` | Top N stocks targeted for new entries; also the hold threshold |
| `trend_filter_col` | `'dist_SMA_100'` | Feature column for the trend filter |
| `trend_filter_threshold` | `1.0` | Stock must be above this value to qualify as a new buy |
| `use_mega` | `False` | Not yet implemented — raises `NotImplementedError` if `True` |
| `model` | `None` | Pass a pretrained `XGBRanker` to skip training (used by the dashboard) |
| `icir_filter` | `False` | If `True`, runs IC/IR feature selection on the full training history before fitting the model. Features with `\|IC IR\| ≤ icir_threshold` are dropped. Same logic as the per-fold filter in `walk_forward_cv`, applied once to the full available history. |
| `icir_threshold` | `0.02` | Minimum `\|IC IR\|` to keep a feature when `icir_filter=True`. |
| `icir_target_col` | `'next_1m_ret'` | Forward-return column used as the IC target. |
| `corr_prune` | `True` | If `True`, runs Spearman correlation pruning on the training history after the IC/IR filter (or on the full candidate pool if `icir_filter=False`). Within each `FEATURE_GROUPS` group, the weaker of any highly-correlated pair (`\|corr\| ≥ corr_threshold`) is dropped. Keeps the feature set consistent with what `walk_forward_cv(corr_prune=True)` used during training. |
| `corr_threshold` | `0.75` | Correlation threshold for `corr_prune`. |

**Returns:** `(buy_list, hold_list, sell_list, not_in_universe_list, ranked_today_df)`

where `ranked_today_df` has columns `Symbol`, `live_score`, `rank`.

---

#### `get_actionable_portfolio_lists(df, current_portfolio, features, **kwargs) → dict`

Thin wrapper around `generate_paper_trade_signals` that returns a clean dictionary:

```python
{
    "BUY": [...],
    "HOLD": [...],
    "SELL": [...],
    "NOT_IN_UNIVERSE": [...],
    "Rankings": DataFrame
}
```

---

## `deep_combiner.py`

**Purpose:** Experimental AlphaForge-style factor combiner that dynamically weights a set of alpha signals based on rolling RankIC/ICIR statistics. Used by the `'alphaforge'` path in `walk_forward_cv`.

**Status:** Experimental. In walk-forward backtesting, AlphaForge produced a higher IC IR but weaker risk-adjusted returns (Sharpe 0.81 vs 0.96, monthly win rate 58.73% vs 63.49%) compared to the XGBoost baseline. The production pipeline uses the XGBoost model only.

### `class AlphaForgeCombiner`

**Architecture (AlphaForge Algorithm 2):**

At each rebalance date, given a factor zoo `Z = {f1, ..., fk}`:
1. Compute rolling RankIC and ICIR for each factor over the past `ic_window` periods
2. **Gate** — drop factors where `|RankIC| < ic_threshold` OR `|ICIR| < icir_threshold`
3. Sort survivors by `|RankIC|`, keep top `max_active_factors`
4. Fit Ridge regression of those factors against recent returns → dynamic weights
5. Apply weights to current-date factor values → `Mega_Alpha` scalar per stock

The paper motivates the linear combiner (over LSTM or attention-based approaches) for interpretability and overfitting resistance on financial noise.

**Constructor:**
```python
combiner = AlphaForgeCombiner(
    ic_window=40,
    ic_threshold=0.02,
    icir_threshold=0.2,
    max_active_factors=13,
    ridge_alpha=1.0,
)
```

**Parameters:**
- `ic_window` — Number of past dates for rolling RankIC (default `40`)
- `ic_threshold` — Minimum |RankIC| to keep a factor active (default `0.02`)
- `icir_threshold` — Minimum |ICIR| to keep a factor active (default `0.2`)
- `max_active_factors` — Top-N factors by |RankIC| passed to Ridge (default `13`)
- `ridge_alpha` — Ridge regularisation strength (default `1.0`)

**`fit_and_predict(hist_df, current_df, alpha_cols, ret_col='next_1m_ret') → Series`**

Fits the dynamic factor weights on `hist_df` (historical data up to the rebalance date) and applies them to `current_df` (today's cross-section) to produce a `Mega_Alpha` score per stock.

```python
mega_alpha_scores = combiner.fit_and_predict(
    hist_df=train_slice,
    current_df=today_df,
    alpha_cols=['WQ_Alpha_001', 'WQ_Alpha_006', ...],
    ret_col='next_1m_ret',
)
# Returns: pd.Series of Mega_Alpha scores, one per stock
```

**`report() → None`**

Prints the currently active factors and their Ridge weights, sorted by absolute weight. Useful for inspecting which signals the combiner is relying on at a given rebalance date.

**Training:** The combiner has no separate training step — it recomputes weights at every `fit_and_predict()` call. `build_mega_combiner()` in `models.py` is the factory function used to construct an instance with configured hyperparameters.

---

## `app.py`

**Purpose:** The Streamlit dashboard. Cloud-hosted on Streamlit Community Cloud; relies on `GITHUB_TOKEN` and `GOOGLE_API_KEY` secrets — not designed for local execution.

### Structure

**1. Data & Signal Loading**
- `load_data()` — Loads raw market data from the GitHub-hosted Parquet file on the `data-storage` branch. Cache key includes today's date to bust daily.
- `load_pretrained()` — Loads pre-computed walk-forward OOS predictions and equity curve from Hugging Face (`PhongHPham/vn_cross_sectional_ranking_data_storage`).
- `display_portfolio_signals_ui()` — Renders the live Buy / Hold / Sell signal table for a user-defined portfolio, calling `inference.generate_paper_trade_signals()` on demand.

**2. Backtest & Analytics Panel**
- Strategy settings (buy fraction, trend filter, rebalance frequency) controlled from the sidebar.
- Interactive Plotly equity curve of the pre-computed backtest.
- Portfolio metrics (Final NAV, CAGR, Sharpe) via `st.metric`.
- Feature importance and IC charts from the pre-loaded OOS predictions.
- Quintile monotonicity chart (`evaluation.plot_return_by_predicted_quintile`) to validate ranking quality visually.

**3. Gemini AI Commentary**
- `process_chat()` — A Messenger-style chat widget powered by the Google Gemini API. Receives a structured prompt containing current backtest metrics and feature IC data; returns natural-language market commentary.
- Chat history maintained in `st.session_state` for the session duration.

---

## `config.py`

**Purpose:** Centralised definition of the production feature set, full candidate feature pool, and base model hyperparameters.

```python
# Legacy static reference — 20 features previously selected by IC/IR analysis + correlation pruning.
# No longer the operative list; walk_forward_cv with icir_filter=True selects features per fold at runtime.
final_features = [
    'dist_52w_high', 'log_ret_skip1m', 'WQ_Alpha_024', 'log_ret_6m',
    'RSI_14', 'log_ret_3m', 'price_vol_divergence', 'dist_SMA_50',
    'dist_SMA_21', 'log_ret_1w', 'log_ret_1m', 'WQ_Alpha_028',
    'volume_surge_weekly', 'log_ret_1y', 'volatility_shock_monthly',
    'volume_surge_monthly', 'volatility_shock_weekly',
    'volatility_1w', 'volatility_1m', 'volatility_6m',
]

# Base model hyperparameters used in all published results
BASE_MODEL_PARAMS = {
    'device':                         'cuda',
    'tree_method':                    'hist',
    'objective':                      'rank:ndcg',
    'random_state':                   42,
    'n_estimators':                   100,
    'max_depth':                      4,
    'min_child_weight':               5,
    'learning_rate':                  0.05,
    'subsample':                      0.7,
    'colsample_bytree':               0.6,
    'reg_lambda':                     2.0,
    'reg_alpha':                      0.5,
    'lambdarank_pair_method':         'topk',
    'lambdarank_num_pair_per_sample': 60,
}
```

`candidate_features` is the operative feature pool passed to `walk_forward_cv`. It contains the full set of ~46 engineered features available, including all 14 WQ alpha columns and structural/context features (`turnover_12m`, `limit_bias_60d`, `herding_dispersion`, `amihud_illiquidity`). When `icir_filter=True`, each fold selects its own active subset from this pool at runtime — no upfront manual selection needed.

---

## Data Flow Summary

```
get_tags()
    └─► update_market_data() ──► market_data.parquet  (raw OHLCV)
                                        │
                                        ▼
                               build_features(df_raw)
                               (features.py)
                               ├── Calls WorldQuantAlphas.generate_all() internally
                               │   → WQ_Alpha_001/002/006/007/013/016/024/028/040/101/103/200/201/202
                               ├── Computes all technical features
                               └── Shifts all features 1 day (look-ahead prevention)
                                        │
                                        ▼
                               build_targets() → next_1m_ret, next_1w_ret
                               target_generating_ranking() → qid, target_quintile
                                        │
                            ┌───────────┴──────────────────────┐
                            │                                  │
                   (optional)                                  │
              search_best_roi_and_sharpe()             config.candidate_features
              (feature_search.py)                      (full feature pool)
              → 127 group combinations                         │
                            │                                  │
                            └──────────────┬───────────────────┘
                                           │
                              walk_forward_cv(df, features,
                                initial_train_months=24,
                                test_months=6, gap_days=21,
                                icir_filter=True,
                                corr_prune=True)
                              (models.py)
                              ├── Per-fold IC/IR filter → active feature subset
                              ├── Per-fold correlation pruning → deduplicated subset
                              ├── model='xgboost': XGBoost LambdaRank only  ✅ Production
                              └── model='alphaforge': AlphaForge Ridge combiner  🔬 Experimental
                                           │
                         ┌─────────────────┴──────────────────┐
                         │                                    │
              compute_top_quantile_win_rate()    generate_paper_trade_signals()
              compute_model_ic()                 (inference.py — live signals)
              simulate_portfolio()
              print_performance_report()
              (evaluation.py — backtest)
                         │
                         ▼
              pretrain_and_save_artifacts()
              → pretrained_predictions.parquet  ──► Hugging Face Dataset
              → pretrained_equity_curve.parquet ──► Hugging Face Dataset
                         │
                         ▼
                      app.py
               (fetches from HF at runtime, no retraining)
```