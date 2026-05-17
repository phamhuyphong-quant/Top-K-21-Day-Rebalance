# Module Documentation — VN100 Cross-Sectional Ranking System

This document describes each module in `src/`, what it does, and how to use it. It is intended for contributors, researchers, or anyone extending the pipeline.

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

**Purpose:** Constructs the VN100 investable universe and fetches/updates OHLCV market data. Supports multiple data sources (`VCI`, `KBS`) with automatic fallback, OHLC sanity cleaning, incremental merging, and logging.

### Functions

---

#### `clean_symbols(symbol_list) → list[str]`

Sanitises a raw list of ticker symbols by stripping whitespace, removing NaN values, deduplicating, and returning a sorted list.

**Parameters:**
- `symbol_list` — Raw list of symbols from a data source (may contain NaN or whitespace)

**Returns:** Sorted, deduplicated list of clean symbol strings.

---

#### `build_vn100(fetching=False) → list[str]`

Returns the VN100 universe.

- If `fetching=False` (default): returns a **hardcoded static list** of 100 tickers — fast and offline-safe, used by default in most workflows.
- If `fetching=True`: queries the KBS data source live via `vnstock` to combine VN30 + VNMidCap dynamically.

**Returns:** Sorted list of ~100 ticker symbols forming the VN100 universe.

**Example:**
```python
from src.data_collect import build_vn100

# Fast path — use hardcoded list (default)
symbols = build_vn100()

# Live path — query KBS for current index constituents
symbols = build_vn100(fetching=True)
# Logs: VN30=30  VNMID=70  VN100=97
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

### Functions

---

#### `seed_everything(seed=42) → None`

Sets random seeds for `random`, `numpy`, and `torch` (including CUDA) to ensure reproducibility across all modules. Called at import time throughout `src/`.

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

Computes multi-horizon log returns. Note: returns are computed relative to the *current* close (no extra lag here); the 1-day lag applied in `build_features()` ensures look-ahead safety.

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

Computes two derived volume features that go beyond raw surge ratios.

**Adds columns:**
- `obv_trend` — On-Balance Volume trend normalised by 21-day average volume: `sign(Δclose) × volume` rolled over 21 days, divided by `vol_1m`
- `price_vol_divergence` — `log_ret_1m / (volume_surge_monthly + 0.001)` — flags stocks where price is rising on declining volume (a warning signal)

> **Note:** `volume_quality()` depends on `volume()` and `return_ln()` having been called first (it uses `vol_1m` and `log_ret_1m`). `build_features()` handles this ordering automatically.

---

#### `build_features(df, min_stocks_per_date=50) → DataFrame`

Master pipeline that calls all feature functions in the correct order: `return_ln → volatility → MA → volume → rsi → volume_quality → price_structure`.

After computing all features, **every feature column is shifted forward by 1 day per symbol** using a grouped shift, preventing same-day look-ahead. Rows with any `inf` or `NaN` in feature columns are then dropped. Dates where fewer than `min_stocks_per_date` symbols survive are also removed (logged as a warning).

**Returns:** Cleaned DataFrame with all features. Note that forward-return targets (`next_1m_ret`, `next_1w_ret`) are **not** added here — call `build_targets()` separately.

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

**Purpose:** Implements six WorldQuant-style quantitative alpha factors and a genetic programming (GP) framework for discovering new ones.

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

Computes six WorldQuant alpha factors. On initialisation, the DataFrame is sorted by `['Symbol', 'date']`. OHLCV data is used as-is — look-ahead is prevented upstream by the 1-day shift applied in `build_features()`.

**Constructor:**
```python
wq = WorldQuantAlphas(df)  # df must have: Symbol, date, open, high, low, close, volume
```

**Alpha Factors:**

| Method | Output Column | Formula Logic |
|---|---|---|
| `get_alpha_006()` | `WQ_Alpha_006` | `-corr(open, volume, 10)` — Negative open-volume correlation over 10 days |
| `get_alpha_012()` | `WQ_Alpha_012` | `sign(Δvolume) × (-Δclose)` — Buys when price drops but volume rises (supply exhaustion) |
| `get_alpha_024()` | `WQ_Alpha_024` | Conditional mean-reversion: compares 100-day price trend direction to decide between reversion and momentum |
| `get_alpha_028()` | `WQ_Alpha_028` | `corr(adv20, low, 5) + midprice - close` — Volume-adjusted midpoint deviation |
| `get_alpha_053()` | `WQ_Alpha_053` | `-Δ((close-low - high-close) / (close-low), 9)` — Momentum of intraday pressure |
| `get_alpha_060()` | `WQ_Alpha_060` | Simplified directional money flow: `((close-low - high-close) / range) × volume` |

**`generate_all() → DataFrame`**

Computes all six alphas, replaces `inf`/`NaN` with `NaN`, and returns a DataFrame with columns `WQ_Alpha_006`, `WQ_Alpha_012`, `WQ_Alpha_024`, `WQ_Alpha_028`, `WQ_Alpha_053`, `WQ_Alpha_060`. This DataFrame should be merged into the raw OHLCV DataFrame *before* calling `build_features()`.

```python
wq = WorldQuantAlphas(df_raw)
wq_cols_df = wq.generate_all()
for col in wq_cols_df.columns:
    df_raw[col] = wq_cols_df[col]
```

> **Note:** The production `final_features` in `config.py` uses three of these: `WQ_Alpha_012`, `WQ_Alpha_024`, and `WQ_Alpha_053`.

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
| `wq_features` | `WQ_Alpha_012`, `WQ_Alpha_024`, `WQ_Alpha_028`, `WQ_Alpha_053`, `WQ_Alpha_060` |
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

Runs the search once and identifies the best combination for both ROI and Sharpe ratio simultaneously, avoiding the need to run two separate searches.

**Returns:** `(best_roi_row, best_sharpe_row, results_df)`

**Example:**
```python
from src.feature_search import search_best_roi_and_sharpe

best_roi_row, best_sharpe_row, result_df = search_best_roi_and_sharpe(
    df,
    initial_capital=10_000,
    walk_forward_kwargs=dict(initial_train_months=24, test_months=6, gap_days=21),
    backtest_kwargs=dict(buy_fraction=0.05, trend_filter_col='dist_SMA_100'),
)
print(best_roi_row['combo_name'], best_roi_row['roi_%'])
```

---

## `models.py`

**Purpose:** Model training, walk-forward cross-validation, Optuna hyperparameter search, and the LSTM sequence combiner training loop.

### Functions

---

#### `base_model() → dict`

Returns the default XGBoost LambdaRank hyperparameters used in production:

```python
{
    'tree_method': 'hist',
    'objective': 'rank:ndcg',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'random_state': 42,
}
```

---

#### `testing_model() → dict`

Returns an alternative hyperparameter set with `lambdarank_pair_method='topk'` and `lambdarank_num_pair_per_sample=10`, intended for experimental comparisons.

---

#### `walk_forward_cv(df, features, model_params=None, initial_train_months=12, test_months=6, gap_days=21, callback=None, use_mega=False, use_gp=False) → DataFrame`

The core training and evaluation function. Simulates live deployment by rolling through time:

1. Train on all data up to `train_cutoff` (= `test_start - gap_days`)
2. Evaluate on `test_start` → `test_start + test_months`
3. Advance the window and repeat

At each fold, the model is fitted on the training slice and predictions are stored as `pred_score` in the OOS test DataFrame. Per-fold NDCG is computed and printed.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `initial_train_months` | `12` | Months of data required before first test fold. **The notebooks and all published results use `24` — always pass this explicitly.** |
| `test_months` | `6` | Length of each test window |
| `gap_days` | `21` | Trading day gap between train end and test start |
| `callback` | `None` | Optional `fn(fold, total_folds, message)` for UI progress updates |
| `use_mega` | `False` | Enable LSTM sequence combiner (experimental, unstable OOS) |
| `use_gp` | `False` | Enable GP alpha mining per fold (experimental) |

> **Note:** The function default of `initial_train_months=12` is a code default only. Always pass `initial_train_months=24` to reproduce the reported backtest results.

**Returns:** Concatenated OOS predictions DataFrame with a `pred_score` column. Pass directly to `simulate_portfolio()` or `compute_top_quantile_win_rate()`.

---

#### `optimize_xgboost_ranker(df, features, n_trials=50) → dict`

Uses Optuna to find the best XGBoost hyperparameters. Trains on data up to 2023-12-11 and validates on 2024, keeping 2025+ as a held-out test set. Maximises mean NDCG across the validation period.

**Returns:** `dict` of best hyperparameters, ready to pass as `model_params` to `walk_forward_cv`.

---

#### `train_mega_combiner(train_df, alpha_cols, epochs=5) → DynamicAlphaCombiner`

Trains the LSTM sequence model (see `deep_combiner.py`) on the training slice to produce a `Mega_Alpha` scalar from a set of alpha signals. Uses Adam optimiser and MSE loss against `risk_adj_ret`.

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

Computes the overall model-level Information Coefficient: the Spearman rank correlation between the model's `pred_score` and actual `next_1m_ret` across all OOS predictions, computed daily then averaged.

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

Plots and prints Information Coefficient (IC) statistics for each feature:

- **IC Mean** — Average Spearman correlation with `target_col` across all test dates
- **IC Std** — Volatility of that correlation over time

**Returns:** DataFrame with columns `IC Mean`, `IC Std` indexed by feature name.

---

#### `plot_feature_ir(test_df, features, target_col='next_1m_ret') → DataFrame`

Extends `plot_feature_ic` to also compute and plot **IC IR** (IC Mean / IC Std) — the signal-to-noise ratio of each feature. Features with IC IR > 0.5 are generally considered worth retaining.

---

#### `plot_feature_rolling_ir(test_df, feature, target_col='next_1m_ret', window=6) → None`

Plots the rolling IC IR for a single feature over time (default: 6-month rolling window). Useful for diagnosing whether a feature's predictive power is stable or regime-dependent.

---

#### `plot_feature_importances(model, features) → None`

Plots a horizontal bar chart of XGBoost gain-based feature importance.

---

#### `plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None) → None`

Plots a bar chart of average forward return by predicted quintile (1 = worst to 5 = best). A well-calibrated ranker should show a strictly increasing return from quintile 1 to 5. Supports both static and dynamic modes.

---

#### `plot_equity_curves(*results, labels=None, normalize=False) → None`

Plots one or more equity curves on the same chart. Pass `normalize=True` to show cumulative return (%) normalised to a common start — useful for comparing the baseline XGBoost model against the experimental LSTM model.

**Example:**
```python
ev.plot_equity_curves(result_basic, result_mega,
                      labels=['XGBoost Baseline', 'LSTM Advanced'],
                      normalize=True)
```

---

#### `simulate_portfolio(df, model, features, initial_capital, buy_fraction, time_of_rebalance, trend_filter_col, settlement_delay, vnindex_df, vol_lookback, vol_percentile, vol_window) → DataFrame`

Simulates a realistic portfolio with VN-market timing conventions.

**Execution model:**
1. **Rebalance morning (day 0):** Model scores stocks → ranks universe → sells exiting positions at today's price; sell proceeds enter `pending_cash` (available after T+`settlement_delay`)
2. **Settlement (day +3):** Pending cash becomes available → new buy orders execute at settlement-day prices

This correctly models VN T+3 settlement — you cannot buy with money from the same-day sell.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Trained XGBRanker; pass `None` to use existing `pred_score` column |
| `buy_fraction` | `0.05` | Top X% of ranked stocks are buy targets |
| `time_of_rebalance` | `'M'` | Pandas period alias: `'M'` = monthly, `'W'` = weekly |
| `trend_filter_col` | `'dist_SMA_100'` | Stock must have this column > 1.0 to qualify as a new buy. Pass `None` to disable. |
| `settlement_delay` | `3` | Trading days between sell and cash availability |
| `vnindex_df` | `None` | Optional VNINDEX DataFrame for volatility regime filtering (see below) |
| `vol_lookback` | `21` | Days for realised VNINDEX volatility calculation |
| `vol_percentile` | `0.80` | Vol percentile threshold; above this = high-vol regime → rebalance skipped |
| `vol_window` | `252` | Rolling window for computing the percentile benchmark |

> **Note:** `hold_fraction`, `trailing_stop`, and `take_profit` are **not** parameters of the current `simulate_portfolio()` implementation. The sell logic is purely rank-based: a stock is sold if it no longer appears in the top `buy_fraction` targets.

> **Volatility Regime Filter:** When `vnindex_df` is provided, rebalance months where VNINDEX realised vol exceeds the `vol_percentile` of its own history are skipped. NAV is still recorded to maintain a continuous equity curve.

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

Utility function used by the GitHub Actions precompute workflow. Runs the full walk-forward CV and backtest pipeline (`initial_train_months=24`, `test_months=6`, `gap_days=21`) and saves three artifacts so the Streamlit app can load them instantly without retraining.

**Parameters:**
- `df` — Fully processed DataFrame (features + targets + `qid` must already exist)
- `selected_features` — List of feature column names to train on
- `use_mega_alpha` — Reserved for future use; currently always runs XGBoost only
- `output_dir` (`str`, default `"data/pretrained/"`) — Directory where artifacts are saved
- `vnindex_df` — Reserved for future use; currently unused

**Returns:** `(predictions_path, equity_curve_path, final_model_path)` — paths to the three saved artifacts:
- `pretrained_predictions.parquet` — Full OOS predictions DataFrame
- `pretrained_equity_curve.parquet` — Equity curve from the backtest
- `pretrained_model.json` — Final XGBoost model trained on all available data

---

## `inference.py`

**Purpose:** Generates live paper-trading signals for a given portfolio by training on all available history and scoring today's universe.

### Functions

---

#### `generate_paper_trade_signals(df, current_portfolio, features, use_mega=False, model=None, buy_n=10, trend_filter_col='dist_SMA_100', trend_filter_threshold=1.0, target_col='target_quintile') → tuple`

The main inference function. If no pretrained `model` is passed, trains a fresh XGBoost ranker on all historical data up to (but not including) today's date, then scores today's VN100 universe.

**Signal generation logic:**

1. **Portfolio review** — For each currently held symbol:
   - If not in today's VN100 → `NOT_VN100`
   - If rank > `buy_n` → `SELL`
   - Otherwise → `HOLD`

2. **New buy candidates** — From the top `buy_n` ranked stocks today:
   - Skip if already held
   - Skip if `trend_filter_col` value ≤ `trend_filter_threshold`
   - Otherwise → `BUY`

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `buy_n` | `10` | Top N stocks targeted for new entries; also the hold threshold |
| `trend_filter_col` | `'dist_SMA_100'` | Feature column for the trend filter |
| `trend_filter_threshold` | `1.0` | Stock must be above this value to qualify as a new buy |
| `use_mega` | `False` | Not yet implemented — raises `NotImplementedError` if `True` |
| `model` | `None` | Pass a pretrained `XGBRanker` to skip training (used by the dashboard) |

> **Note:** `use_mega=True` currently raises `NotImplementedError`. The LSTM inference path is not yet implemented in this function.

**Returns:** `(buy_list, hold_list, sell_list, not_vn100_list, ranked_today_df)`

where `ranked_today_df` has columns `Symbol`, `live_score`, `rank`.

---

#### `get_actionable_portfolio_lists(df, current_portfolio, features, **kwargs) → dict`

Thin wrapper around `generate_paper_trade_signals` that returns a clean dictionary:

```python
{
    "BUY": [...],
    "HOLD": [...],
    "SELL": [...],
    "NOT_VN100": [...],
    "Rankings": DataFrame
}
```

---

## `deep_combiner.py`

**Purpose:** Experimental LSTM sequence model that learns to dynamically weight a set of alpha signals based on market context.

**Status:** Experimental. In walk-forward backtesting this model was found to be less stable than the baseline XGBoost ranker — it exhibited erratic equity curves, significant sensitivity to hyperparameters, and overfitting to specific historical market regimes. The production pipeline uses XGBoost only. The `use_mega=True` path in `inference.py` is a placeholder stub and is not yet connected to this model.

### `class DynamicAlphaCombiner(nn.Module)`

**Architecture:**
1. **Context LSTM** — Reads a sequence of alpha values (`input_size=num_alphas`, `hidden_size=16`) and encodes the market state into a 16-dimensional hidden vector
2. **Attention Scoring** — A two-layer MLP (`Linear(16→8) → Tanh → Linear(8→num_alphas) → Softmax`) maps the hidden state to a probability distribution over alpha signals
3. **Weighted Sum** — Current-step alpha values are combined using the attention weights to produce a single `Mega_Alpha` scalar

**Constructor:**
```python
model = DynamicAlphaCombiner(num_alphas=6)  # one per WorldQuant alpha column
```

**Forward pass:**
```python
mega_alpha, attention_weights = model(alphas_seq)
# alphas_seq: Tensor of shape [batch, time_steps, num_alphas]
# mega_alpha: Tensor of shape [batch] — the combined alpha score
# attention_weights: Tensor of shape [batch, num_alphas]
```

**Training:** See `train_mega_combiner()` in `models.py`. Trained with Adam + MSE loss against `risk_adj_ret`. In the walk-forward loop, the model is given a single time-step (`unsqueeze(1)`) rather than a true sequence — this simplification limits the temporal modelling capacity.

---

## `app.py`

**Purpose:** The Streamlit dashboard that serves as the user-facing interface for the entire system. It is cloud-hosted on Streamlit Community Cloud and relies on GitHub Secrets (`GITHUB_TOKEN`, `GOOGLE_API_KEY`) — it is not designed for local execution.

### Structure

The app is organised into three main sections rendered on a single page:

**1. Data & Signal Loading**
- `load_data()` — Loads raw market data from the GitHub-hosted Parquet file, keyed by today's date to bust the cache daily.
- `load_pretrained()` — Loads pre-computed walk-forward OOS predictions and equity curve from the `data/pretrained/` directory (populated by the `precompute_model.yml` GitHub Actions workflow).
- `display_portfolio_signals_ui()` — Renders the live Buy / Hold / Sell signal table for a user-defined portfolio, using `inference.generate_paper_trade_signals()` on demand.

**2. Backtest & Analytics Panel**
- Strategy settings (buy fraction, trend filter, rebalance frequency) are controlled from the sidebar.
- Renders an interactive Plotly equity curve of the pre-computed backtest.
- Displays portfolio metrics (Final NAV, CAGR, Sharpe) via `st.metric`.
- Feature importance and IC charts are computed from the pre-loaded OOS predictions.
- Quintile monotonicity chart (`evaluation.plot_return_by_predicted_quintile`) validates ranking quality visually.

**3. Gemini AI Commentary**
- `process_chat()` — A Messenger-style chat widget powered by the Google Gemini API (`google-genai`). It receives a structured prompt containing the current backtest metrics and feature IC data, and returns natural-language market commentary.
- The chat history is maintained in `st.session_state` for the duration of the session.

---

## `config.py`

**Purpose:** Centralised definition of the production feature set and full candidate feature list.

```python
final_features = [
    'log_ret_6m',
    'volatility_1w', 'volatility_1m',
    'volatility_shock_monthly', 'volatility_shock_weekly',
    'dist_EMA_9',
    'volume_surge_monthly', 'volume_surge_weekly',
    'WQ_Alpha_012', 'WQ_Alpha_024', 'WQ_Alpha_053',
    'dist_EMA_100',
]
```

`dist_EMA_100` was re-introduced despite being pruned in the initial correlation analysis — ablation testing showed consistent OOS improvement, suggesting it carries complementary information within the tree ensemble.

`candidate_features` contains the full set of engineered features available for feature group search experiments.

---

## Data Flow Summary

```
build_vn100()
    └─► update_market_data() ──► market_data.parquet
                                        │
                                        ▼
                           WorldQuantAlphas.generate_all()
                           (alpha_mining.py)
                           Produces: WQ_Alpha_006/012/024/028/053/060
                                        │
                                        ▼
                               build_features()
                               build_targets()
                               (features.py)
                                        │
                                        ▼
                          target_generating_ranking()
                                        │
                            ┌───────────┴──────────────────────┐
                            │                                  │
                   (optional)                                  │
              search_best_roi_and_sharpe()             config.final_features
              (feature_search.py)                              │
              → select best feature groups                     │
                            │                                  │
                            └──────────────┬───────────────────┘
                                           │
                                  walk_forward_cv(df, features,
                                    initial_train_months=24,
                                    test_months=6, gap_days=21)
                                  (models.py)
                                           │
                         ┌─────────────────┴──────────────────┐
                         │                                    │
              compute_top_quantile_win_rate()    generate_paper_trade_signals()
              compute_model_ic()              (inference.py — live signals)
              simulate_portfolio()
              print_performance_report()
              (evaluation.py — backtest)
                         │
                         ▼
              pretrain_and_save_artifacts()
              → pretrained_predictions.parquet
              → pretrained_equity_curve.parquet
              → pretrained_model.json
                         │
                         ▼
                      app.py
              (Streamlit dashboard — loads artifacts)
```