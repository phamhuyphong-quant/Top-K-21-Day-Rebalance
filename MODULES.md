# Module Documentation — VN100 Cross-Sectional Ranking System

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

**Purpose:** Constructs the VN100 investable universe and fetches/updates OHLCV market data. Supports multiple data sources (`VCI`, `KBS`) with automatic fallback, OHLC sanity cleaning, incremental merging, and logging.

### Functions

---

#### `clean_symbols(symbol_list) → list[str]`

Sanitises a raw list of ticker symbols by stripping whitespace, removing NaN values, deduplicating, and returning a sorted list.

**Parameters:**
- `symbol_list` — Raw list of symbols from a data source (may contain NaN or whitespace)

**Returns:** Sorted, deduplicated list of clean symbol strings.

---

#### `get_tags(fetching=False) → list[str]`

Returns the VN100 universe.

- If `fetching=False` (default): returns a **hardcoded static list** of ~100 tickers — fast and offline-safe, used by default in most workflows.
- If `fetching=True`: queries the KBS data source live via `vnstock` to combine VN30 + VNMidCap dynamically.

**Returns:** Sorted list of ~100 ticker symbols forming the VN100 universe.

**Example:**
```python
from src.data_collect import get_tags

# Fast path — use hardcoded list (default)
symbols = get_tags()

# Live path — query KBS for current index constituents
symbols = get_tags(fetching=True)
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

> **Important:** `build_features()` calls `WorldQuantAlphas.generate_all()` internally. Do **not** call `WorldQuantAlphas` manually on `df_raw` before passing it to `build_features()` — this will compute the alpha columns twice.

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

After computing all features, **every feature column is shifted forward by 1 day per symbol** using a grouped shift, preventing same-day look-ahead. Rows with any `inf` or `NaN` in feature columns are then dropped. Dates where fewer than `min_stocks_per_date` symbols survive are also removed (logged as a warning).

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

> **Note:** Of the 14 alphas, `WQ_Alpha_024` and `WQ_Alpha_028` appear in `config.final_features` (the production feature set). The full set is available in `config.candidate_features` for feature search experiments.

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
| `wq_features` | `WQ_Alpha_024`, `WQ_Alpha_028`, and other WQ alphas in `candidate_features` |
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
    backtest_kwargs=dict(buy_fraction=0.20, trend_filter_col='dist_SMA_100'),
)
print(best_roi_row['combo_name'], best_roi_row['roi_%'])
```

---

## `models.py`

**Purpose:** Model training, walk-forward cross-validation, Optuna hyperparameter search, and the AlphaForge (GP + LSTM) training loop.

### Functions

---

#### `walk_forward_cv(df, features, model_params=None, initial_train_months=12, test_months=6, gap_days=21, model='basic', use_gp=False, liquidity_filter=False) → DataFrame`

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
| `model` | `'basic'` | `'basic'` for XGBoost only; `'alphaforge'` for GP + LSTM-Attention |
| `use_gp` | `False` | Enable GP alpha mining (used automatically when `model='alphaforge'`) |
| `liquidity_filter` | `False` | Filter out illiquid stocks before training |

> **Note:** The function default of `initial_train_months=12` is a code fallback only. Always pass `initial_train_months=24` to reproduce the reported backtest results (87-month period, 15 folds).

**Returns:** Concatenated OOS predictions DataFrame with a `pred_score` column. Pass directly to `simulate_portfolio()` or `compute_top_quantile_win_rate()`.

---

#### `optimize_xgboost_ranker(df, features, n_trials=50) → dict`

Uses Optuna to find the best XGBoost hyperparameters. Trains on data up to 2023-12-11 and validates on 2024, keeping 2025+ as a held-out test set. Maximises mean NDCG across the validation period.

**Returns:** `dict` of best hyperparameters, ready to pass as `model_params` to `walk_forward_cv`.

---

#### `train_mega_combiner(train_df, alpha_cols, epochs=5) → DynamicAlphaCombiner`

Trains the LSTM-Attention model (see `deep_combiner.py`) on the training slice to produce a `Mega_Alpha` scalar from a set of alpha signals. Uses Adam optimiser and MSE loss against `risk_adj_ret`. Called internally by the AlphaForge path in `walk_forward_cv`.

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

Plots one or more equity curves on the same chart. Pass `normalize=True` to show cumulative return (%) normalised to a common start — useful for comparing Basic vs AlphaForge.

**Example:**
```python
ev.plot_equity_curves(result_basic, result_alphaforge,
                      labels=['Basic XGBoost', 'AlphaForge'],
                      normalize=True)
```

---

#### `simulate_portfolio(df, model, features, initial_capital, buy_fraction, time_of_rebalance, trend_filter_col, settlement_delay, vnindex_df, vol_lookback, vol_percentile, vol_window) → DataFrame`

Simulates a realistic portfolio with VN-market timing conventions.

**Execution model:**
1. **Rebalance day (day 0):** Model scores stocks → ranks universe → sells exiting positions at today's price; sell proceeds enter `pending_cash` (available after T+`settlement_delay`)
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
| `vnindex_df` | `None` | Optional VNINDEX DataFrame for volatility regime filtering |
| `vol_lookback` | `21` | Days for realised VNINDEX volatility calculation |
| `vol_percentile` | `0.80` | Vol percentile threshold; above this = high-vol regime → rebalance skipped |
| `vol_window` | `252` | Rolling window for computing the percentile benchmark |
| `liquidity_filter` | `False` | Whether to filter out low-liquidity stocks before ranking |

> **Volatility Regime Filter:** When `vnindex_df` is provided, rebalance months where VNINDEX realised vol exceeds the `vol_percentile` of its own history are skipped entirely. The equity curve is still recorded continuously. In the 87-month backtest, 19 months were skipped by this filter (predominantly the COVID drawdown in 2020 and the 2022 correction).

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
- `use_mega_alpha` — Reserved for future use; currently always runs the Basic XGBoost path
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

**Purpose:** Experimental LSTM-Attention model that learns to dynamically weight a set of alpha signals based on market context. Used by the AlphaForge path in `walk_forward_cv`.

**Status:** Experimental. In walk-forward backtesting, AlphaForge produced a higher IC IR (0.683 vs 0.337) but weaker risk-adjusted returns — Sharpe 0.37 vs 0.53, monthly win rate 52.87% vs 59.77%. The production pipeline uses the Basic XGBoost model only. The `use_mega=True` path in `inference.py` is a placeholder stub not yet connected to this model.

### `class DynamicAlphaCombiner(nn.Module)`

**Architecture:**
1. **Context LSTM** — Reads a sequence of alpha values (`input_size=num_alphas`, `hidden_size=16`) and encodes the market state into a 16-dimensional hidden vector
2. **Attention Scoring** — A two-layer MLP (`Linear(16→8) → Tanh → Linear(8→num_alphas) → Softmax`) maps the hidden state to a probability distribution over alpha signals
3. **Weighted Sum** — Current-step alpha values are combined using the attention weights to produce a single `Mega_Alpha` scalar

**Constructor:**
```python
model = DynamicAlphaCombiner(num_alphas=14)  # one per WQ alpha column
```

**Forward pass:**
```python
mega_alpha, attention_weights = model(alphas_seq)
# alphas_seq: Tensor of shape [batch, time_steps, num_alphas]
# mega_alpha: Tensor of shape [batch] — the combined alpha score
# attention_weights: Tensor of shape [batch, num_alphas]
```

**Training:** See `train_mega_combiner()` in `models.py`. Trained with Adam + MSE loss against `risk_adj_ret`. In the walk-forward loop, the model receives a single time-step (`unsqueeze(1)`) rather than a true sequence — this simplification limits temporal modelling capacity and is a known limitation of the current implementation.

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
# Production feature set — 20 features selected by IC/IR analysis + correlation pruning
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

`candidate_features` contains the full set of ~46 engineered features available for feature group search experiments, including all 14 WQ alpha columns and structural/context features (`turnover_12m`, `limit_bias_60d`, `herding_dispersion`, `amihud_illiquidity`).

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
              search_best_roi_and_sharpe()             config.final_features
              (feature_search.py)                      (20 features)
              → 127 group combinations                         │
                            │                                  │
                            └──────────────┬───────────────────┘
                                           │
                              walk_forward_cv(df, features,
                                initial_train_months=24,
                                test_months=6, gap_days=21)
                              (models.py)
                              ├── Basic: XGBoost LambdaRank only  ✅ Production
                              └── AlphaForge: + GP alphas + LSTM-Attention  🔬 Experimental
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
              → pretrained_model.json           ──► Hugging Face Dataset
                         │ 
                         ▼
                      app.py
               (fetches from HF at runtime, no retraining)
```