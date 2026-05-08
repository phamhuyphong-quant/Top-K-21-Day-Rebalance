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
10. [Data Flow Summary](#data-flow-summary)

---

## `data_collect.py`

**Purpose:** Constructs the VN100 investable universe and fetches/updates OHLCV market data from the VCI source via `vnstock`.

### Functions

---

#### `clean_symbols(symbol_list) → list[str]`

Sanitises a raw list of ticker symbols by stripping whitespace, removing NaN values, deduplicating, and returning a sorted list.

**Parameters:**
- `symbol_list` — Raw list of symbols from a data source (may contain NaN or whitespace)

**Returns:** Sorted, deduplicated list of clean symbol strings.

---

#### `build_vn100() → list[str]`

Constructs the VN100 universe by combining the VN30 and VNMidCap indices from the VCI data source. Prints component counts to stdout.

**Returns:** Sorted list of ~100 ticker symbols forming the VN100 universe.

**Example:**
```python
from src.data_collect import build_vn100
symbols = build_vn100()
# VN30: 30 | VNMID: 70 | VN100: 97
```

---

#### `update_market_data(file_path, symbols, start_date, batch_size) → None`

Fetches daily OHLCV history for all symbols and saves to a Parquet file. Performs **incremental updates**: reads the existing file (if any), identifies the last date for each symbol, and only downloads new data — skipping symbols that are already up to date.

**Parameters:**
- `file_path` (`str`) — Path to the output `.parquet` file (e.g., `"data/market_data.parquet"`)
- `symbols` (`list[str]`) — List of ticker symbols to fetch
- `start_date` (`str`, default `"2018-01-01"`) — Earliest date for initial fetch; ignored for symbols already in the file
- `batch_size` (`int`, default `5`) — Number of symbols fetched per batch before a brief sleep, to respect API rate limits

**Notes:**
- Automatically handles weekends: if run on Monday, sets `latest_market_day` to last Friday.
- Data is written in Parquet format using PyArrow for fast I/O.

---

## `features.py`

**Purpose:** Transforms raw OHLCV data into the full feature matrix used for model training. Also generates the ranking target variable.

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

Computes multi-horizon log returns using a 1-day lag (i.e., all returns are computed relative to yesterday's close to avoid same-day look-ahead).

**Adds columns:** `log_ret_1w`, `log_ret_1m`, `log_ret_3m`, `log_ret_6m`, `log_ret_1y`

---

#### `volatility(df) → DataFrame`

Computes rolling annualised volatility at multiple horizons and volatility shock ratios (recent vol / longer-term vol).

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

#### `build_features(df) → DataFrame`

Master pipeline that calls all feature functions in the correct order: `return_ln → volatility → MA → volume → rsi → volume_quality → price_structure`. Also appends the forward-return labels used during evaluation:

- `next_1m_ret` — Log return over the next 21 trading days (the main target)
- `next_1w_ret` — Log return over the next 5 trading days

Removes rows with `inf` or `NaN` in any feature column.

**Returns:** Cleaned DataFrame with all features and forward return labels.

---

#### `target_generating_ranking(df, freq='M') → DataFrame`

Constructs the ranking target variable used to train the XGBoost ranker.

**Process:**
1. Computes `risk_adj_ret = next_1m_ret / volatility_3m` (or `next_1w_ret / volatility_1m` for weekly)
2. Bins each stock into 5 quintiles *within each trading day* using `pd.qcut` on risk-adjusted returns
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

**Purpose:** Implements WorldQuant-style quantitative alpha factors and a genetic programming (GP) framework for discovering new ones.

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

Computes nine WorldQuant alpha factors. On initialisation, OHLCV columns are shifted by 1 day (`T-1`) to prevent any same-day look-ahead.

**Constructor:**
```python
wq = WorldQuantAlphas(df)  # df must have: Symbol, date, open, high, low, close, volume
```

**Alpha Factors:**

| Method | Formula Logic |
|---|---|
| `get_alpha_012()` | `sign(Δvolume) × (-Δclose)` — Buys when price drops but volume rises (supply exhaustion) |
| `get_alpha_041()` | `√(high × low) - close` — Geometric mean of range vs. close |
| `get_alpha_054()` | `-(low - close) × open⁵ / ((low - high) × close⁵)` — Intraday open/close range anomaly |
| `get_alpha_101()` | `(close - open) / (high - low + 0.001)` — Intraday close strength (momentum) |
| `get_alpha_006()` | `-corr(open, volume, 10)` — Negative open-volume correlation |
| `get_alpha_024()` | Conditional mean-reversion: uses 100-day price trend direction |
| `get_alpha_028()` | `corr(adv20, low, 5) + midprice - close` — Volume-adjusted midpoint deviation |
| `get_alpha_053()` | `-Δ((close-low - high-close) / (close-low), 9)` — Momentum of intraday pressure |
| `get_alpha_060()` | Money flow: `((close-low - high-close) / range) × volume` |

**`generate_all() → DataFrame`**

Computes all nine alphas, handles `inf`/`NaN`, and returns a DataFrame with columns `WQ_Alpha_012`, `WQ_Alpha_041`, etc.

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
    backtest_kwargs=dict(buy_fraction=0.05, hold_fraction=0.15,
                         trend_filter_col='dist_SMA_100'),
)
print(best_roi_row['combo_name'], best_roi_row['roi_%'])
```

---

## `models.py`

**Purpose:** Model training, walk-forward cross-validation, Optuna hyperparameter search, and the LSTM-Attention combiner training loop.

### Functions

---

#### `base_model() → dict`

Returns the default XGBoost LambdaRank hyperparameters:

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

#### `alpha_model() → dict`

Returns GPU-accelerated hyperparameters (requires CUDA) with stronger regularisation, intended for the alpha combiner experiments.

---

#### `walk_forward_cv(df, features, model_params, initial_train_months, test_months, gap_days, callback, use_mega, use_gp) → DataFrame`

The core training and evaluation function. Simulates live deployment by rolling through time:

1. Train on all data up to `train_cutoff` (= `test_start - gap_days`)
2. Evaluate on `test_start` → `test_start + test_months`
3. Advance the window and repeat

At each fold, the model is fitted with the OOS test set as `eval_set`, allowing XGBoost's built-in early stopping logic.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `initial_train_months` | `12` | Months of data required before first test fold. The notebooks use `24`. |
| `test_months` | `6` | Length of each test window |
| `gap_days` | `21` | Trading day gap between train end and test start |
| `callback` | `None` | Optional `fn(fold, total_folds, message)` for UI progress updates |
| `use_mega` | `False` | Enable LSTM-Attention alpha combiner (experimental, unstable OOS) |
| `use_gp` | `False` | Enable GP alpha mining per fold (experimental) |

> **Note:** The published backtest results use `initial_train_months=24`. The function default of `12` is a code default only. Always pass `initial_train_months=24` to reproduce the reported numbers.

**Returns:** Concatenated OOS predictions DataFrame with a `pred_score` column.

---

#### `optimize_xgboost_ranker(df, features, n_trials=50) → dict`

Uses Optuna to find the best XGBoost hyperparameters. Trains on data up to 2023-12-11 and validates on 2024, keeping 2025+ as a held-out test set. Maximises mean NDCG across the validation period.

**Returns:** `dict` of best hyperparameters, ready to pass as `model_params` to `walk_forward_cv`.

---

#### `train_mega_combiner(train_df, alpha_cols, epochs=5) → DynamicAlphaCombiner`

Trains the LSTM-Attention model (see `deep_combiner.py`) on the training slice to produce a `Mega_Alpha` scalar from a set of alpha signals.

---

#### `test_train_spliter(df, test_start, features) → tuple`

Simple utility for a single train/test split at a given date (with the 30-day gap). Used for one-off experiments outside the walk-forward loop.

---

## `evaluation.py`

**Purpose:** Backtest engine, portfolio performance metrics, and feature quality metrics on OOS predictions.

### Functions

---

#### `evaluate_ranking_performance(test_df, X_test, ranker, top_quantile, ret_col) → dict`

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

#### `predicted_quintile_chart(test_df, X_test, ranker) → None`

Plots a bar chart of average forward return by predicted quintile (1 = worst to 5 = best). A well-calibrated ranker should show a strictly increasing return from quintile 1 to 5. Supports both static and dynamic modes (same as `evaluate_ranking_performance`).

---

#### `feature_influence_ic(test_df, features, target_col) → DataFrame`

Plots and prints Information Coefficient (IC) statistics for each feature:

- **IC Mean** — Average Spearman correlation with `target_col` (e.g., `next_1m_ret`) across all test dates
- **IC Std** — Volatility of that correlation over time
- **IC IR** — IC Mean / IC Std — the "signal-to-noise ratio" of each feature

Features with IC IR > 0.5 are generally considered worth retaining.

**Returns:** DataFrame with columns `IC Mean`, `IC Std`, `IC IR` indexed by feature name.

---

#### `feature_influence(model, features) → None`

Plots a horizontal bar chart of XGBoost gain-based feature importance.

---

#### `plot_rolling_ic_ir(test_df, feature, target_col, window) → None`

Plots the rolling IC IR for a single feature over time (default: 6-month rolling window). Useful for diagnosing whether a feature's predictive power is stable or regime-dependent.

---

#### `run_xgboost_backtest(df, model, features, initial_capital, buy_fraction, hold_fraction, trailing_stop, take_profit, time_of_rebalance, trend_filter_col, settlement_delay) → DataFrame`

Simulates a realistic portfolio with VN-market timing conventions.

**Execution model:**
1. **Rebalance morning (day 0):** Model scores stocks → ranks universe → sells exiting positions at today's price
2. **Settlement (day +3):** Sell proceeds become available → new buy orders execute at settlement-day prices

This correctly models VN T+2.5 settlement — you cannot buy with money from the same-day sell.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Trained XGBRanker; pass `None` to use existing `pred_score` column |
| `buy_fraction` | `0.05` | Top X% of ranked stocks are buy targets |
| `hold_fraction` | `0.15` | Top Y% of ranked stocks are hold targets (grace band, Y > X) |
| `trailing_stop` | `-0.10` | Sell if drawdown from peak ≤ this value |
| `take_profit` | `0.50` | Sell if return since buy ≥ this value |
| `time_of_rebalance` | `'M'` | Pandas period alias: `'M'` = monthly, `'W'` = weekly |
| `trend_filter_col` | `'dist_SMA_50'` | Stock must have this column > 1.0 to qualify as a new buy. Pass `None` to disable. |
| `settlement_delay` | `3` | Trading days between sell and cash availability |

**Returns:** DataFrame with columns `date`, `total_value`, `cash`, `pending_cash`, `number_of_holdings`.

---

#### `compute_metrics(result, initial_capital, rf_annual) → dict`

Computes and prints a full risk-adjusted performance report from a backtest equity curve.

**Metrics computed:** Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Monthly Win Rate, Profit Factor.

**Parameters:**
- `result` — DataFrame returned by `run_xgboost_backtest`
- `initial_capital` — Starting capital; if `None`, uses `result['total_value'].iloc[0]`
- `rf_annual` — Annual risk-free rate (default `0.045` — approximate Vietnam T-bill rate)

**Returns:** `dict` with keys `total_return`, `cagr`, `sharpe`, `sortino`, `calmar`, `max_drawdown`, `win_rate`, `profit_factor`.

---

#### `capital_over_time(result) → None`

Plots the portfolio equity curve (total value over time) from a backtest result DataFrame.

---

#### `plot_model_comparison(res_base, res_lstm) → None`

Plots two equity curves on the same chart as cumulative return (%), normalised to a common start. Used to compare the baseline XGBoost model against the experimental LSTM model.

---

#### `generate_and_save_pretrained_model(df, selected_features, use_mega_alpha, output_dir) → (str, str)`

Utility function used by the GitHub Actions precompute workflow. Runs the full walk-forward CV and backtest pipeline and saves the resulting predictions and equity curve to Parquet files so the Streamlit app can load them instantly without retraining.

**Returns:** `(predictions_path, equity_curve_path)`

---

## `inference.py`

**Purpose:** Generates live paper-trading signals for a given portfolio by training on all available history and scoring today's universe.

### Functions

---

#### `generate_paper_trade_signals(df, current_portfolio, features, use_mega, buy_n, hold_n, trend_filter_col, trend_filter_threshold, target_col) → tuple`

The main inference function. Trains on all data up to (but not including) today's date, then scores today's VN100 universe.

**Signal generation logic:**

1. **Portfolio review** — For each currently held symbol:
   - If not in today's VN100 → `NOT_VN100`
   - If rank > `hold_n` → `SELL`
   - Otherwise → `HOLD`

2. **New buy candidates** — From the top `buy_n` ranked stocks today:
   - Skip if already held
   - Skip if `trend_filter_col` value ≤ `trend_filter_threshold`
   - Otherwise → `BUY`

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `buy_n` | `5` | Number of top-ranked stocks targeted for new entries |
| `hold_n` | `15` | Grace band — hold any existing position ranked within top `hold_n` |
| `trend_filter_col` | `'dist_SMA_50'` | Feature column for the trend filter |
| `trend_filter_threshold` | `1.0` | Stock must be above this value to qualify as a new buy |
| `use_mega` | `False` | Placeholder — LSTM inference path is not yet wired up in this function |

**Returns:** `(buy_list, hold_list, sell_list, not_vn100_list, ranked_today_df)`

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

**Purpose:** Experimental LSTM-Attention model that learns to dynamically weight a set of alpha signals based on current market context.

**Status:** Experimental. In walk-forward backtesting this model was found to be less stable than the baseline XGBoost ranker — it exhibited erratic equity curves and significant sensitivity to hyperparameter tuning. The production pipeline uses XGBoost only. The `use_mega=True` path in `inference.py` is a placeholder stub and is not yet connected to this model.

### `class DynamicAlphaCombiner(nn.Module)`

**Architecture:**
1. **Context LSTM** — Reads a sequence of alpha values and encodes the market state into a 16-dimensional hidden vector
2. **Attention Scoring** — A two-layer MLP (`Linear → Tanh → Linear → Softmax`) maps the hidden state to a probability distribution over alpha signals
3. **Weighted Sum** — Current-step alpha values are combined using the attention weights to produce a single `Mega_Alpha` scalar

**Constructor:**
```python
model = DynamicAlphaCombiner(num_alphas=9)
```

**Forward pass:**
```python
mega_alpha, attention_weights = model(alphas_seq)
# alphas_seq: Tensor of shape [batch, time_steps, num_alphas]
# mega_alpha: Tensor of shape [batch] — the combined alpha score
# attention_weights: Tensor of shape [batch, num_alphas]
```

**Training:** See `train_mega_combiner()` in `models.py`. Trained with Adam + MSE loss against `risk_adj_ret`.

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
- Strategy settings (buy/hold fraction, trend filter, rebalance frequency) are controlled from the sidebar.
- Renders an interactive Plotly equity curve of the pre-computed backtest.
- Displays portfolio metrics (Final NAV, CAGR, Sharpe) via `st.metric`.
- Feature importance and IC charts are computed from the pre-loaded OOS predictions.
- Quintile monotonicity chart (`evaluation.predicted_quintile_chart`) validates ranking quality visually.

**3. Gemini AI Commentary**
- `process_chat()` — A Messenger-style chat widget powered by the Google Gemini API. It receives a structured prompt containing the current backtest metrics and feature IC data, and returns a natural-language market commentary.
- The chat history is maintained in `st.session_state` for the duration of the session.

---

## Data Flow Summary

```
build_vn100()
    └─► update_market_data() ──► market_data.parquet
                                        │
                                        ▼
                               build_features()
                               (features.py)
                                        │
                                        ▼
                           WorldQuantAlphas.generate_all()
                           (alpha_mining.py)
                                        │
                                        ▼
                          target_generating_ranking()
                                        │
                            ┌───────────┴──────────────────────┐
                            │                                  │
                   (optional)                                  │
              search_best_roi_and_sharpe()                     │
              (feature_search.py)                              │
              → select best feature groups                     │
                            │                                  │
                            └──────────────┬───────────────────┘
                                           │
                                  walk_forward_cv()
                                  (models.py)
                                           │
                         ┌─────────────────┴──────────────────┐
                         │                                    │
              evaluate_ranking_performance()    generate_paper_trade_signals()
              run_xgboost_backtest()            (inference.py — live signals)
              compute_metrics()
              (evaluation.py — backtest)
                         │
                         ▼
              generate_and_save_pretrained_model()
              (evaluation.py — saves artifacts)
                         │
                         ▼
                      app.py
              (Streamlit dashboard — loads artifacts)
```