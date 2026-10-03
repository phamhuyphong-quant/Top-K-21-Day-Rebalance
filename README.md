# VN Cross-Sectional Ranking: Backtest Toolkit

A modular Python toolkit for building and backtesting **Top-K cross-sectional ranking strategies** on Vietnamese stocks. Plug in a ranking model and an optional market-level entry filter, run a walk-forward evaluation, simulate the portfolio with Vietnam-specific trading rules, and test whether differences are statistically meaningful. The same pipeline can publish a daily paper-trading signal.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cross-sectional-ranking-vn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL v3.0](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

> **Disclaimer.** For research and education only, not financial advice. Backtests are historical simulations and do not guarantee future results.

> **Public code snapshot.** This repository contains the code only. The raw market data and the live daily pipeline (data collection, retraining, signal generation) run from a private companion repository. GitHub Actions here are disabled, and the notebooks read datasets hosted on Kaggle that are not included. To run the toolkit, collect your own data with `src/data_collect.py` (step 1 below).

## What it does

```
 data ─► features ─► walk-forward ranking ─► entry filter ─► portfolio simulation ─► evaluation & tests
```

| Stage | What you get | Module |
|---|---|---|
| Data | Incremental OHLCV and VNINDEX download via `vnstock`, with cleaning rules | `src/data_collect.py` |
| Features | ~40 look-ahead-safe features, WorldQuant-style alphas, market regime columns, forward-return targets | `src/features.py`, `src/alpha_mining.py` |
| Ranking | Walk-forward CV (rolling or expanding) with swappable models: XGBoost LambdaRank, XGBoost MSE, Lasso, LSTM, AlphaForge combiner | `src/models.py` |
| Entry filter | Composable exposure filters evaluated on market columns | `src/simulation.py` |
| Simulation | Day-by-day Top-K portfolio with fees, tax, settlement lag and lot sizes | `src/simulation.py` |
| Evaluation | IC, NDCG, precision@k, CAGR, Sharpe, Sortino, Calmar, drawdown, equity curves | `src/evaluation.py` |
| Significance | Block bootstrap and permutation tests, Holm correction, Sharpe-difference test | `src/significance_test.py` |
| Live signal | Daily Top-20 ranking, Kaggle run, Streamlit page | `src/inference.py`, `kaggle_kernel/`, `src/app.py` |

Every feature at date *T* uses only data up to the close of *T-1*, and all feature selection inside a fold is fitted on that fold's training window only.

## Installation

```bash
git clone https://github.com/phamhuyphong-quant/Top-K-21-Day-Rebalance.git
cd Top-K-21-Day-Rebalance
pip install -r requirements-dev.txt     # full toolkit (torch, gplearn, statsmodels, ...)
pip install -r requirements.txt         # Streamlit page only
```

`vnstock` comes from `https://vnstocks.com/api/simple` (set in `requirements-dev.txt`). `BASE_MODEL_PARAMS` uses `device='cuda'`; change it to `'cpu'` without a GPU.

## Pipeline walkthrough

### 1. Data: `src/data_collect.py`

```python
from src.data_collect import get_tags, update_market_data, fetch_indicator_data

update_market_data("market_data.parquet", get_tags(), start_date="2016-01-01")
fetch_indicator_data("VNINDEX", "2016-01-01", "vnindex_data.parquet")
```

This builds the dataset the rest of the pipeline expects (columns `Symbol`, `date`, `open`, `high`, `low`, `close`, `volume`). It is not included in this repository, and a full download is slow because the script pauses between symbols to respect `vnstock` rate limits. `get_tags()` returns a built-in ticker list; `get_tags(fetching=True)` pulls the live `VNALL` members. Up-to-date symbols are skipped, others are re-fetched and merged by `(date, Symbol)`. `clean_ohlcv` repairs rows that break OHLC rules.

### 2. Features and targets: `src/features.py`

```python
from src.features import build_features

df_train  = build_features(df_raw, adtv_limit=2_500_000)                         # with targets
df_latest = build_features(df_raw, adtv_limit=2_500_000, generate_target=False)  # keeps today
```

- Feature families: returns, volatility, SMA/EMA distance, volume surge, RSI, OBV, WorldQuant alphas, structural features.
- Market columns: `regime_bucket_monthly` (dispersion quartile, `Q1` lowest), `market1m_ema21`, `market3m_ema63`, breadth.
- Label: `target_magnitude`, the next 21-day log return over trailing volatility, min-max scaled to 0–100 per date.
- `adtv_limit` flags illiquid rows: excluded from cross-sectional statistics, removed once at the end.

### 3. Walk-forward ranking: `src/models.py`

```python
from src.models import walk_forward_cv
from config import candidate_features, BASE_MODEL_PARAMS

df_pred, fold_ndcg = walk_forward_cv(
    df_train, candidate_features,
    model="xgboost_ndcg",        # xgboost_mse | linear | lstm_mse | alphaforge
    model_params=BASE_MODEL_PARAMS,
    window_mode="rolling",       # or "expanding"
    initial_train_dates=756, train_percent=0.8, gap_dates=21,
    corr_prune=True,             # optional: icir_filter=True
)
```

- Window lengths are counts of trading dates. `RollingWindow` and `ExpandingWindow` are `WindowPolicy` classes, so new window schemes are one subclass.
- `gap_dates` separates train and test to avoid target overlap.
- Optional per-fold feature selection: `icir_filter` (|IC IR| threshold) and `corr_prune` (drop correlated features inside a `FEATURE_GROUPS` group).
- Output `df_pred` holds every test row with `pred_score`; `fold_ndcg` has one NDCG per fold.
- To add a model: write `predict_<name>(fold, features, params)` returning the test frame with `pred_score`, and register it in `MODEL_REGISTRY`.

### 4. Simulation and entry filters: `src/simulation.py`

```python
from src.simulation import build_data_engines, OrderManager, FilterGroup, StepFilter

raw_e, pred_e, cond_e = build_data_engines(df_prices, df_pred, df_pred)

p1 = FilterGroup(
    condition="market1m_ema21-market3m_ema63<0 and regime_bucket_monthly=='Q1'",
    transformations=[StepFilter(0)],          # 0 = no new buys while the condition holds
)
om = OrderManager(initial=500_000, topk=20, df_raw=raw_e, df_predict=pred_e,
                  df_condition=cond_e, regime_filter=p1)
om.run_strategy(allocation_strategy="equal")  # or "rank_weighted"
history = pd.DataFrame(om.portfolio.history, columns=["date", "total_value"])
```

| Component | Role |
|---|---|
| `DataEngine`, `build_data_engines` | Aligned, cursor-based data access with date-set checks (no look-ahead) |
| `Portfolio` | Cash and positions, 0.1% fee per side, 0.1% sell tax, 3-day settlement lag |
| `OrderManager` | Rebalances every 21 days into the top-`topk` by `pred_score`, 100-share lots; sells first, buys 3 days later |
| `StepFilter`, `RampFilter`, `FilterGroup` | Exposure multipliers; a `FilterGroup` applies its transformations when its condition string is true |
| `NullHypothesisTest` | Same engine with random picks, for Monte Carlo baselines |

The filter multiplier scales the buy shortfall on the buy day. No slippage is modelled.

### 5. Evaluation: `src/evaluation.py`

```python
import src.evaluation as ev

ev.print_performance_report(history, rf_annual=0.045)   # return, CAGR, Sharpe, Sortino, MDD, Calmar
ev.compute_model_ic(df_pred)                            # per-date Spearman IC
ev.plot_equity_curves(history_a, history_b, labels=["A", "B"])
```

Also available: feature IC/IR plots, win rate of top quantile, return by predicted quintile, feature importances.

### 6. Significance tests: `src/significance_test.py`

```python
from src.significance_test import run_analysis, sharpe_diff_block_bootstrap

dfs = {"ndcg": df_a, "mse": df_b}          # each needs date, pred_score, target_magnitude
run_analysis(dfs, base_model="ndcg", mode="both", k=20, block=21)
sharpe_diff_block_bootstrap(history_a, history_b, block=21)
```

Compares a base model against others and each model against its universe, using block resampling (forward returns overlap across days), Holm correction, autocorrelation diagnostics and offset-sensitivity checks.

### 7. Live signal: `src/inference.py`

```python
from src.inference import generate_paper_trade_signals

buy, hold, sell, gone, ranked = generate_paper_trade_signals(
    df=train_plus_latest_row, current_portfolio=[], features=candidate_features,
    buy_n=20, trend_filter_col=None, target_col="target_magnitude", corr_prune=True,
)
```

Retrains on the latest 735 labelled trading dates and ranks today's stocks. `kaggle_kernel/kernel.py` wraps this daily with the entry-filter check and uploads `today_signals.parquet` to Hugging Face; `streamlit run src/app.py` displays it.

## Configuration: `config.py`

`candidate_features`, `FEATURE_GROUPS` (pruning groups), `BASE_MODEL_PARAMS` (XGBoost LambdaRank defaults), `usedSymbols` (the 280-symbol universe used by the live run), `groups_to_features()`.

## Repository layout

```
config.py
src/            data_collect · features · alpha_mining · models · simulation
                evaluation · significance_test · inference · deep_combiner · app
kaggle_kernel/  daily production run
notebooks/      significance-test notebook; research notebooks and paper in "NTH RESEARCH/"
.github/workflows/  daily_update · precompute_model · keep_alive
MODULES.md      per-module function reference
```

## Automation (disabled in this repo)

The workflows are kept for documentation. They reference a `data-storage` branch and secrets that exist only in the private companion repo, and `kaggle_kernel/kernel.py` clones that private repo, so none of this runs from the public snapshot.

| Workflow | Trigger | What it does in the private repo |
|---|---|---|
| `daily_update.yml` | Manual | Updates market data on the `data-storage` branch, then triggers the next workflow |
| `precompute_model.yml` | Manual or after data update | Pushes `kernel.py` to Kaggle and waits for it to finish |
| `keep_alive.yml` | Every 6 hours | Keeps the Streamlit app awake |

Secrets used there: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GH_PAT`, `HF_TOKEN`.

## Research background

The project grew out of a research paper (Vietnamese) in `notebooks/NTH RESEARCH/`, which evaluates this pipeline's models and entry filters. Details live there; this repo is the tooling.

## Author and license

**Phong Phạm Huy**: [LinkedIn](https://www.linkedin.com/in/phong-phạm-huy-b64331377) · [GitHub](https://github.com/phamhuyphong-quant)

GPL v3.0, see [LICENSE](LICENSE). Built on `vnstock`, XGBoost, scikit-learn, PyTorch, `gplearn` (GPL v3), AlphaForge (arXiv:2406.18394) and Kakushadze's *101 Formulaic Alphas*.