# 📈 VN100 Cross-Sectional Ranking System

> A machine learning pipeline that ranks Vietnamese stocks in the VN100 universe by predicted forward returns, and generates actionable paper-trading signals through an interactive Streamlit dashboard.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://crosssectionalrankvn100-analyzing.streamlit.app/)

---

## 🧠 Project Overview

This project applies a **cross-sectional ranking approach** to the Vietnamese stock market. Instead of predicting absolute prices, the model ranks stocks within the VN100 universe each trading day by their expected relative performance over the next month. The top-ranked stocks are flagged as Buy signals, while lower-ranked held positions are flagged as Sell or Hold.

The core model is an **XGBoost LambdaRank** (NDCG objective), trained using walk-forward cross-validation to simulate real-world out-of-sample performance. An experimental **LSTM-Attention alpha combiner** is also included for ensemble research.

---

## 🗂️ Project Structure

```
project 1/
├── notebooks/
│   └── VN100_CROSS_SECTIONAL_RANKING/
│       ├── 01_Data_Collection.ipynb        # Data fetching & storage
│       ├── 02_Feature_Engineering.ipynb    # Feature construction & EDA
│       └── 03_Model_Training_and_Evaluation.ipynb  # Training, backtest & analysis
└── src/
    ├── data_collect.py     # VN100 universe construction & incremental data fetching
    ├── features.py         # Technical feature engineering (RSI, MA, volatility, etc.)
    ├── alpha_mining.py     # WorldQuant-style alpha factors
    ├── models.py           # XGBoost ranker, walk-forward CV, LSTM combiner
    ├── evaluation.py       # Backtest metrics, win rates, IC analysis
    ├── inference.py        # Live signal generation (Buy / Hold / Sell)
    ├── deep_combiner.py    # LSTM-Attention dynamic alpha weighting (experimental)
    └── app.py              # Streamlit dashboard
```

---

## ✨ Key Features

- **VN100 Universe Construction** — Automatically combines VN30 + VNMidCap from the VCI data source via `vnstock`.
- **Incremental Data Updates** — Smart incremental fetching; only downloads new trading days, skipping up-to-date symbols.
- **Rich Feature Set** — RSI (14-period), multi-horizon log returns (1W/1M/3M/6M/1Y), volume surge ratios, annualised volatility, Simple & Exponential Moving Averages, and WorldQuant-style alpha factors (Alpha #12, #41, #54, #101, etc.).
- **XGBoost LambdaRank** — Optimises NDCG directly for ranking quality rather than regression error.
- **Walk-Forward Validation** — Simulates live deployment; avoids look-ahead bias by strictly training only on past data.
- **IC Analysis** — Evaluates each feature's Information Coefficient (Spearman rank correlation) against future returns.
- **Live Signal Engine** — Produces daily Buy / Hold / Sell / Not-VN100 lists with a configurable grace band (`hold_n`) and trend filter.
- **Streamlit Dashboard** — Interactive UI for backtesting, signal viewing, and Gemini-powered AI commentary.

---

## 🌐 Live Demo

The application is deployed and publicly accessible:

**🔗 [https://crosssectionalrankvn100-analyzing.streamlit.app/](https://crosssectionalrankvn100-analyzing.streamlit.app/)**

The dashboard lets you:
- Run backtests interactively on the VN100 universe
- View today's Buy / Hold / Sell signals in real time
- Explore feature importance and IC charts
- Get AI-powered market commentary via Gemini

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Masterokadanori/Cross_Sectional_Rank_VN100.git
cd Cross_Sectional_Rank_VN100

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `vnstock` | Vietnamese market data |
| `xgboost` | LambdaRank model |
| `gplearn` | Symbolic regression for alpha mining |
| `torch` | LSTM-Attention combiner |
| `streamlit` | Interactive dashboard |
| `pandas`, `numpy` | Data processing |
| `optuna` | Hyperparameter optimisation |
| `google-generativeai` | Gemini AI commentary in dashboard |

---

## 🚀 Usage

### 1. Data Collection

Run `notebooks/01_Data_Collection.ipynb` or call directly:

```python
from src.data_collect import build_vn100, update_market_data

symbols = build_vn100()
update_market_data("data/market_data.parquet", symbols)
```

### 2. Feature Engineering

Run `notebooks/02_Feature_Engineering.ipynb` or:

```python
from src.features import build_features
df = build_features(raw_df)
```

### 3. Model Training & Backtest

Run `notebooks/03_Model_Training_and_Evaluation.ipynb` or:

```python
from src.models import walk_forward_cv
results = walk_forward_cv(df, features)
```

### 4. Dashboard

The dashboard is **cloud-hosted** and does not need to be run locally. Access it directly at:

**🔗 [https://crosssectionalrankvn100-analyzing.streamlit.app/](https://crosssectionalrankvn100-analyzing.streamlit.app/)**

> `app.py` relies on Streamlit Cloud secrets (`GITHUB_TOKEN`, `GOOGLE_API_KEY`) for live data fetching and AI commentary, so it is not intended for local execution.

---

## 📊 Model Architecture

```
Raw OHLCV Data
      │
      ▼
Feature Engineering
  ├── Momentum:    log returns (1W, 1M, 3M, 6M, 1Y)
  ├── Trend:       SMA/EMA crossovers, distance from MA
  ├── Volatility:  rolling std, volatility shock ratios
  ├── Volume:      surge ratios (weekly, monthly)
  ├── Oscillator:  RSI-14
  └── Alpha Factors: WorldQuant Alpha #12, #41, #54, #101
      │
      ▼
Walk-Forward Cross-Validation
  ├── Train on: months 1 → T-1 (with 30-day gap)
  └── Predict on: month T
      │
      ▼
XGBoost LambdaRank (rank:ndcg)
      │
      ▼
Daily Cross-Sectional Ranking → Buy / Hold / Sell Signals
```

---

## 📈 Signal Logic

The inference engine generates daily signals as follows:

| Signal | Condition |
|---|---|
| 🟢 **BUY** | Stock ranks in top `buy_n` (default: 5) AND passes the trend filter |
| 🔵 **HOLD** | Currently held AND ranks within top `hold_n` (default: 15) grace band |
| 🟠 **SELL** | Currently held BUT falls outside the hold grace band |
| 🔴 **NOT VN100** | Currently held BUT no longer part of today's VN100 universe |

---

## 🔬 Evaluation Metrics

- **Win Rate Lift** — Top-quintile pick win rate vs. market baseline
- **Information Coefficient (IC)** — Spearman correlation of each feature with next-month returns
- **NDCG Score** — Ranking quality metric optimised directly by the model
- **Feature Importance** — XGBoost gain-based feature attribution

---

## 👤 Author

**Phong Phạm Huy**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/phong-phạm-huy-b64331377)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Masterokadanori)

---

## 📄 License

This project is for research and educational purposes. Please contact the author before using it in commercial applications.
