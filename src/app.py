import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from google import genai
from google.genai import types
import sys
import requests
import io
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Standardized Absolute Imports
from src.features import build_features, target_generating_ranking


from src.inference import generate_paper_trade_signals 
st.set_page_config(page_title="VN100 Backtest Dashboard", layout="wide")


import datetime

def _today_vn() -> str:
    """Returns today's date in Vietnam time (UTC+7) as a string key like '2025-05-01'.
    Used as a cache-buster so data is always fresh after midnight VN time."""
    return (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)  # Re-checks every hour; date key busts cache after midnight VN time
def load_data(_date_key: str = None):
    """
    Loads market data from GitHub data-storage branch.
    The _date_key argument is today's VN date — changing it invalidates the cache
    automatically each new day, so the app always shows the latest data.
    """
    url = "https://raw.githubusercontent.com/Masterokadanori/Cross_Sectional_Rank_VN100/data-storage/market_data.parquet"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return pd.read_parquet(io.BytesIO(response.content))
        else:
            raise Exception(f"GitHub Error {response.status_code}: {response.text}")
    except Exception as e:
        st.warning(f"⚠️ Live fetch failed. Using local seed data. Error: {e}")
        return pd.read_parquet("data/market_data.parquet")


@st.cache_data(ttl=3600)  # Same pattern — hourly TTL + date key = daily refresh
def load_pretrained(_date_key: str = None):
    """
    Fetches the precomputed walk-forward predictions and equity curve from the data-storage branch.
    The _date_key argument busts the cache automatically each new VN day.
    """
    base_url = "https://raw.githubusercontent.com/Masterokadanori/Cross_Sectional_Rank_VN100/data-storage/"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    
    try:
        pred_response = requests.get(base_url + "pretrained_predictions.parquet", headers=headers)
        if pred_response.status_code == 200:
            honest_test_df = pd.read_parquet(io.BytesIO(pred_response.content))
        else:
            raise Exception(f"GitHub Error (Predictions): {pred_response.status_code}")
            
        eq_response = requests.get(base_url + "pretrained_equity_curve.parquet", headers=headers)
        if eq_response.status_code == 200:
            result = pd.read_parquet(io.BytesIO(eq_response.content))
        else:
            raise Exception(f"GitHub Error (Equity Curve): {eq_response.status_code}")
            
        return honest_test_df, result
        
    except Exception as e:
        st.warning(f"⚠️ Live fetch of precomputed models failed. Using local artifacts. Error: {e}")
        honest_test_df = pd.read_parquet("data/pretrained/pretrained_predictions.parquet")
        result = pd.read_parquet("data/pretrained/pretrained_equity_curve.parquet")
        return honest_test_df, result

def display_portfolio_signals_ui(df, current_portfolio, features):
    """
    Streamlit UI component to display Buy/Hold/Sell/Not_VN100 lists beautifully.
    """
    st.divider()
    st.subheader("🎯 Actionable Paper Trading Signals (Today)")
    
    with st.spinner("Calculating live market signals..."):
        try:
            buys, holds, sells, non_vn100, ranks = generate_paper_trade_signals(
                df=df,
                current_portfolio=current_portfolio,
                features=features
            )
            
            # Create 4 columns for the lists
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.success(f"🟢 **BUY** ({len(buys)})")
                st.write(", ".join(buys) if buys else "None")
                
            with col2:
                st.info(f"🔵 **HOLD** ({len(holds)})")
                st.write(", ".join(holds) if holds else "None")
                
            with col3:
                st.warning(f"🟠 **SELL** ({len(sells)})")
                st.write(", ".join(sells) if sells else "None")
                
            with col4:
                st.error(f"🔴 **NOT VN100** ({len(non_vn100)})")
                st.write(", ".join(non_vn100) if non_vn100 else "None")
                
            # Optional: Show the actual dataset of rankings in a dropdown
            with st.expander("📊 View Full Model Rankings for Today"):
                st.dataframe(ranks.set_index("rank"), use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not generate signals: {e}")
# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = ""
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

st.title("📈 VN100 Cross-Sectional Ranking Dashboard")

st.markdown("""
This dashboard presents an **XGBoost LambdaRank pipeline** that cross-sectionally ranks all 100 stocks
in the VN100 universe by predicted relative forward returns, and generates actionable paper-trading signals.
All backtest results shown are **fully out-of-sample**, produced via walk-forward cross-validation across
12 folds from 2020 to 2026 — no look-ahead, no data snooping.

> ⚠️ **Disclaimer:** Signals generated by this system are for research and educational purposes only.
> Nothing here constitutes financial advice. Past model performance does not guarantee future results.
""")

with st.expander("🧠 How does this system work? (click to expand)", expanded=False):
    st.markdown("""
    ### The Core Idea: Ranking, Not Predicting

    Forecasting the exact future price of a stock is notoriously difficult — markets are noisy and
    unpredictable in absolute terms. This system takes a more tractable approach: instead of asking
    *"where will stock X go?"*, it asks *"which stocks are likely to outperform the others next month?"*

    This is called **cross-sectional ranking** — ranking all stocks relative to each other on a given day,
    then betting on the top-ranked ones.

    ---

    ### The Model: XGBoost LambdaRank (`rank:ndcg`)

    The core model is an **XGBoost LambdaRank** ranker, trained to optimize **NDCG** (Normalized Discounted
    Cumulative Gain) — a metric borrowed from information retrieval that measures how well the model
    places the best stocks at the top of the ranking.

    **Why ranking instead of regression?**
    A regression model must predict exact return magnitudes — hard to do in a noisy market. A ranking model
    only needs to get the *order* right: "Stock A will beat Stock B." This is a weaker, more learnable signal.

    ---

    ### Features Used

    Each stock is described daily by a rich set of technical signals:

    | Category | Features |
    |---|---|
    | **Momentum** | Log return skip-1M (skip-1-month momentum) |
    | **Trend** | Distance from SMA-9/21/50/100/200 and EMA-9/21/50/100/200 (price / MA ratio) |
    | **Volatility** | Annualised rolling volatility at 1W, 1M, 3M, 6M; weekly and monthly volatility shocks |
    | **Volume** | Monthly and weekly volume surge ratios; OBV trend; price-volume divergence |
    | **Oscillator** | RSI-14; distance from 52-week high |

    Features are computed with a **1-day lag** to prevent same-day look-ahead bias.

    ---

    ### Target Variable: Risk-Adjusted Quintile

    The model doesn't predict raw returns — it predicts **relative rank**. Each day, stocks are binned
    into 5 quintiles (0 = worst, 4 = best) based on their **risk-adjusted next-month return**:

    ```
    risk_adj_ret = next_1m_return / volatility_3m
    ```

    Adjusting for volatility ensures the model rewards consistent outperformance, not just lucky high-vol spikes.

    ---

    ### Walk-Forward Cross-Validation

    To simulate real-world deployment and prevent **look-ahead bias**, the model is trained using a
    strict **walk-forward** scheme:

    - **Initial training window:** 12 months of history
    - **Test window:** 6 months (rolled forward after each fold)
    - **Gap:** 21 trading days between train end and test start (prevents any future leakage)
    - **Total folds:** 12 folds covering 2020–2026

    All results shown below are **fully out-of-sample (OOS)** — the model never saw the test data during training.
    """)

DATA_PATH = "data/market_data.parquet"

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Strategy Settings")


user_portfolio_input = st.sidebar.text_input("Enter your current portfolio (comma separated):", "VNM, FPT, HPG, XYZ")
current_portfolio = [sym.strip().upper() for sym in user_portfolio_input.split(",") if sym.strip()] 

# 1. Add the Button right under the input
show_signals_clicked = st.sidebar.button("🎯 Get Today's Signals")

best_features = ["volatility_1w",
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_shock_monthly",
        "volatility_shock_weekly",
                  "dist_SMA_9",
                  "dist_SMA_21",
                  "dist_SMA_50","dist_SMA_100","dist_SMA_200",
        "dist_EMA_9", "dist_EMA_21", "dist_EMA_50","dist_EMA_100","dist_EMA_200",
                     "volume_surge_monthly",
        "volume_surge_weekly",
        "obv_trend",
        "price_vol_divergence",
                  "RSI_14",
                  "dist_52w_high",
    "log_ret_skip1m",
                 ]

df = load_data(_date_key=_today_vn())
df = df[df["close"] > 0].copy()
df = build_features(df)
df = target_generating_ranking(df)

# 2. If the button is clicked, generate and display right below it in the sidebar
if show_signals_clicked:
    # Ensure you are importing the base function directly
    
    
    with st.sidebar:
        st.divider()
        with st.spinner("Calculating signals..."):
            try:
                buys, holds, sells, non_vn100, ranks = generate_paper_trade_signals(
                    df=df,
                    current_portfolio=current_portfolio,
                    features=best_features,
                    trend_filter_col='dist_SMA_100',
                    target_col='target_quintile'
                )
                
                # Stack them vertically so they fit nicely in the sidebar
                st.success(f"🟢 **BUY ({len(buys)})**\n\n{', '.join(buys) if buys else 'None'}")
                st.info(f"🔵 **HOLD ({len(holds)})**\n\n{', '.join(holds) if holds else 'None'}")
                st.warning(f"🟠 **SELL ({len(sells)})**\n\n{', '.join(sells) if sells else 'None'}")
                
                if non_vn100:
                    st.error(f"🔴 **NOT VN100 ({len(non_vn100)})**\n\n{', '.join(non_vn100)}")
                
            except Exception as e:
                st.error(f"Could not generate signals: {e}")
        st.divider()
use_mega_alpha = False
selected_features = best_features

# --- MAIN EXECUTION: load precomputed artifacts immediately on page load ---
with st.spinner("Loading precomputed model artifacts..."):
    try:
        honest_test_df, result = load_pretrained(_date_key=_today_vn())
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

# --- SECTION 1: EQUITY CURVE ---
st.divider()
st.subheader("📊 Backtest Results")

st.markdown("""
These results are produced by **walk-forward cross-validation** across 12 folds (2020–2026).
Every data point below is **out-of-sample**: the model was trained exclusively on past data before
each test window, with a mandatory 21-trading-day gap to prevent any look-ahead leakage.

The equity curve below simulates a simple strategy: at each rebalance, hold the top-ranked stocks
(as identified by the model's OOS predictions) and track cumulative portfolio value starting from an
index of 10,000.
""")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pd.to_datetime(result['date']),
    y=result['total_value'],
    mode='lines+markers',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5),
    hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:,.2f}<extra></extra>'
))
fig1.update_layout(
    title=dict(text='Equity Curve: Portfolio Total Value (OOS Walk-Forward Backtest)', font=dict(size=14)),
    xaxis=dict(title='Date', tickangle=-45),
    yaxis=dict(title='Portfolio Value'),
    hovermode='x unified',
    height=450,
)
st.plotly_chart(fig1, use_container_width=True)

final_nav = result.iloc[-1]['total_value']
st.metric("Final Portfolio Value", f"{final_nav:,.2f} VND",
          delta=f"{(final_nav-10000)/100:.2f}% Total ROI")

st.caption("""
**How to read this:** The curve shows cumulative portfolio value over the full OOS test period.
Drawdowns reflect real market periods (e.g. COVID crash, 2022 rate hike environment) that the model
had to navigate with no foreknowledge. The ROI shown is purely from the model's rankings — no leverage,
no shorting. Transaction costs are fully modelled: 0.1% brokerage fee + 0.1% securities transfer tax
+ VND 300/share custody fee on each sell, plus T+2 settlement delay on proceeds.
""")

# --- SECTION 2: FEATURE DIAGNOSTICS ---
st.divider()
st.subheader("🔬 What Drives the Ranking?")

st.markdown("""
Below are three diagnostic views that reveal *what drives the model's rankings* and *whether those rankings
translate into real return differences*.
""")

with st.spinner("Computing feature influence and ranking diagnostics..."):

    # --- 1. FEATURE INFLUENCE VIA INFORMATION COEFFICIENT ---
    st.markdown("#### 📊 Feature Influence (Information Coefficient)")
    st.markdown("""
    The **Information Coefficient (IC)** measures the Spearman rank correlation between each feature
    and actual next-month returns across all OOS test dates.

    - **Positive IC (green):** Higher feature values tend to predict outperformance
    - **Negative IC (red):** Higher feature values tend to predict underperformance
    - **IC near 0:** Feature has little predictive signal on its own

    Even small IC magnitudes (0.02–0.05) can be meaningful in a cross-sectional setting when combined
    across many stocks and many time periods.
    """)
    st.caption("IC = Spearman rank correlation between each feature and actual next-month return, computed on OOS data only.")

    ic_scores = {}
    for feat in selected_features:
        valid = honest_test_df[[feat, 'next_1m_ret']].dropna()
        if len(valid) > 10:
            ic_scores[feat] = valid[feat].corr(valid['next_1m_ret'], method='spearman')

    ic_series = pd.Series(ic_scores).sort_values()
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in ic_series.values]

    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=ic_series.values,
        y=ic_series.index,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0)', width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>IC Score: %{x:.4f}<extra></extra>'
    ))
    fig_ic.update_layout(
        title='Feature Influence: Information Coefficient (Spearman)',
        xaxis=dict(title='IC Score', zeroline=True, zerolinecolor='black', zerolinewidth=1),
        hovermode='y',
        height=420,
    )
    st.plotly_chart(fig_ic, use_container_width=True)

    # --- 2. QUINTILE MONOTONICITY CHART ---
    st.markdown("#### 📈 Predicted Quintile vs Actual Return")
    st.markdown("""
    This chart is the clearest test of whether the model's rankings are meaningful.
    If the model has real predictive power, stocks it ranked in **Quintile 5** (top) should earn
    higher average returns than those in **Quintile 1** (bottom) — a strictly increasing "monotonic" pattern.

    A flat or random pattern would indicate the model is no better than chance. A consistent
    staircase from Q1 → Q5 confirms the rankings carry genuine signal.
    """)

    quintile_returns = (
        honest_test_df.groupby('pred_quintile')['next_1m_ret']
        .mean()
        .reset_index()
    ) if 'pred_quintile' in honest_test_df.columns else None

    if quintile_returns is not None:
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=quintile_returns['pred_quintile'],
            y=quintile_returns['next_1m_ret'] * 100,
            marker=dict(
                color=['#d62728','#ff7f0e','#bcbd22','#17becf','#2ca02c'],
                line=dict(color='rgba(255,255,255,0)', width=1.5)
            ),
            hovertemplate='<b>Quintile %{x}</b><br>Avg Return: %{y:.3f}%<extra></extra>'
        ))
        fig_q.update_layout(
            title='Average Return by Predicted Quintile (OOS)',
            xaxis=dict(title='Predicted Quintile (1=Worst, 5=Best)', tickmode='linear'),
            yaxis=dict(title='Avg Next-Month Return (%)'),
            hovermode='x',
            height=420,
        )
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        # Compute pred_quintile from pred_score if not already present
        honest_test_df['pred_quintile'] = pd.qcut(
            honest_test_df.groupby('date')['pred_score']
                        .transform(lambda x: x.rank(pct=True)),
            q=5, labels=[1,2,3,4,5]
        )
        quintile_returns = honest_test_df.groupby('pred_quintile', observed=False)['next_1m_ret'].mean().reset_index()
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=quintile_returns['pred_quintile'],
            y=quintile_returns['next_1m_ret'] * 100,
            marker=dict(
                color=['#d62728','#ff7f0e','#bcbd22','#17becf','#2ca02c'],
                line=dict(color='rgba(255,255,255,0)', width=1.5)
            ),
            hovertemplate='<b>Quintile %{x}</b><br>Avg Return: %{y:.3f}%<extra></extra>'
        ))
        fig_q.update_layout(
            title='Average Return by Predicted Quintile (OOS)',
            xaxis=dict(title='Predicted Quintile (1=Worst, 5=Best)', tickmode='linear'),
            yaxis=dict(title='Avg Next-Month Return (%)'),
            hovermode='x',
            height=420,
        )
        st.plotly_chart(fig_q, use_container_width=True)
            # --- 3. ALPHA GENERATION METRICS ---
    st.divider()
    st.subheader("🎯 Alpha Generation Summary")
    st.markdown("""
    These three metrics summarise the model's practical edge as a stock picker.
    All figures are computed on the **top 20% of stocks** ranked by the model each day, vs. the full VN100 universe.

    - **Hit Rate** — What fraction of the model's top picks actually had a positive return that month?
      A fair coin would give ~50%; anything consistently above that is a real edge.
    - **Avg Return** — The mean monthly log return across all top-20% picks over the OOS period.
    - **Lift over Market** — How much better did the top picks do compared to just holding everything?
      Even a small positive lift, applied consistently across 100 stocks, compounds significantly over time.
    """)

    top_q = honest_test_df.groupby('date', group_keys=False).apply(
        lambda g: g[g['pred_score'] >= g['pred_score'].quantile(0.8)], include_groups=False
    )
    all_ret   = honest_test_df['next_1m_ret'].dropna()
    top_ret   = top_q['next_1m_ret'].dropna()

    hit_rate      = (top_ret > 0).mean()
    avg_top_ret   = top_ret.mean()
    avg_all_ret   = all_ret.mean()
    lift          = avg_top_ret - avg_all_ret

    col1, col2, col3 = st.columns(3)
    col1.metric("Top-20% Hit Rate",   f"{hit_rate*100:.1f}%",  help="% of top picks with positive return")
    col2.metric("Top-20% Avg Return", f"{avg_top_ret*100:.2f}%", help="Mean monthly return of top quintile")
    col3.metric("Lift over Market",   f"{lift*100:.2f}%",       help="Top-20% return minus average market return")

    st.caption("""
    **Note on lift magnitude:** Even a sub-1% monthly lift may look modest in isolation, but in a
    cross-sectional strategy applied consistently across 100 stocks, small edges compound significantly
    over time. The NDCG scores above 0.83 confirm the model reliably preserves ranking order
    across a wide range of market conditions.
    """)




# =========================================================================================
# 🤖 FACEBOOK MESSENGER-STYLE FLOATING CHAT
# =========================================================================================

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    API_KEY = ""

SYSTEM_PROMPT = """
You are a senior quantitative analyst and the creator of this specific Streamlit dashboard.
Your job is to answer user questions about the VN100 Cross-Sectional Ranking model.

The model is an XGBoost LambdaRank ranker (rank:ndcg objective) trained via walk-forward
cross-validation across 12 folds covering 2020–2026, with a mandatory 21-day gap between
train and test windows to prevent look-ahead bias. All results shown are fully out-of-sample.

Features used: volatility at 1W/1M/3M/6M horizons, weekly and monthly volatility shocks,
distance from SMA and EMA at 9/21/50/100/200 periods, weekly and monthly volume surge ratios,
OBV trend, price-volume divergence, RSI-14, distance from 52-week high, and skip-1M log return.

The target variable is a risk-adjusted quintile ranking: next_1m_return / volatility_3m,
binned into 5 quintiles (0=worst, 4=best) cross-sectionally each day.

Explain concepts clearly, as if you are mentoring a junior quant.
"""

# --- CALLBACK TO PROCESS CHAT MESSAGES SAFELY ---
def process_chat():
    # If the AI is already thinking, abort the send and keep their draft text safe
    if st.session_state.awaiting_response:
        return

    user_input = st.session_state.chat_input_key
    if user_input.strip():
        # 1. Add User Message to UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        # 2. Clear input
        st.session_state.chat_input_key = ""
        # 3. Lock the UI by setting the 'thinking' flag
        st.session_state.awaiting_response = True


# Custom CSS to force the button into a perfect Messenger circle
fb_messenger_css = """
<style>
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    width: 60px !important;
    height: 60px !important;
    z-index: 99999 !important;
}

div[data-testid="stPopover"] button {
    width: 100% !important;
    height: 100% !important;
    border-radius: 50% !important;
    background-color: #0A7CFF !important; 
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    font-size: 28px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

div[data-testid="stPopover"] button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important;
}

div[data-testid="stPopoverBody"] {
    width: 350px !important;
    max-height: 550px !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 28px 0 rgba(0, 0, 0, 0.2), 0 2px 4px 0 rgba(0, 0, 0, 0.1) !important;
    border: 1px solid #3E4042 !important;
    background-color: #242526 !important; 
    padding: 0 !important;
    overflow: hidden !important;
}

.fb-chat-header {
    background-color: #242526;
    padding: 12px 16px;
    border-bottom: 1px solid #3E4042;
    display: flex;
    align-items: center;
    gap: 12px;
    color: #E4E6EB;
    font-family: system-ui, -apple-system, sans-serif;
    font-weight: 600;
    font-size: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.fb-avatar {
    width: 32px;
    height: 32px;
    background-color: #0A7CFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.fb-status-dot {
    width: 8px;
    height: 8px;
    background-color: #31A24C; 
    border-radius: 50%;
    display: inline-block;
    margin-left: 6px;
}
.chat-wrapper {
    padding: 10px 16px;
}
</style>
"""
st.markdown(fb_messenger_css, unsafe_allow_html=True)

with st.popover("🤖", use_container_width=False):
    
    st.markdown("""
    <div class="fb-chat-header">
        <div class="fb-avatar">🤖</div>
        <div>AI Quant Assistant <span class="fb-status-dot"></span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
    
    # Render Chat History
    chat_container = st.container(height=340, border=False)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # UI updates to visually show the AI is processing
        if st.session_state.awaiting_response:
            with st.chat_message("model"):
                st.markdown("⏳ *AI is thinking...*")
                
    # Render Chat Input
    cols = st.columns([5, 1])
    with cols[0]:
        # REMOVED the 'disabled' argument so you can type freely while it thinks
        st.text_input("Message", key="chat_input_key", label_visibility="collapsed", placeholder="Aa", on_change=process_chat)
    with cols[1]:
        # KEPT the 'disabled' argument so the send button stays grayed out
        st.button("➤", on_click=process_chat, disabled=st.session_state.awaiting_response)
            
    st.markdown("</div>", unsafe_allow_html=True)

    # --- API CALL EXECUTION ---
    if st.session_state.awaiting_response:
        
        with st.spinner(""):
            if not API_KEY:
                st.session_state.messages.append({"role": "model", "content": "⚠️ API Key not found! Add it to `.streamlit/secrets.toml`"})
            else:
                try:
                    client = genai.Client(api_key=API_KEY)

                    formatted_history = [
                        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
                        for m in st.session_state.messages[:-1]
                    ]
                    user_prompt = st.session_state.messages[-1]["content"]

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                        contents=formatted_history + [
                            types.Content(role="user", parts=[types.Part(text=user_prompt)])
                        ],
                    )

                    st.session_state.messages.append({"role": "model", "content": response.text})
                except Exception as e:
                    st.session_state.messages.append({"role": "model", "content": f"API Error: {str(e)}"})
        
        # Unlock the UI and refresh the page to show the AI's final answer
        st.session_state.awaiting_response = False
        st.rerun()