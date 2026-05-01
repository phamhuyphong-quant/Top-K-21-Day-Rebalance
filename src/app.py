import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
import sys
import requests
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Standardized Absolute Imports
from src.features import build_features, target_generating_ranking


from src.inference import generate_paper_trade_signals 
st.set_page_config(page_title="VN100 Backtest Dashboard", layout="wide")


@st.cache_data(ttl="1d")
def load_data():
    
    
    
    # This URL points specifically to your storage branch
    url = "https://raw.githubusercontent.com/Masterokadanori/Cross_Sectional_Rank_VN100/data-storage/market_data.parquet"
    
    # Pass your secret token so GitHub knows you have permission
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Successfully fetched the parquet as bytes
            return pd.read_parquet(io.BytesIO(response.content))
        else:
            raise Exception(f"GitHub Error {response.status_code}: {response.text}")
    except Exception as e:
        # Fallback to local data if the internet or token fails
        st.warning(f"⚠️ Live fetch failed. Using local seed data. Error: {e}")
        return pd.read_parquet("data/market_data.parquet")
    

@st.cache_data(ttl="1d")
def load_pretrained():
    """
    Fetches the precomputed walk-forward predictions and equity curve from the data-storage branch.
    """
    base_url = "https://raw.githubusercontent.com/Masterokadanori/Cross_Sectional_Rank_VN100/data-storage/"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    
    try:
        # Fetch Predictions
        pred_response = requests.get(base_url + "pretrained_predictions.parquet", headers=headers)
        if pred_response.status_code == 200:
            honest_test_df = pd.read_parquet(io.BytesIO(pred_response.content))
        else:
            raise Exception(f"GitHub Error (Predictions): {pred_response.status_code}")
            
        # Fetch Equity Curve
        eq_response = requests.get(base_url + "pretrained_equity_curve.parquet", headers=headers)
        if eq_response.status_code == 200:
            result = pd.read_parquet(io.BytesIO(eq_response.content))
        else:
            raise Exception(f"GitHub Error (Equity Curve): {eq_response.status_code}")
            
        return honest_test_df, result
        
    except Exception as e:
        st.warning(f"⚠️ Live fetch of precomputed models failed. Using local artifacts. Error: {e}")
        # Fallback to local files if API/Internet fails
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
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = ""
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

st.title("📈 VN100 Cross-Sectional Ranking Dashboard")

DATA_PATH = "data/market_data.parquet"

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Strategy Settings")


user_portfolio_input = st.sidebar.text_input("Enter your current portfolio (comma separated):", "VNM, FPT, HPG, XYZ")
current_portfolio = [sym.strip().upper() for sym in user_portfolio_input.split(",") if sym.strip()] 

# 1. Add the Button right under the input
show_signals_clicked = st.sidebar.button("🎯 Get Today's Signals")

best_features = [#'log_ret_daily',
                  'volatility_1w', 'volatility_1m', 'volatility_3m', 'volatility_6m',
    #              'volatility_shock_monthly',
    #'volatility_shock_weekly' , 
                  'dist_SMA_100',
    'dist_SMA_14', #'dist_SMA_50',
                  'log_ret_1w','log_ret_1m',
    #'log_ret_3m',
    'log_ret_6m','log_ret_1y',
'RSI_14',
    'volume_surge_monthly'
]

df = load_data()
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


if st.sidebar.button("🚀 Run Backtest"):
    st.session_state.backtest_run = True

# --- MAIN EXECUTION ---
if st.session_state.backtest_run:
    with st.spinner("Loading precomputed model artifacts..."):
        try:
            honest_test_df, result = load_pretrained()
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")
            st.stop()

        # --- VISUALIZATION: EQUITY CURVE ---
        st.divider()
        st.subheader(f"Results:")
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(pd.to_datetime(result['date']), result['total_value'], 
                marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        ax1.set_title('Equity Curve: Portfolio Total Value', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        final_nav = result.iloc[-1]['total_value']
        st.metric("Final Portfolio Value", f"{final_nav:,.2f} VND", 
                  delta=f"{(final_nav-10000)/100:.2f}% Total ROI")

        # --- VISUALIZATION: FEATURE IMPORTANCE (IC-BASED) ---
        st.divider()
        st.subheader("What drives the ranking?")

        with st.spinner("Computing feature influence and ranking diagnostics..."):

            # --- 1. FEATURE INFLUENCE VIA INFORMATION COEFFICIENT ---
            st.markdown("#### 📊 Feature Influence (Information Coefficient)")
            st.caption("IC = rank correlation between each feature and actual next-month return. Higher = more predictive.")

            ic_scores = {}
            for feat in selected_features:
                valid = honest_test_df[[feat, 'next_1m_ret']].dropna()
                if len(valid) > 10:
                    ic_scores[feat] = valid[feat].corr(valid['next_1m_ret'], method='spearman')

            ic_series = pd.Series(ic_scores).sort_values()
            colors = ['#d62728' if v < 0 else '#2ca02c' for v in ic_series.values]

            fig_ic, ax_ic = plt.subplots(figsize=(10, 5))
            ic_series.plot(kind='barh', ax=ax_ic, color=colors)
            ax_ic.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax_ic.set_title('Feature Influence: Information Coefficient (Spearman)', fontsize=13, fontweight='bold')
            ax_ic.set_xlabel('IC Score')
            ax_ic.grid(True, linestyle='--', alpha=0.5, axis='x')
            st.pyplot(fig_ic)

            # --- 2. QUINTILE MONOTONICITY CHART ---
            st.markdown("#### 📈 Predicted Quintile vs Actual Return")
            st.caption("A good ranking model should show strictly increasing returns from Quintile 1 → 5.")

            quintile_returns = (
                honest_test_df.groupby('pred_quintile')['next_1m_ret']
                .mean()
                .reset_index()
            ) if 'pred_quintile' in honest_test_df.columns else None

            if quintile_returns is not None:
                fig_q, ax_q = plt.subplots(figsize=(8, 5))
                ax_q.bar(quintile_returns['pred_quintile'], quintile_returns['next_1m_ret'] * 100,
                        color=['#d62728','#ff7f0e','#bcbd22','#17becf','#2ca02c'])
                ax_q.set_title('Average Return by Predicted Quintile (OOS)', fontsize=13, fontweight='bold')
                ax_q.set_xlabel('Predicted Quintile (1=Worst, 5=Best)')
                ax_q.set_ylabel('Avg Next-Month Return (%)')
                ax_q.axhline(0, color='black', linewidth=0.8)
                ax_q.grid(True, linestyle='--', alpha=0.5, axis='y')
                st.pyplot(fig_q)
            else:
                # Compute pred_quintile from pred_score if not already present
                honest_test_df['pred_quintile'] = pd.qcut(
                    honest_test_df.groupby('date')['pred_score']
                                .transform(lambda x: x.rank(pct=True)),
                    q=5, labels=[1,2,3,4,5]
                )
                quintile_returns = honest_test_df.groupby('pred_quintile')['next_1m_ret'].mean().reset_index()
                fig_q, ax_q = plt.subplots(figsize=(8, 5))
                ax_q.bar(quintile_returns['pred_quintile'].astype(int), quintile_returns['next_1m_ret'] * 100,
                        color=['#d62728','#ff7f0e','#bcbd22','#17becf','#2ca02c'])
                ax_q.set_title('Average Return by Predicted Quintile (OOS)', fontsize=13, fontweight='bold')
                ax_q.set_xlabel('Predicted Quintile (1=Worst, 5=Best)')
                ax_q.set_ylabel('Avg Next-Month Return (%)')
                ax_q.axhline(0, color='black', linewidth=0.8)
                ax_q.grid(True, linestyle='--', alpha=0.5, axis='y')
                st.pyplot(fig_q)

            # --- 3. ALPHA GENERATION METRICS ---
            st.markdown("#### 🎯 Predictive Power (Alpha Generation)")

            top_q = honest_test_df.groupby('date', group_keys=False).apply(
                lambda g: g[g['pred_score'] >= g['pred_score'].quantile(0.8)]
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

else:
    st.info("Click 'Run Backtest' in the sidebar.")


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
You know that the model uses XGBoost Ranker, Walk-Forward Cross Validation, and features like 
RSI, Volatility, and Momentum. Explain concepts clearly, as if you are mentoring a junior quant.
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
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=SYSTEM_PROMPT)
                    
                    formatted_history = [{"role": m["role"], "parts": [m["content"]]} for m in st.session_state.messages[:-1]]
                    chat = model.start_chat(history=formatted_history)
                    
                    user_prompt = st.session_state.messages[-1]["content"]
                    response = chat.send_message(user_prompt)
                    
                    st.session_state.messages.append({"role": "model", "content": response.text})
                except Exception as e:
                    st.session_state.messages.append({"role": "model", "content": f"API Error: {str(e)}"})
        
        # Unlock the UI and refresh the page to show the AI's final answer
        st.session_state.awaiting_response = False
        st.rerun()