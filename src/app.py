import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import google.generativeai as genai

# Ensure local imports work
from features import build_features, target_generating_ranking
from evaluation import run_xgboost_backtest
from models import walk_forward_cv, test_train_spliter # Import thêm nếu cần

st.set_page_config(page_title="VN100 Backtest Dashboard", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None

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

mode = st.sidebar.selectbox(
    "Choose Backtest Mode:",
    ["Use Pretrained Model", "Train via Walk-Forward CV"]
)

best_features = [
    'log_ret_1m', 'log_ret_3m', 'log_ret_1y',      # Momentum
    'volatility_shock_monthly', 'volatility_3m',   # Risk
    'dist_SMA_100',                                # Fast Trend
    'RSI_14','volume_surge_monthly'
]

if mode == "Use Pretrained Model":
    selected_features = best_features
    st.sidebar.info("Using the optimized 8 features to reproduce the pretrained model's Walk-Forward results.")
else:
    all_available_features = [
    'log_ret_1m', 'log_ret_3m', 'log_ret_6m',  'log_ret_1y',
    'volatility_shock_monthly', 'volatility_3m', 'volatility_6m', 'volatility_1m' ,   'volatility_1w' , 
    'dist_SMA_100', 'dist_SMA_14','dist_SMA_50'    ,                     
    'RSI_14','volume_surge_monthly', 'vol_3m_avg'
]
    selected_features = st.sidebar.multiselect(
        "Select Features:", 
        all_available_features, 
        default=best_features
    )
    
    # THÊM NÚT KÍCH HOẠT MEGA-ALPHA
    use_mega_alpha = st.sidebar.checkbox("🔥 Kích hoạt Mega-Alpha (LSTM-Attention)", value=False)

if st.sidebar.button("🚀 Run Backtest"):
    st.session_state.backtest_run = True

# --- MAIN EXECUTION ---
if st.session_state.backtest_run:
    df_raw = load_data(DATA_PATH)
    
    if df_raw is None:
        st.error(f"Dataset not found at {DATA_PATH}")
    else:
        with st.spinner("Processing features..."):
            df = df_raw[df_raw["close"] > 0].copy()
            df = build_features(df)
            df = target_generating_ranking(df)

            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                st.error(f"⚠️ WARNING: These features are missing from your dataset: {missing_features}. Check your features.py file!")
                selected_features = [f for f in selected_features if f in df.columns]

        # --- WALK-FORWARD CV ---
        st.subheader(f"Walk-Forward CV Progress ({mode})")
        progress_bar = st.progress(0)
        status_log = st.empty()
        all_messages = []

        def streamlit_callback(fold_num, total_folds, msg):
            all_messages.append(msg)
            status_log.code("\n".join(all_messages))
            progress_bar.progress(fold_num / total_folds)

        honest_test_df = walk_forward_cv(
            df, selected_features, 
            initial_train_months=12, test_months=6, gap_days=21,
            callback=streamlit_callback,
            use_mega=use_mega_alpha
        )
        
        with st.spinner("Backtesting OOS results..."):
            result = run_xgboost_backtest(
                honest_test_df, model=None, features=selected_features,
                time_of_rebalance='M', trailing_stop=-0.10
            )

        # --- VISUALIZATION: EQUITY CURVE ---
        st.divider()
        st.subheader(f"Results: {mode}")
        
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

        # --- VISUALIZATION: FEATURE IMPORTANCE ---
        st.divider()
        st.subheader("What drives the ranking?")
        
        with st.spinner("Training final global model for feature importance..."):
            final_model = xgb.XGBRanker(
                tree_method='hist', objective='rank:ndcg', 
                n_estimators=100, learning_rate=0.1, max_depth=4,
                colsample_bytree=0.7, subsample=0.8, random_state=42
            )
            
            X_all = df[selected_features]
            y_all = df['target_quintile']
            qids_all = df['qid']
            
            final_model.fit(X_all, y_all, qid=qids_all, verbose=False)
            
            importances = pd.Series(final_model.feature_importances_, index=selected_features).sort_values()
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            importances.plot(kind='barh', ax=ax2, color='#2ca02c') 
            ax2.set_title('Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.6, axis='x')
            st.pyplot(fig2)

else:
    st.info("Select your strategy in the sidebar and click 'Run Backtest'.")


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