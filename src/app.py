import streamlit as st
import pandas as pd
import requests
import io
import datetime

st.set_page_config(page_title="Scientific Research", layout="centered")


def _today_vn() -> str:
    """Returns today's date in Vietnam time (UTC+7) as a string key like '2025-05-01'.
    Used as a cache-buster so data is always fresh after midnight VN time."""
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=7)).strftime("%Y-%m-%d")


@st.cache_data(ttl=3600)
def load_today_signals(_date_key: str = None):
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    base_url = "https://huggingface.co/datasets/PhongHPham/vn_cross_sectional_ranking_data_storage/resolve/main/"
    try:
        response = requests.get(base_url + "today_signals.parquet", headers=headers)
        if response.status_code != 200:
            raise Exception(f"HF Error: {response.status_code}")
        return pd.read_parquet(io.BytesIO(response.content))
    except Exception as e:
        st.warning(f"⚠️ Could not load today's signals: {e}")
        return None


st.title("APPLICATION OF A MACHINE LEARNING RANKING MODEL COMBINED WITH AN ENTRY-POINT FILTER ON THE VIETNAMESE STOCK MARKET")

st.caption("The signal forecasts on this website are for research and academic purposes only. They should not be regarded as investment advice. Users are responsible for their own investment decisions.")

with st.spinner("Loading today's P1 signal..."):
    today_signals = load_today_signals(_date_key=_today_vn())

if today_signals is None or today_signals.empty:
    st.info("No signal available yet.")
else:
    signal_date = today_signals["signal_date"].iloc[0] if "signal_date" in today_signals.columns else None
    if signal_date:
        st.caption(f"📅 Signal generated using data up to {signal_date}")

    filter_active = bool(today_signals["filter_active"].iloc[0]) if "filter_active" in today_signals.columns else False

    if filter_active:
        st.warning("🛑 **Hold all cash — do not buy today**\n\nThe P1 filter is active.")
    else:
        column_labels = {
            "Symbol": "Stock ticker",
            "rank": "Rank",
        }
        display_cols = [c for c in column_labels if c in today_signals.columns]
        display_df = (
            today_signals.sort_values("rank")[display_cols]
            .rename(columns=column_labels)
            .reset_index(drop=True)
        )
        display_df["Capital allocation"] = "5%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)