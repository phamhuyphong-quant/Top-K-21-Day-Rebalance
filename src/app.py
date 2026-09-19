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


st.title("ỨNG DỤNG MÔ HỌC MÁY XẾP HẠNG KẾT HỢP BỘ LỌC ĐIỂM VÀO LỆNH TRÊN THỊ TRƯỜNG CHỨNG KHOÁN VIỆT NAM")

st.caption("Các dự báo tín hiệu trên trang web này chỉ sử dụng cho mục đích nghiên cứu và học thuật. Không được xem là lời khuyên đầu tư. Người dùng chịu trách nhiệm về quyết định đầu tư của mình.")

with st.spinner("Loading today's P1 signal..."):
    today_signals = load_today_signals(_date_key=_today_vn())

if today_signals is None or today_signals.empty:
    st.info("No signal available yet.")
else:
    signal_date = today_signals["signal_date"].iloc[0] if "signal_date" in today_signals.columns else None
    if signal_date:
        st.caption(f"📅 Tín hiệu được tạo ra sử dụng dữ liệu đến ngày {signal_date}")

    filter_active = bool(today_signals["filter_active"].iloc[0]) if "filter_active" in today_signals.columns else False

    if filter_active:
        st.warning("🛑 **Giữ tất cả tiền mặt — không mua hôm nay**\n\nBộ lọc P1 đang được kích hoạt.")
    else:
        column_labels = {
            "Symbol": "Mã cổ phiếu",
            "rank": "Thứ hạng",
        }
        display_cols = [c for c in column_labels if c in today_signals.columns]
        display_df = (
            today_signals.sort_values("rank")[display_cols]
            .rename(columns=column_labels)
            .reset_index(drop=True)
        )
        display_df["Tỉ lệ vốn phân bổ"] = "5%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)