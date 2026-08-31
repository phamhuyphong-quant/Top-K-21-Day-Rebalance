import os, sys, random, numpy as np

random.seed(42)
np.random.seed(42)
# ── 1. Clone main branch (source code) ───────────────────────────────────────
GH_PAT = os.environ.get('GH_PAT') or os.environ.get('GITHUB_TOKEN', '')
os.system(f"git clone --depth 1 --branch main https://{GH_PAT}@github.com/phamhuyphong-quant/cross_sectional_rank_vn.git /kaggle/working/repo")
sys.path.append('/kaggle/working/repo')

# ── 2. Pull market_data.parquet from data-storage branch ─────────────────────
os.system(f"git clone --depth 1 --branch data-storage https://{GH_PAT}@github.com/phamhuyphong-quant/cross_sectional_rank_vn.git /kaggle/working/data-storage")
os.makedirs('/kaggle/working/repo/data', exist_ok=True)
os.system("cp /kaggle/working/data-storage/market_data.parquet /kaggle/working/repo/data/market_data.parquet")

# ── 3. Install dependencies ───────────────────────────────────────────────────
os.system("pip install -q -r /kaggle/working/repo/requirements-dev.txt")

# ── 4. Build features/targets (required input for signal generation) ────────
os.chdir('/kaggle/working/repo')
sys.path.insert(0, '/kaggle/working/repo')

from src.features import build_features, target_generating_ranking, build_targets
from src.inference import generate_paper_trade_signals
import pandas as pd
from config import candidate_features,usedSymbols

df_raw = pd.read_parquet('data/market_data.parquet')
df_raw = df_raw[df_raw['Symbol'].isin(usedSymbols)]
df_raw['date'] = pd.to_datetime(df_raw['date'])

excluded_dates = pd.to_datetime(['2018-01-23', '2018-01-24'])

df_raw = df_raw[~df_raw['date'].isin(excluded_dates)].copy()


df = build_features(df=df_raw,adtv_limit=2_500_000)
df_raw = build_features(df=df_raw,adtv_limit=2_500_000,generate_target=False)
best_features = candidate_features

# ── 5. Evaluate P1's combined entry filter for the latest date ───────────────
# P1 (per the research paper) only goes to cash when BOTH conditions are true:
#   P2 (dispersion): regime_bucket_monthly == "Q1" (bottom-25% dispersion regime)
#   P3 (trend):      market1m_ema21 - market3m_ema63 < 0
#
# NOTE: latest_date is taken from df_raw, not df. build_features(generate_target=True)
# calls build_targets(), which needs shift(-21)/shift(-5) forward returns and therefore
# dropna()'s the last ~21 trading days per symbol out of `df`. `df_raw` (generate_target=False)
# keeps those rows, so it's the one that actually has today's date. regime_bucket_monthly /
# market1m_ema21 / market3m_ema63 are computed unconditionally in build_features (outside the
# generate_target branch), so they exist identically in df_raw too.
latest_date = df_raw['date'].max()
latest_rows = df_raw[df_raw['date'] == latest_date].copy()
latest = latest_rows.iloc[0]

p2_active = latest['regime_bucket_monthly'] == 'Q1'
p3_active = (latest['market1m_ema21'] - latest['market3m_ema63']) < 0
p1_filter_active = bool(p2_active and p3_active)

signal_date_str = str(latest_date.date())

if p1_filter_active:
    print(f"🛑 P1 combined filter ACTIVE on {signal_date_str} — holding cash, no signals generated.")
    today_signals = pd.DataFrame([{
        'signal_date': signal_date_str,
        'filter_active': True,
        'rank': None,
        'Symbol': None,
        'live_score': None,
    }])
else:
    print(f"✅ P1 combined filter inactive on {signal_date_str} — generating top-20 signals.")

    # generate_paper_trade_signals() requires a 'qid' column (used for XGBRanker's
    # group boundaries). build_features()/build_targets() never create one — only
    # target_generating_ranking() does, and kernel.py never calls that — so without
    # this line training would fail with KeyError: 'qid'.
    #
    # build_features() sorts df by ["Symbol", "date"], so same-date rows are NOT
    # contiguous — they're scattered across each symbol's block. XGBRanker requires
    # qid to be sorted in non-decreasing order (same-qid rows contiguous), so we
    # re-sort by date first, THEN assign qid, so groupby('date').ngroup() lines up
    # with row order.
    df = df.sort_values('date').reset_index(drop=True)
    df['qid'] = df.groupby('date').ngroup()

    # Train on df (has target_magnitude, but its tail is trimmed off), predict on
    # df_raw's true latest date. generate_paper_trade_signals() internally does:
    #   latest_date = combined['date'].max()
    #   train_df    = combined[(date < latest_date) & (target_magnitude.notna())]
    #   inference_df = combined[date == latest_date]
    # so appending just today's row(s) from df_raw (target left NaN) makes it train
    # on df's full history and predict on today, with no change needed in inference.py.
    latest_rows['target_magnitude'] = np.nan
    train_predict_df = pd.concat([df, latest_rows], ignore_index=True, sort=False)

    _, _, _, _, ranked_today = generate_paper_trade_signals(
        df=train_predict_df,
        current_portfolio=[],       # portfolio-agnostic; Streamlit filters at runtime
        features=best_features,
        model=None,                 # retrains on full history inside the function
        buy_n=20,                   # K=20, matching the paper's P1 Top-K
        trend_filter_col=None,      # P1's only filter is the market-level one above
        target_col='target_magnitude',
        icir_filter=False,  # matches Saving_DataFrame.ipynb, which produced the paper's df_predict_ndcg
        corr_prune=True,
    )
    today_signals = ranked_today.sort_values('rank').head(20).copy()
    today_signals['signal_date'] = signal_date_str
    today_signals['filter_active'] = False

# ── 6. Save + upload today's P1 signals to Hugging Face ──────────────────────
os.makedirs('/kaggle/working/repo/data/pretrained', exist_ok=True)
today_signals.to_parquet('/kaggle/working/repo/data/pretrained/today_signals.parquet', index=False)

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get('HF_TOKEN', '')
api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj='/kaggle/working/repo/data/pretrained/today_signals.parquet',
    path_in_repo='today_signals.parquet',
    repo_id="PhongHPham/vn_cross_sectional_ranking_data_storage",
    repo_type="dataset",
)
print("✅ Uploaded today_signals.parquet to Hugging Face")