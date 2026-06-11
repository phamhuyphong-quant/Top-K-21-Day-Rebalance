import os, sys, random, numpy as np

random.seed(42)
np.random.seed(42)
# ── 1. Clone main branch (source code) ───────────────────────────────────────
GH_PAT = os.environ.get('GH_PAT') or os.environ.get('GITHUB_TOKEN', '')
os.system(f"git clone --depth 1 --branch main https://{GH_PAT}@github.com/phamhuyphong-quant/Cross_Sectional_Rank_VN.git /kaggle/working/repo")
sys.path.append('/kaggle/working/repo')

# ── 2. Pull market_data.parquet from data-storage branch ─────────────────────
os.system(f"git clone --depth 1 --branch data-storage https://{GH_PAT}@github.com/phamhuyphong-quant/Cross_Sectional_Rank_VN.git /kaggle/working/data-storage")
os.makedirs('/kaggle/working/repo/data', exist_ok=True)
os.system("cp /kaggle/working/data-storage/market_data.parquet /kaggle/working/repo/data/market_data.parquet")

# ── 3. Install dependencies ───────────────────────────────────────────────────
os.system("pip install -q -r /kaggle/working/repo/requirements-dev.txt")
# ── 4. Run precompute ─────────────────────────────────────────────────────────
os.chdir('/kaggle/working/repo')
sys.path.insert(0, '/kaggle/working/repo')

from src.features import build_features, target_generating_ranking,build_targets
from src.evaluation import pretrain_and_save_artifacts
import pandas as pd
from config import candidate_features
df_raw = pd.read_parquet('data/market_data.parquet')
df_raw = df_raw[df_raw["close"] > 0].copy()
df = build_features(df_raw)
df = build_targets(df)
df = target_generating_ranking(df)

best_features = candidate_features


pretrain_and_save_artifacts(
    df=df,
    selected_features=best_features,
    use_mega_alpha=False,
    output_dir='/kaggle/working/repo/data/pretrained'
)

# ── 5. Upload outputs to Hugging Face ────────────────────────────────────────
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get('HF_TOKEN', '')
api = HfApi(token=HF_TOKEN)

for fname in ["pretrained_predictions.parquet", "pretrained_equity_curve.parquet"]:
    api.upload_file(
        path_or_fileobj=f"/kaggle/working/repo/data/pretrained/{fname}",
        path_in_repo=fname,
        repo_id="PhongHPham/vn_cross_sectional_ranking_data_storage",
        repo_type="dataset",
    )
    print(f"✅ Uploaded {fname} to Hugging Face")

# ── 6. Generate today's signals and upload ───────────────────────────────────
from src.inference import generate_paper_trade_signals

_, _, _, _, ranked_today = generate_paper_trade_signals(
    df=df,
    current_portfolio=[],       # portfolio-agnostic; Streamlit filters at runtime
    features=best_features,
    model=None,                 # retrains on full history inside the function
    buy_n=30,
    trend_filter_col='dist_SMA_100',
    target_col='target_quintile',
    icir_filter=True,
    corr_prune = True,
)

ranked_today = ranked_today.assign(signal_date=str(df['date'].max().date()))
ranked_today.to_parquet('/kaggle/working/repo/data/pretrained/today_signals.parquet', index=False)

api.upload_file(
    path_or_fileobj='/kaggle/working/repo/data/pretrained/today_signals.parquet',
    path_in_repo='today_signals.parquet',
    repo_id="PhongHPham/vn_cross_sectional_ranking_data_storage",
    repo_type="dataset",
)
print("✅ Uploaded today_signals.parquet to Hugging Face")