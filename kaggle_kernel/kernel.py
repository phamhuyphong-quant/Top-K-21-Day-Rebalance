import os, sys

# ── 1. Clone main branch (source code) ───────────────────────────────────────
GH_PAT = os.environ['GH_PAT']
os.system(f"git clone --depth 1 --branch main https://{GH_PAT}@github.com/Masterokadanori/Cross_Sectional_Rank_VN100.git /kaggle/working/repo")
sys.path.append('/kaggle/working/repo')

# ── 2. Pull market_data.parquet from data-storage branch ─────────────────────
os.system(f"git clone --depth 1 --branch data-storage https://{GH_PAT}@github.com/Masterokadanori/Cross_Sectional_Rank_VN100.git /kaggle/working/data-storage")
os.makedirs('/kaggle/working/repo/data', exist_ok=True)
os.system("cp /kaggle/working/data-storage/market_data.parquet /kaggle/working/repo/data/market_data.parquet")

# ── 3. Install dependencies ───────────────────────────────────────────────────
os.system("pip install -q -r /kaggle/working/repo/requirements.txt")

# ── 4. Run precompute ─────────────────────────────────────────────────────────
os.chdir('/kaggle/working/repo')
sys.path.insert(0, '/kaggle/working/repo')

from src.features import build_features, target_generating_ranking
from src.evaluation import generate_and_save_pretrained_model
import pandas as pd

df = pd.read_parquet('data/market_data.parquet')
df = df[df['close'] > 0].copy()
df = build_features(df)
df = target_generating_ranking(df)

best_features = [
    'volatility_1w', 'volatility_1m', 'volatility_3m', 'volatility_6m',
    'dist_SMA_100', 'dist_SMA_14',
    'log_ret_1w', 'log_ret_1m', 'log_ret_6m', 'log_ret_1y',
    'RSI_14', 'volume_surge_monthly'
]

generate_and_save_pretrained_model(
    df=df,
    selected_features=best_features,
    use_mega_alpha=False,
    output_dir='/kaggle/working/repo/data/pretrained'
)

# ── 5. Copy outputs to /kaggle/working so GitHub Actions can download them ────
os.system("cp /kaggle/working/repo/data/pretrained/pretrained_predictions.parquet /kaggle/working/")
os.system("cp /kaggle/working/repo/data/pretrained/pretrained_equity_curve.parquet /kaggle/working/")
print("✅ Done!")