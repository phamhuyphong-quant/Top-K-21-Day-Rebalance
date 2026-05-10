import os, sys


# ── 1. Clone main branch (source code) ───────────────────────────────────────
GH_PAT = os.environ.get('GH_PAT') or os.environ.get('GITHUB_TOKEN', '')
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
from src.alpha_mining import WorldQuantAlphas
import pandas as pd

df_raw = pd.read_parquet('data/market_data.parquet')
wq = WorldQuantAlphas(df_raw)
wq_cols_df = wq.generate_all()
df_raw[wq_cols_df.columns] = wq_cols_df.values
df = build_features(df_raw)
df = target_generating_ranking(df)

best_features = ['log_ret_1w',
                 'log_ret_1m', 
                 'log_ret_3m', 'log_ret_6m', 
                 'log_ret_1y',
                 
                 'volatility_1w',
                 'volatility_1m',
                 'volatility_3m',
                 'volatility_6m',
                 'volatility_shock_monthly',
                 'volatility_shock_weekly',
                 'volume_surge_monthly',
                 'volume_surge_weekly',
                 'obv_trend',
                 'price_vol_divergence',
                 
                 'WQ_Alpha_012',
                 'WQ_Alpha_024', 
                 'WQ_Alpha_028', 
                 'WQ_Alpha_053',
                 'WQ_Alpha_060',
                 
                 'dist_52w_high', 
                 'log_ret_skip1m']

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