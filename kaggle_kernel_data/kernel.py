import os, sys

# ── 1. Clone private repo ─────────────────────────────────────────────────────
GH_PAT = os.environ['GH_PAT']
os.system(f"git clone --depth 1 --branch main https://{GH_PAT}@github.com/Masterokadanori/Cross_Sectional_Rank_VN100.git /kaggle/working/repo")
sys.path.insert(0, '/kaggle/working/repo')

# ── 2. Install dependencies ───────────────────────────────────────────────────
os.system("pip install -q -U vnstock vnai pandas pyarrow")

# ── 3. Run data collection ────────────────────────────────────────────────────
os.chdir('/kaggle/working/repo')
os.makedirs('data', exist_ok=True)
os.system("python src/data_collect.py")

# ── 4. Copy output to /kaggle/working so GitHub Actions can download it ───────
os.system("cp /kaggle/working/repo/data/market_data.parquet /kaggle/working/")
print("✅ Data collection done!")