import os
import sys
import pandas as pd

# Add the project root to the Python path so we can cleanly import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import build_features, target_generating_ranking,build_targets
from src.evaluation import pretrain_and_save_artifacts
from config import candidate_features
def main():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    print("🚀 Starting precomputation pipeline...")
    
    # 1. Load the raw market data
    # Assuming you run this from the root directory of your project
    data_path = os.path.join("data", "market_data.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Could not find {data_path}. Ensure you run this from the project root.")
    vnindex_path = os.path.join("data","vnindex_data.parquet") 
    if not os.path.exists(vnindex_path):
        raise FileNotFoundError(f"❌ Could not find {vnindex_path}. Ensure you run this from the project root.")   
    print(f"📦 Loading data from {data_path}...")
    df_raw = pd.read_parquet(data_path)
    vnindex_df = pd.read_parquet(vnindex_path)
    df_raw = df_raw[df_raw["close"] > 0].copy()
    # 2. Process features and targets (matching app.py logic exactly)
    print("⚙️ Building features and targets...")
    df = build_features(df_raw)
    df = build_targets(df)
    df = target_generating_ranking(df)

    # 3. Define the optimized feature list
    best_features = candidate_features
    
    # Sanity check: Ensure all features were built successfully
    missing_features = [f for f in best_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"⚠️ Missing features after build_features: {missing_features}")
        
    # 4. Generate and save the artifacts
    output_dir = os.path.join("data", "pretrained")
    
    pretrain_and_save_artifacts(
        df=df,
        selected_features=best_features,
        use_mega_alpha=False,
        output_dir=output_dir,
        vnindex_df=vnindex_df,
    )
    
    print("\n🎉 Precomputation complete! Your Streamlit app is ready to be updated.")

if __name__ == "__main__":
    main()