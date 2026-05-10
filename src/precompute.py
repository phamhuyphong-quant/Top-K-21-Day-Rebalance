import os
import sys
import pandas as pd

# Add the project root to the Python path so we can cleanly import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import build_features, target_generating_ranking
from src.evaluation import generate_and_save_pretrained_model
from src.alpha_mining import WorldQuantAlphas
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
        
    print(f"📦 Loading data from {data_path}...")
    df_raw = pd.read_parquet(data_path)
    
    # 2. Process features and targets (matching app.py logic exactly)
    print("⚙️ Building features and targets...")
    wq = WorldQuantAlphas(df_raw)
    wq_cols_df = wq.generate_all()
    df_raw[wq_cols_df.columns] = wq_cols_df.values
    df = build_features(df_raw)
    df = target_generating_ranking(df)
    
    # 3. Define the optimized feature list
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
    
    # Sanity check: Ensure all features were built successfully
    missing_features = [f for f in best_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"⚠️ Missing features after build_features: {missing_features}")
        
    # 4. Generate and save the artifacts
    output_dir = os.path.join("data", "pretrained")
    
    generate_and_save_pretrained_model(
        df=df,
        selected_features=best_features,
        use_mega_alpha=False,
        output_dir=output_dir
    )
    
    print("\n🎉 Precomputation complete! Your Streamlit app is ready to be updated.")

if __name__ == "__main__":
    main()