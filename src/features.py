import pandas as pd
import numpy as np

def rsi(df, window_length=14):
    grouped = df.groupby("Symbol")
    delta = grouped["close"].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use the same grouping logic for consistency
    avg_gain = gain.groupby(df['Symbol']).transform(lambda x: x.ewm(com=window_length-1, adjust=False).mean())
    avg_loss = loss.groupby(df['Symbol']).transform(lambda x: x.ewm(com=window_length-1, adjust=False).mean())
    
    # 4. Calculate Relative Strength (RS) and the final RSI
    rs = avg_gain / avg_loss
    df[f'RSI_{window_length}'] = 100 - (100 / (1 + rs))
    
    # 5. Handle edge cases
    df[f'RSI_{window_length}'] = df[f'RSI_{window_length}'].fillna(100)
    
    return df

def volume(df):
   grouped_volume = df.groupby("Symbol")["volume"]
   df['vol_5d'] = grouped_volume.transform(lambda x: x.rolling(window=5).mean())
   df['vol_1m'] = grouped_volume.transform(lambda x: x.rolling(window=21).mean())
   df['vol_3m'] = grouped_volume.transform(lambda x: x.rolling(window=63).mean())
   df['volume_surge_monthly'] = df['vol_1m'] / df['vol_3m']
   df['volume_surge_weekly'] = df['vol_5d'] / df['vol_1m']
   return df

def return_ln(df):
    grouped_close = df.groupby("Symbol")["close"]
    close_shift_1 = grouped_close.shift(1)
    
    df["log_ret_1w"] = np.log(close_shift_1 / grouped_close.shift(6))
    df["log_ret_1m"] = np.log(close_shift_1 / grouped_close.shift(21))
    df["log_ret_3m"] = np.log(close_shift_1 / grouped_close.shift(63))
    df["log_ret_6m"] = np.log(close_shift_1 / grouped_close.shift(126))
    df["log_ret_1y"] = np.log(close_shift_1 / grouped_close.shift(252))
    
    return df

def volatility(df):
    grouped = df.groupby("Symbol")
    df["log_ret_daily"] = np.log(df["close"] / grouped["close"].shift(1))

    df["volatility_1w"] = grouped["log_ret_daily"].transform(lambda x: x.rolling(5).std() * np.sqrt(252))
    df["volatility_1m"] = grouped["log_ret_daily"].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    df["volatility_3m"] = grouped["log_ret_daily"].transform(lambda x: x.rolling(63).std() * np.sqrt(252))
    df["volatility_6m"] = grouped["log_ret_daily"].transform(lambda x: x.rolling(126).std() * np.sqrt(252))
    
    df["volatility_shock_monthly"] = df['volatility_1m']/df['volatility_3m']
    df["volatility_shock_weekly"] = df['volatility_1w']/df['volatility_1m']
    
    return df

def MA(df):
    grouped = df.groupby("Symbol")
    
    df['SMA_14'] = grouped['close'].transform(lambda x: x.rolling(window=14).mean())
    df['SMA_20'] = grouped['close'].transform(lambda x: x.rolling(window=20).mean())
    df['SMA_50'] = grouped['close'].transform(lambda x: x.rolling(window=50).mean())
    df['SMA_100'] = grouped['close'].transform(lambda x: x.rolling(window=100).mean())
    
    df['EMA_14'] = grouped['close'].transform(lambda x: x.ewm(span=14, adjust=False).mean())
    df['EMA_20'] = grouped['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = grouped['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA_100'] = grouped['close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    
    df['dist_SMA_100'] = df['close'] / df['SMA_100']
    df['dist_SMA_14'] = df['close']/df['SMA_14']
    df['dist_SMA_50'] = df['close']/df['SMA_50']

    return df 

def build_features(df):
    df = df.copy()
    df = df.sort_values(["Symbol", "date"])
    
    df = return_ln(df)
    df = volatility(df)
    df = MA(df)
    df = volume(df)
    df = rsi(df)
    
    feature_cols = [col for col in df.columns if col not in ['Symbol', 'date', 'close', 'high', 'low', 'open', 'volume']]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=feature_cols, inplace=True)
    
    df['next_1m_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-21) / x))
    df['next_1w_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-5) / x))
    
    df.index = range(1, len(df) + 1)
    return df

def target_generating_ranking(df, freq='M'):
    df = df.copy()
    
    if freq not in ['M', 'W']:
        raise ValueError("The 'freq' parameter must be either 'M' or 'W'.")

    df = df.sort_values(by=['date', 'Symbol'])
    df['qid'] = df.groupby('date').ngroup()

    if freq == 'M':
        vol = df['volatility_3m'].replace(0, np.nan)
        df['risk_adj_ret'] = df['next_1m_ret'] / vol
    elif freq == 'W':
        vol = df['volatility_1m'].replace(0, np.nan)
        df['risk_adj_ret'] = df['next_1w_ret'] / vol

    df['risk_adj_ret'] = df['risk_adj_ret'].replace([np.inf, -np.inf], np.nan)

    def robust_qcut(x):
        if x.dropna().shape[0] < 5:  # need at least q=5 valid values to form quintiles
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x.rank(method='first'), q=5, labels=False)

    df['target_quintile'] = df.groupby('date')['risk_adj_ret'].transform(robust_qcut)

    df = df.dropna(subset=['target_quintile'])
    df['target_quintile'] = df['target_quintile'].astype(int) 
    df = df.sort_values(by=['date', 'Symbol'])
    
    return df

def apply_cross_sectional_ranking(df, feature_cols):
    df_ranked = df.copy()
    
    if 'date' not in df_ranked.columns:
        raise ValueError("DataFrame must have a 'date' column.")
        
    csr_features = []
    print(f"⚖️ Applying cross-sectional ranking to {len(feature_cols)} features...")
    
    for col in feature_cols:
        if col not in df_ranked.columns:
            continue
            
        csr_col_name = f"csr_{col}"
        df_ranked[csr_col_name] = df_ranked.groupby('date')[col].rank(pct=True)
        csr_features.append(csr_col_name)
    
    print(f"✅ Successfully created {len(csr_features)} cross-sectionally normalised features.")
    return df_ranked, csr_features