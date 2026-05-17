import pandas as pd
import numpy as np
import torch
import random
import os
import logging
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # If using PyTorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
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
    
    df["log_ret_1w"] = np.log(df["close"] / grouped_close.shift(5))
    df["log_ret_1m"] = np.log(df["close"] / grouped_close.shift(21))
    df["log_ret_3m"] = np.log(df["close"] / grouped_close.shift(63))
    df["log_ret_6m"] = np.log(df["close"] / grouped_close.shift(126))
    df["log_ret_1y"] = np.log(df["close"] / grouped_close.shift(252))
    
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
    df.drop(columns=['log_ret_daily'], inplace=True)
    return df

def MA(df):
    grouped = df.groupby("Symbol")
    
    df['SMA_9'] = grouped['close'].transform(lambda x: x.rolling(window=9).mean())
    df['SMA_21'] = grouped['close'].transform(lambda x: x.rolling(window=21).mean())
    df['SMA_50'] = grouped['close'].transform(lambda x: x.rolling(window=50).mean())
    df['SMA_100'] = grouped['close'].transform(lambda x: x.rolling(window=100).mean())
    df['SMA_200'] = grouped['close'].transform(lambda x: x.rolling(window=200).mean())
    

    df['EMA_9'] = grouped['close'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['EMA_21'] = grouped['close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['EMA_50'] = grouped['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA_100'] = grouped['close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    df['EMA_200'] = grouped['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    df['dist_SMA_9'] = df['close'] / df['SMA_9']
    df['dist_SMA_21'] = df['close']/df['SMA_21']
    df['dist_SMA_50'] = df['close']/df['SMA_50']
    df['dist_SMA_100'] = df['close'] / df['SMA_100']
    df['dist_SMA_200'] = df['close'] / df['SMA_200']

    df['dist_EMA_9'] = df['close'] / df['EMA_9']
    df['dist_EMA_21'] = df['close']/df['EMA_21']
    df['dist_EMA_50'] = df['close']/df['EMA_50']
    df['dist_EMA_100'] = df['close'] / df['EMA_100']
    df['dist_EMA_200'] = df['close'] / df['EMA_200']
    return df 

# In features.py — new function:
def price_structure(df):
    grouped = df.groupby("Symbol")
    df['dist_52w_high']    = df['close'] / grouped['close'].transform(lambda x: x.rolling(252).max())
    df['log_ret_skip1m']   = np.log(grouped['close'].shift(21) / grouped['close'].shift(126))
    return df
def volume_quality(df):
    # On-Balance Volume trend (normalized)
    grouped = df.groupby("Symbol")
    df['obv_trend'] = (grouped.apply(
    lambda x: (np.sign(x['close'].diff()) * x['volume']).rolling(21).sum(),
    include_groups=False
)).reset_index(level=0, drop=True) / df['vol_1m']

    # Price-volume divergence — rising price on falling volume is a warning
    df['price_vol_divergence'] = df['log_ret_1m'] / (df['volume_surge_monthly'] + 0.001)
    return df

def build_features(df, min_stocks_per_date: int = 50):
    """
    Build all features for the dataset.

    min_stocks_per_date: dates where fewer than this many symbols survive
    the NaN-drop are removed entirely from the output. Defaults to 50.
    """
    df = df.copy()
    df = df.sort_values(["Symbol", "date"])

    df = return_ln(df)
    df = volatility(df)
    df = MA(df)
    df = volume(df)
    df = rsi(df)
    df = volume_quality(df)
    df = price_structure(df)

    feature_cols = [col for col in df.columns if col not in ['Symbol', 'date', 'close', 'high', 'low', 'open', 'volume']]
    df[feature_cols] = df.groupby('Symbol')[feature_cols].shift(1)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=feature_cols, inplace=True)

    # Drop dates that have fewer than min_stocks_per_date surviving symbols
    stocks_per_date = df.groupby("date")["Symbol"].transform("count")
    thin_mask = stocks_per_date < min_stocks_per_date
    if thin_mask.any():
        dropped_dates = df.loc[thin_mask, "date"].nunique()
        import logging
        logging.getLogger(__name__).warning(
            "Dropping %d date(s) with fewer than %d stocks after NaN removal.",
            dropped_dates, min_stocks_per_date,
        )
        df = df[~thin_mask]

 

    df.index = range(1, len(df) + 1)
    return df
def build_targets(df):
    df = df.copy()
    df['next_1m_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-21) / x))
    df['next_1w_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-5) / x))
    return df


def target_generating_ranking(df, freq='M'):
    df = df.copy()
    
    if freq not in ['M', 'W']:
        raise ValueError("The 'freq' parameter must be either 'M' or 'W'.")

    df = df.sort_values(by=['date', 'Symbol'])
    
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
    df['qid'] = df.groupby('date').ngroup() 
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