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
    
    # 5. Handle edge cases (if a stock only goes up, avg_loss is 0, making RS infinity and RSI exactly 100)
    df[f'RSI_{window_length}'] = df[f'RSI_{window_length}'].fillna(100)
    
    return df

def volume(df):
   grouped_volume = df.groupby("Symbol")["volume"]
   df['vol_5d_avg'] = grouped_volume.transform(lambda x: x.rolling(window=5).mean())
   df['vol_21d_avg'] = grouped_volume.transform(lambda x: x.rolling(window=21).mean())
   # The ratio: >1 means volume is increasing, <1 means volume is drying up
   df['vol_3m_avg'] = grouped_volume.transform(lambda x: x.rolling(window=63).mean())
   df['volume_surge_monthly'] = df['vol_21d_avg'] / df['vol_3m_avg']
   df['volume_surge_weekly'] = df['vol_5d_avg'] / df['vol_21d_avg']
   return df

def return_ln(df):
    # 1. Create the grouped object ONCE
    grouped_close = df.groupby("Symbol")["close"]
    
    # 2. Calculate the base shift (yesterday's close) ONCE
    close_shift_1 = grouped_close.shift(1)
    
    # 3. Calculate returns using the pre-grouped objects
    df["log_ret_1w"] = np.log(close_shift_1 / grouped_close.shift(6))
    df["log_ret_1m"] = np.log(close_shift_1 / grouped_close.shift(21))
    df["log_ret_3m"] = np.log(close_shift_1 / grouped_close.shift(63))
    df["log_ret_6m"] = np.log(close_shift_1 / grouped_close.shift(126))
    df["log_ret_1y"] = np.log(close_shift_1 / grouped_close.shift(252))
    
    return df

def volatility(df):
    # 1. Pre-group the data
    grouped = df.groupby("Symbol")

    # 2. Calculate Daily Log Returns first 
    df["log_ret_daily"] = np.log(df["close"] / grouped["close"].shift(1))

    df["volatility_1w"] = grouped["log_ret_daily"].transform(
        lambda x: x.rolling(5).std() * np.sqrt(252)
    )

    # 3. Calculate Rolling Volatility using Trading Day windows
    df["volatility_1m"] = grouped["log_ret_daily"].transform(
        lambda x: x.rolling(21).std() * np.sqrt(252)
    )
    df["volatility_3m"] = grouped["log_ret_daily"].transform(
        lambda x: x.rolling(63).std() * np.sqrt(252)
    )
    df["volatility_6m"] = grouped["log_ret_daily"].transform(
        lambda x: x.rolling(126).std() * np.sqrt(252)
    )
    df["volatility_shock_monthly"] = df['volatility_1m']/df['volatility_3m']
    df["volatility_shock_weekly"] = df['volatility_1w']/df['volatility_1m']
    
    return df

def MA(df):
    grouped = df.groupby("Symbol")
    
    # Using transform for SMA is safer and avoids index alignment issues

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

    # FIXED: Added the missing return statement
    return df 

def build_features(df):
    df = df.copy()
    df = df.sort_values(["Symbol", "date"])
    
    # 1. Build Features
    df = return_ln(df)
    df = volatility(df)
    df = MA(df)
    df = volume(df)
    df = rsi(df)
    
    # 2. Define Feature Columns 
    # Create a list of all the features you just built
    feature_cols = [col for col in df.columns if col not in ['Symbol', 'date', 'close', 'high', 'low', 'open', 'volume']]
    
    # 3. Safely Drop Feature NaNs ONLY
    # This removes the early rows (e.g., first 126 days for MA_100) 
    # but keeps the most recent days intact.
    df.dropna(subset=feature_cols, inplace=True)
    
    # 4. Generate Targets LAST
    df['next_1m_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-21) / x))
    df['next_1w_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-5) / x))
    
    df.index = range(1, len(df) + 1)
    return df

def target_generating_ranking(df, freq='M'):
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # 1. Input Validation
    if freq not in ['M', 'W']:
        raise ValueError("The 'freq' parameter must be either 'M' or 'W'.")

    # 2. QID generation
    df = df.sort_values(by=['date', 'Symbol'])
    df['qid'] = df.groupby('date').ngroup()

    # 3. Calculate Risk-Adjusted Return
    if freq == 'M':
        # Replace 0 volatility with NaN to avoid division by zero
        vol = df['volatility_3m'].replace(0, np.nan)
        df['risk_adj_ret'] = df['next_1m_ret'] / vol
    elif freq == 'W':
        vol = df['volatility_1m'].replace(0, np.nan)
        df['risk_adj_ret'] = df['next_1w_ret'] / vol

    # Replace any infinities that might have slipped through with NaN
    df['risk_adj_ret'] = df['risk_adj_ret'].replace([np.inf, -np.inf], np.nan)

    # 4. Rank and cut (Handling ties safely)
    def robust_qcut(x):
        # x.rank(method='first') breaks ties so we always get exactly 5 bins
        # It prevents duplicates='drop' from silently changing the number of classes
        if x.dropna().empty:
            return np.nan
        return pd.qcut(x.rank(method='first'), q=5, labels=False)

    # Rank cross-sectionally
    df['target_quintile'] = df.groupby('date')['risk_adj_ret'].transform(robust_qcut)

    # 5. Cleanup
    df = df.dropna(subset=['target_quintile'])
    
    # Ensure the target is an integer for ML models (dropna makes this safe)
    df['target_quintile'] = df['target_quintile'].astype(int) 

    # Final sort for modeling
    df = df.sort_values(by=['date', 'Symbol'])
    
    return df



def test_train_spliter(df, test_start, features):
    df = df.copy()
    
    # 1. Force the date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Ensure test_start is also a Timestamp
    test_start = pd.to_datetime(test_start)
    train_cutoff = test_start - pd.Timedelta(days=30)
    
    # Now the comparison will work perfectly
    train_df = df[df['date'] < train_cutoff]
    test_df = df[df['date'] >= test_start]

    # 3. Extract X, y, and qids for Training
    X_train = train_df[features]
    y_train = train_df['target_quintile']
    qids_train = train_df['qid']

    # 4. Extract X, y, and qids for Testing
    X_test = test_df[features]
    y_test = test_df['target_quintile']
    qids_test = test_df['qid']

    return X_train,y_train,qids_train,X_test,y_test,qids_test,test_df

def base_model():
    return {
        'tree_method': 'hist',
        'objective': 'rank:ndcg', 
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'random_state': 42
    }

