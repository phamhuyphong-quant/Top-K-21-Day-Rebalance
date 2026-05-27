import pandas as pd
import numpy as np
import torch
import random
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.alpha_mining import WorldQuantAlphas
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # If using PyTorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
# In WorldQuantAlphas or features.py
def rsi(df, window_length=14):
    """
    Computes RSI using only data available on or before each row's date.

    Row i (date T) receives RSI computed from close prices up to and including
    close_{T-1} (i.e. the previous day's close), so no same-day information leaks
    into the feature. Internally we shift close by 1 before differencing.
    """
    grouped = df.groupby("Symbol")
    # Shift close by 1 so every calculation at row T uses data up to T-1
    lagged_close = grouped["close"].shift(1)
    delta = lagged_close.groupby(df["Symbol"]).diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(df["Symbol"]).transform(
        lambda x: x.ewm(com=window_length - 1, adjust=False).mean()
    )
    avg_loss = loss.groupby(df["Symbol"]).transform(
        lambda x: x.ewm(com=window_length - 1, adjust=False).mean()
    )

    rs = avg_gain / avg_loss
    df[f"RSI_{window_length}"] = 100 - (100 / (1 + rs))
    return df


def volume(df):
    """
    Computes rolling volume averages using only data available before each row's date.

    Row i (date T) receives volume features computed from volume up to T-1.
    We shift volume by 1 so the rolling windows see only past data.
    """
    grouped_volume = df.groupby("Symbol")["volume"].shift(1).groupby(df["Symbol"])
    df["vol_5d"] = grouped_volume.transform(lambda x: x.rolling(window=5).mean())
    df["vol_1m"] = grouped_volume.transform(lambda x: x.rolling(window=21).mean())
    df["vol_3m"] = grouped_volume.transform(lambda x: x.rolling(window=63).mean())
    df["volume_surge_monthly"] = df["vol_1m"] / df["vol_3m"]
    df["volume_surge_weekly"] = df["vol_5d"] / df["vol_1m"]
    return df


def return_ln(df):
    """
    Computes log returns using only data available before each row's date.

    Row i (date T) receives log returns computed entirely from close prices at
    T-1 and earlier. We shift by an extra 1 relative to the original offsets so
    log_ret_1w at row T = log(close_{T-1} / close_{T-6}), etc.
    """
    grouped_close = df.groupby("Symbol")["close"]

    df["log_ret_1w"] = np.log(grouped_close.shift(1) / grouped_close.shift(6))
    df["log_ret_1m"] = np.log(grouped_close.shift(1) / grouped_close.shift(22))
    df["log_ret_3m"] = np.log(grouped_close.shift(1) / grouped_close.shift(64))
    df["log_ret_6m"] = np.log(grouped_close.shift(1) / grouped_close.shift(127))
    df["log_ret_1y"] = np.log(grouped_close.shift(1) / grouped_close.shift(253))
    return df


def volatility(df):
    """
    Computes rolling volatility using only data available before each row's date.

    Daily log returns are computed from lagged close (shift 1 → shift 2), so the
    rolling windows at row T only see returns up to the T-1 → T-2 pair.
    """
    grouped = df.groupby("Symbol")
    # Daily log return: log(close_{T-1} / close_{T-2}) so row T has no T leakage
    lagged_close = grouped["close"].shift(1)
    log_ret_daily = np.log(lagged_close / lagged_close.groupby(df["Symbol"]).shift(1))

    df["volatility_1w"] = log_ret_daily.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(5).std() * np.sqrt(252)
    )
    df["volatility_1m"] = log_ret_daily.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(21).std() * np.sqrt(252)
    )
    df["volatility_3m"] = log_ret_daily.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(63).std() * np.sqrt(252)
    )
    df["volatility_6m"] = log_ret_daily.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(126).std() * np.sqrt(252)
    )

    df["volatility_shock_monthly"] = df["volatility_1m"] / df["volatility_3m"]
    df["volatility_shock_weekly"] = df["volatility_1w"] / df["volatility_1m"]
    return df


def MA(df):
    """
    Computes moving averages and distance-from-MA features using only past data.

    Row i (date T) receives MAs and distance ratios computed from close prices up
    to T-1. The distance ratio numerator also uses close_{T-1} (not close_T), so
    the feature is fully self-consistent and look-ahead-free.
    """
    grouped = df.groupby("Symbol")
    lagged_close = grouped["close"].shift(1)
    lagged_grouped = lagged_close.groupby(df["Symbol"])

    df["SMA_9"]   = lagged_grouped.transform(lambda x: x.rolling(window=9).mean())
    df["SMA_21"]  = lagged_grouped.transform(lambda x: x.rolling(window=21).mean())
    df["SMA_50"]  = lagged_grouped.transform(lambda x: x.rolling(window=50).mean())
    df["SMA_100"] = lagged_grouped.transform(lambda x: x.rolling(window=100).mean())
    df["SMA_200"] = lagged_grouped.transform(lambda x: x.rolling(window=200).mean())

    df["EMA_9"]   = lagged_grouped.transform(lambda x: x.ewm(span=9,   adjust=False).mean())
    df["EMA_21"]  = lagged_grouped.transform(lambda x: x.ewm(span=21,  adjust=False).mean())
    df["EMA_50"]  = lagged_grouped.transform(lambda x: x.ewm(span=50,  adjust=False).mean())
    df["EMA_100"] = lagged_grouped.transform(lambda x: x.ewm(span=100, adjust=False).mean())
    df["EMA_200"] = lagged_grouped.transform(lambda x: x.ewm(span=200, adjust=False).mean())

    # Distance ratios: lagged_close / lagged_MA — both sides use T-1 data
    df["dist_SMA_9"]   = lagged_close / df["SMA_9"]
    df["dist_SMA_21"]  = lagged_close / df["SMA_21"]
    df["dist_SMA_50"]  = lagged_close / df["SMA_50"]
    df["dist_SMA_100"] = lagged_close / df["SMA_100"]
    df["dist_SMA_200"] = lagged_close / df["SMA_200"]

    df["dist_EMA_9"]   = lagged_close / df["EMA_9"]
    df["dist_EMA_21"]  = lagged_close / df["EMA_21"]
    df["dist_EMA_50"]  = lagged_close / df["EMA_50"]
    df["dist_EMA_100"] = lagged_close / df["EMA_100"]
    df["dist_EMA_200"] = lagged_close / df["EMA_200"]
    return df


def price_structure(df):
    """
    Computes price-structure features using only data available before each row's date.

    dist_52w_high: ratio of close_{T-1} to the rolling 252-day max of close up to T-1.
    log_ret_skip1m: skip-1-month momentum = log(close_{T-22} / close_{T-127}),
        entirely in the past relative to T.
    """
    grouped = df.groupby("Symbol")
    lagged_close = grouped["close"].shift(1)
    df["dist_52w_high"] = lagged_close / lagged_close.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(252).max()
    )
    df["log_ret_skip1m"] = np.log(
        grouped["close"].shift(22) / grouped["close"].shift(127)
    )
    return df


def volume_quality(df):
    """
    Computes OBV trend and price-volume divergence using only past data.

    Both features are built from vol_1m, volume_surge_monthly, and log_ret_1m,
    which are themselves already computed from lagged inputs by their respective
    functions. No additional shift is needed here.
    """
    grouped = df.groupby("Symbol")
    # OBV: sign of daily close change × volume, rolled over 21 days, normalised by vol_1m.
    # Both close (shifted by 1 inside volume/rsi) and volume (shifted by 1 in volume())
    # are already lagged, so this combination is look-ahead-free.
    lagged_close = grouped["close"].shift(1)
    lagged_volume = grouped["volume"].shift(1)

    obv_raw = (
        grouped.apply(
            lambda x: (
                np.sign(x["close"].shift(1).diff()) * x["volume"].shift(1)
            ).rolling(21).sum(),
            include_groups=False,
        )
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )
    df["obv_trend"] = obv_raw / df["vol_1m"]

    # Price-volume divergence uses already-lagged log_ret_1m and volume_surge_monthly
    df["price_vol_divergence"] = df["log_ret_1m"] / (df["volume_surge_monthly"] + 0.001)
    return df


def turnover_ratio(df):
    """
    Structural turnover proxy: 252d avg volume / 504d avg volume.
    Captures whether a stock is in a persistently high- or low-activity regime
    relative to its own 2-year baseline. Slow-moving — valid at monthly rebalance.
    Uses lagged volume (shift 1) so row T sees data up to T-1 only.
    NOT a return signal — goes to XGBoost directly, not the deep combiner.
    """
    lagged_vol = df.groupby("Symbol")["volume"].shift(1)
    vol_252 = lagged_vol.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(252, min_periods=126).mean()
    )
    vol_504 = lagged_vol.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(504, min_periods=252).mean()
    )
    df["turnover_12m"] = vol_252 / (vol_504 + 1)
    return df


def limit_bias(df, d=60):
    """
    Up-limit days minus down-limit days over d=60 days.
    Measures sustained demand pressure (up-locks) vs supply pressure (down-locks)
    using VN's 7% daily price ceiling as the threshold.
    Window extended to 60d (from 20d) so the signal persists across a monthly hold.
    NOT a directional return signal — provides market microstructure context for XGBoost.
    Uses lagged close (shift 1) for full look-ahead-free compliance.
    """
    lagged_close = df.groupby("Symbol")["close"].shift(1)
    daily_ret = lagged_close.groupby(df["Symbol"]).pct_change()
    up_days = (daily_ret > 0.065).astype(float)
    dn_days = (daily_ret < -0.065).astype(float)
    up_rate = up_days.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(d, min_periods=d // 2).mean()
    )
    dn_rate = dn_days.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(d, min_periods=d // 2).mean()
    )
    df["limit_bias_60d"] = up_rate - dn_rate
    return df


def herding_dispersion(df):
    """
    Cross-sectional std of lagged daily returns across all symbols on each date.
    Low value = stocks moving together (herding regime, retail-driven VN market).
    High value = stocks diverging (stock-picking environment).
    Same value for every symbol on a given date — pure market-regime context for XGBoost.
    NOT a per-stock return signal, so meaningless in the deep combiner
    (zero cross-sectional variance → no gradient signal during attention training).
    Uses lagged close (shift 1) for look-ahead-free compliance.
    """
    lagged_close = df.groupby("Symbol")["close"].shift(1)
    daily_ret = lagged_close.groupby(df["Symbol"]).pct_change()
    df["herding_dispersion"] = daily_ret.groupby(df["date"]).transform("std")
    return df


def amihud_illiquidity(df):
    """
    Amihud (2002) illiquidity ratio: |daily_ret| / volume, averaged over 21 days.
    Log-transformed (log1p) to compress the extreme right tail — raw values span
    several orders of magnitude across VN stocks and would dominate any linear model.
    Captures the illiquidity premium: less-traded stocks earn higher returns in thin
    markets like VN. Structural characteristic — valid at monthly rebalance.
    NOT placed in the deep combiner because its raw scale is incompatible with
    the other WQ_Alpha signals (would corrupt attention weight training).
    Uses lagged close and volume (shift 1) for look-ahead-free compliance.
    """
    lagged_close = df.groupby("Symbol")["close"].shift(1)
    lagged_vol   = df.groupby("Symbol")["volume"].shift(1)
    abs_ret = lagged_close.groupby(df["Symbol"]).pct_change().abs()
    illiq_raw = abs_ret / (lagged_vol + 1)
    illiq_21d = illiq_raw.groupby(df["Symbol"]).transform(
        lambda x: x.rolling(21, min_periods=5).mean()
    )
    df["amihud_illiquidity"] = np.log1p(illiq_21d)
    return df

def adtv(df, window=20):
    lagged_value = df.groupby('Symbol').apply(
        lambda x: (x['close'] * x['volume']).shift(1)
    ).reset_index(level=0, drop=True)
    df['adtv'] = lagged_value.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window, min_periods=window//2).mean()
    )
    return df

def build_features(df, min_stocks_per_date: int = 50):
    """
    Build all features for the dataset.

    Contract: row i (date T) contains the OHLCV of date T and feature values
    computed exclusively from data on or before date T-1. Each sub-function
    handles its own lagging internally, so no bulk shift is applied here.

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

    # Context / structural features for XGBoost — not return signals, not in deep combiner
    df = turnover_ratio(df)
    df = limit_bias(df)
    df = herding_dispersion(df)
    df = amihud_illiquidity(df)
    df = adtv(df)
    # WorldQuant alphas: WorldQuantAlphas.__init__ builds self.ldf (lagged OHLCV)
    # internally, so generate_all() respects the same T-1 contract as the functions above.
    wq_cols_df = WorldQuantAlphas(df).generate_all()
    for col in wq_cols_df.columns:
        df[col] = wq_cols_df[col]

    feature_cols = [
        col for col in df.columns
        if col not in ["Symbol", "date", "close", "high", "low", "open", "volume"]
    ]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=feature_cols, inplace=True)

    # Drop dates that have fewer than min_stocks_per_date surviving symbols
    stocks_per_date = df.groupby("date")["Symbol"].transform("count")
    thin_mask = stocks_per_date < min_stocks_per_date
    if thin_mask.any():
        dropped_dates = df.loc[thin_mask, "date"].nunique()
        logging.getLogger(__name__).warning(
            "Dropping %d date(s) with fewer than %d stocks after NaN removal.",
            dropped_dates,
            min_stocks_per_date,
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