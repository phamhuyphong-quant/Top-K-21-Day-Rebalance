import pandas as pd
import numpy as np
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.alpha_mining import WorldQuantAlphas
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

def robust_market_regime_pipeline_monthly(df, return_horizon=21, rolling_window=63, num_buckets=4,
                                            liquidity_mask=None):
    """
    Same as robust_market_regime_pipeline, but dispersion (MAD) is computed on
    ~1-month (return_horizon-day) forward-looking-safe returns instead of
    1-day returns.

    Why: the original daily-MAD regime measures day-to-day co-movement, which
    is a different (and much noisier/faster) statistic than how much stocks'
    *monthly* returns end up separated from each other -- which is the
    horizon your ranker's target (next_1m_ret) actually lives on. Using the
    same horizon for the regime signal as for the target return makes the
    regime bucket a much more relevant conditioning variable for quintile
    spread analysis.

    Look-ahead safety: at row t, the return used is close[t-1] vs
    close[t-1-return_horizon] -- i.e. an already-realized return_horizon-day
    return as of t-1, fully in the past relative to date t. It does NOT
    overlap with the forward next_1m_ret window starting at t.

    liquidity_mask: optional boolean Series aligned to df's index. When given,
    the cross-sectional median/MAD at each date is computed only over rows
    flagged liquid — rows are NOT dropped, so the per-Symbol pct_change/shift
    above still sees a continuous history regardless of which stocks are
    liquid on any given day.
    """
    df = df.copy()
    df["_is_liquid"] = liquidity_mask.reindex(df.index) if liquidity_mask is not None else True

    # Bước 1: Sắp xếp dữ liệu cấu trúc chuỗi thời gian
    df = df.sort_values(["Symbol", "date"]).reset_index(drop=True)

    # Bước 2: Tính return_horizon-day return, rồi lag 1 để tránh contamination
    # (groupby lại trước khi shift để tránh Cross-symbol contamination)
    # Luôn chạy trên TOÀN BỘ chuỗi liên tục (không phụ thuộc liquidity_mask).
    monthly_ret = df.groupby("Symbol")["close"].pct_change(return_horizon)
    df["monthly_ret_lag1"] = monthly_ret.groupby(df["Symbol"]).shift(1)

    # Bước 3: Tính Robust Dispersion bằng Vectorized MAD trên monthly return.
    # Cross-sectional theo date -> chỉ tổng hợp trên các dòng đủ thanh khoản.
    masked_ret = df["monthly_ret_lag1"].where(df["_is_liquid"])
    daily_median = masked_ret.groupby(df["date"]).transform("median")
    abs_deviation = (masked_ret - daily_median).abs()

    daily_mad = abs_deviation.groupby(df["date"]).median()

    # Bước 4: Làm mượt bằng Rolling Mean để xác định Regime bền vững
    rolling_mad = daily_mad.rolling(window=rolling_window, min_periods=max(1, rolling_window // 2)).mean()

    # Bước 5: Tính Point-in-time Percentile qua Expanding Rank
    pit_percentile = rolling_mad.rolling(window=252).rank(pct=True)

    # Bước 6: Fixed Bins cố định thay vì để pd.cut tự tính toán
    bin_edges = np.linspace(0, 1, num_buckets + 1)
    bucket_labels = [f"Q{i}" for i in range(1, num_buckets + 1)]

    pit_buckets = pd.cut(pit_percentile, bins=bin_edges, labels=bucket_labels, include_lowest=True)

    # Bước 7: Ánh xạ kết quả trở lại DataFrame tổng
    df["regime_disp_raw_monthly"] = df["date"].map(rolling_mad)
    df["regime_percentile_monthly"] = df["date"].map(pit_percentile)
    df["regime_bucket_monthly"] = df["date"].map(pit_buckets)

    df = df.drop(columns=["_is_liquid"])
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

def adtv(df, window=21):
    lagged_value = df.groupby('Symbol').apply(
        lambda x: (x['close'] * x['volume']).shift(1)
    ).reset_index(level=0, drop=True)
    df['adtv'] = lagged_value.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window, min_periods=window//2).mean()
    )
    return df
def market_breadth(df, liquidity_mask=None):
    """
    liquidity_mask: optional boolean Series aligned to df's index. When given,
    breadth on each date is the fraction of *liquid* stocks with positive
    log_ret_1m, rather than the fraction across the whole (incl. illiquid)
    universe.
    """
    df = df.copy()
    mask = liquidity_mask.reindex(df.index) if liquidity_mask is not None else pd.Series(True, index=df.index)
    masked_ret = df['log_ret_1m'].where(mask)
    daily_breadth = masked_ret.groupby(df['date']).apply(lambda x: (x.dropna() > 0).mean()).sort_index()
    breadth_ema21 = daily_breadth.ewm(span=21, adjust=False).mean()
    df['breadth_ema21'] = df['date'].map(breadth_ema21)
    breadth_ema21_df = breadth_ema21.to_frame('breadth_ema21')
    rolling_25pct = breadth_ema21_df['breadth_ema21'].shift(1).rolling(252, min_periods=126).quantile(0.25)
    df['breadth_25pct_threshold'] = df['date'].map(rolling_25pct)
    df['narrow_breadth'] = df['breadth_ema21'] < df['breadth_25pct_threshold']
    return df
def make_magnitude_label(df, target_col='risk_adj_ret', date_col='date',
                          scale_max=100.0, winsor_pct=0.0):
    df = df.copy()

    def _scale_one_day(g):
        s = g[target_col]
        lo_clip, hi_clip = s.quantile(winsor_pct), s.quantile(1 - winsor_pct)
        s_clipped = s.clip(lo_clip, hi_clip)
        lo, hi = s_clipped.min(), s_clipped.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return pd.Series(np.nan, index=g.index)
        return (s_clipped - lo) / (hi - lo) * scale_max

    df['target_magnitude'] = (
        df.groupby(date_col, group_keys=False)
          .apply(lambda g: _scale_one_day(g))
    )
    return df
def make_adaptive_bucket_label(df, target_col='next_1m_ret', date_col='date',
                                 target_bucket_size=20, min_buckets=2,
                                 risk_adjust=True, vol_col='volatility_3m'):
    """
    Assigns each stock a bucket label (0 = worst, n_buckets-1 = best) per date,
    with n_buckets chosen so each bucket has ~target_bucket_size stocks.

    risk_adjust: if True (default), buckets are built on target_col / vol_col
    (same construction as target_generating_ranking's risk_adj_ret) instead
    of the raw return. Without this, the top bucket is just "highest raw
    return," which skews toward high-beta/high-volatility names — exactly
    the names that hurt a concentrated buy-only book in drawdowns. Set to
    False to recover the original raw-return behavior.

    vol_col: volatility column used as the risk-adjustment denominator. Must
    already exist on df (e.g. 'volatility_3m' from build_features) — this
    function does not compute it.

    Returns df with a new 'target_bucket' column. NaN where a date has fewer
    valid (non-NaN, non-zero-vol) observations than n_buckets — caller should
    dropna(subset=['target_bucket']) same as target_generating_ranking does
    for target_quintile.
    """
    df = df.copy()

    if risk_adjust:
        vol = df[vol_col].replace(0, np.nan)
        score = df[target_col] / vol
        score = score.replace([np.inf, -np.inf], np.nan)
    else:
        score = df[target_col]
    df['_bucket_score'] = score

    def _bucket_one_day(g):
        n = len(g)
        n_buckets = max(min_buckets, round(n / target_bucket_size))
        # Need at least n_buckets valid (non-NaN) scores to form n_buckets groups.
        if g['_bucket_score'].notna().sum() < n_buckets:
            return pd.Series(np.nan, index=g.index)
        # rank ascending on the (risk-adjusted) score -> highest score gets highest label
        # NaN scores get NaN rank (na_option='keep' default) -> qcut passes them through as NaN
        ranks = g['_bucket_score'].rank(method='first', ascending=True)
        return pd.qcut(ranks, n_buckets, labels=False)

    df['target_bucket'] = (
        df.groupby(date_col, group_keys=False)
          .apply(lambda g: _bucket_one_day(g))
    )
    df = df.drop(columns='_bucket_score')
    return df

def build_targets(df):
    df = df.copy()
    df['next_1m_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-21) / x))
    df['next_1w_ret'] = df.groupby('Symbol')['close'].transform(lambda x: np.log(x.shift(-5) / x))
    df = df.dropna(subset=['next_1m_ret', 'next_1w_ret'])
    return df


def build_features(df, min_stocks_per_date: int = 50, adtv_limit = None, generate_target:bool=True):
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
    
    df = amihud_illiquidity(df)

    if generate_target:
            df = build_targets(df)
            # risk_adj_ret: cùng công thức risk-adjustment mà make_adaptive_bucket_label
            # vừa dùng nội bộ để xếp target_bucket (next_1m_ret / volatility_3m) và mà
            # target_generating_ranking dùng cho risk_adj_ret/target_quintile — tính
            # tường minh ở đây để target_magnitude bên dưới đo cùng đại lượng risk-adjusted
            # với target_bucket, thay vì lợi nhuận thô.
            vol = df['volatility_3m'].replace(0, np.nan)
            df['risk_adj_ret'] = (df['next_1m_ret'] / vol).replace([np.inf, -np.inf], np.nan)
            
            df = make_magnitude_label(df)
        

    # Tính ADTV nhưng CHƯA xoá dòng: các phép rolling/lag theo Symbol (bên dưới,
    # trong WorldQuantAlphas và các hàm market-wide) cần chuỗi ngày liên tục cho
    # từng mã, nên nếu xoá dòng ngay bây giờ sẽ tạo khoảng trống (gap) làm sai
    # các cửa sổ rolling dài (100, 252 ngày...). Thay vào đó, dùng is_liquid như
    # một "mask": mọi phép cross-sectional (rank/mean theo ngày) sẽ chỉ tổng hợp
    # trên các dòng is_liquid=True, còn các phép rolling theo Symbol vẫn chạy
    # trên toàn bộ chuỗi liên tục.
    if adtv_limit is not None:
        df = adtv(df)
        is_liquid = df['adtv'] >= adtv_limit
    else:
        is_liquid = pd.Series(True, index=df.index)

    # WorldQuant alphas: WorldQuantAlphas.__init__ builds self.ldf (lagged OHLCV)
    # internally, so generate_all() respects the same T-1 contract as the functions
    # above. Cross-sectional pieces inside (rank/mean theo date, vd alpha 002, 013,
    # 016, 040, 101, 202) chỉ tổng hợp trên rổ liquidity_mask; per-symbol rolling
    # pieces vẫn chạy trên df đầy đủ, liên tục.
    wq_cols_df = WorldQuantAlphas(df, liquidity_mask=is_liquid).generate_all()
    for col in wq_cols_df.columns:
        df[col] = wq_cols_df[col]

    # TODO (cùng pattern): robust_market_regime_pipeline_monthly() và
    # market_breadth() bên dưới cũng trộn per-symbol rolling (pct_change,
    # log_ret) với cross-sectional median/mean theo date — nên áp dụng
    # is_liquid mask tương tự thay vì dựa vào việc lọc dòng trước đó.

    # Các hàm market-wide vẫn chạy trên df ĐẦY ĐỦ (chưa lọc dòng), dùng
    # is_liquid làm mask cho phần cross-sectional — để pct_change/log_ret theo
    # Symbol phía trong chúng không bị gãy chuỗi vì thiếu dòng.
    df = robust_market_regime_pipeline_monthly(df, liquidity_mask=is_liquid)

    # robust_market_regime_pipeline_monthly() reset lại index bên trong (sort
    # theo Symbol,date rồi reset_index) nên is_liquid (Series ngoài) không còn
    # canh đúng hàng nữa. Tính lại trực tiếp từ cột 'adtv' — cột thật luôn đi
    # đúng theo hàng qua mọi lần sort/reset — thay vì tái sử dụng Series cũ.
    is_liquid = df['adtv'] >= adtv_limit if adtv_limit is not None else pd.Series(True, index=df.index)

    market_ret = df['log_ret_3m'].where(is_liquid).groupby(df['date']).mean().sort_index()
    market_ema63 = market_ret.ewm(span=63, adjust=False).mean()
    df['market3m_ema63'] = df['date'].map(market_ema63)

    market_ret = df['log_ret_1m'].where(is_liquid).groupby(df['date']).mean().sort_index()
    market_ema21 = market_ret.ewm(span=21, adjust=False).mean()
    df['market1m_ema21'] = df['date'].map(market_ema21)

    df = market_breadth(df, liquidity_mask=is_liquid)

    # Lọc dòng thật sự CHỈ MỘT LẦN, sau khi mọi phép rolling/cross-sectional đã
    # tính xong — ngay trước khi hình thành bucket (bucket chỉ nên hình thành
    # trên rổ đã lọc thanh khoản; đây là phép cross-sectional theo số dòng mỗi
    # ngày, không phải rolling, nên an toàn khi chạy trên df đã lọc).
    if adtv_limit is not None:
        df = df[is_liquid]

 
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