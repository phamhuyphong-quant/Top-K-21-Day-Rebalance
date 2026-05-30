import numpy as np
import pandas as pd

from scipy.stats import spearmanr
# --- CÁC TOÁN TỬ WORLDQUANT CƠ BẢN ---
# --- WORLDQUANT BASE OPERATORS ---
# --- CẬP NHẬT TRONG alpha_mining.py ---
def to_series(df, col):
    if isinstance(col, str):
        return df[col]
    if isinstance(col, pd.Series):
        return col
    # Convert numpy array to a Series aligned with df's index
    return pd.Series(col, index=df.index)

def ts_delay(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).shift(d)

def ts_delta(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).diff(d)

def ts_mean(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).mean()
    )

def ts_min(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).min()
    )

def ts_rank(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).rank(pct=True)
    )

def ts_argmax(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).apply(np.argmax) + 1
    )
class WorldQuantAlphas:
    def __init__(self, df):
        self.df = df.copy().sort_values(by=["Symbol", "date"])

        # Build a lagged view of all OHLCV columns so that every alpha method
        # at row i (date T) only ever sees data from T-1 and earlier.
        # All ts_* operators applied to self.ldf are therefore look-ahead-free
        # without any additional shifting inside the individual alpha methods.
        ohlcv = ["open", "high", "low", "close", "volume"]
        self.ldf = self.df.copy()
        for col in ohlcv:
            self.ldf[col] = self.df.groupby("Symbol")[col].shift(1)

    def get_alpha_006(self):
        # Alpha#6: -corr(open, volume, 10)
        # ldf already has open_{T-1} and volume_{T-1}, so the 10-day rolling
        # correlation at row T uses days T-1 … T-10 — no leakage.
        g = self.ldf.groupby("Symbol")
        return (
            -1
            * g.apply(lambda x: x["open"].rolling(10).corr(x["volume"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )

    def get_alpha_024(self):
        # Alpha#24: conditional mean-reversion on 100-day price trend
        # All ts_* calls operate on ldf (close_{T-1} and earlier).
        mean_100  = ts_mean( self.ldf, "close", 100)
        delta_mean = ts_delta(self.ldf, mean_100,   100)
        low_delay  = ts_delay(self.ldf, "close",    100)
        cond = (delta_mean / (low_delay + 0.001)) <= 0.05
        return np.where(
            cond,
            -1 * (self.ldf["close"] - ts_min(self.ldf, "close", 100)),
            -1 * ts_delta(self.ldf, "close", 21),
        )

    def get_alpha_028(self):
        # Alpha#28: corr(adv20, low, 5) + (high + low) / 2 - close
        # adv20 = 20-day avg volume; all from ldf so past-only.
        ldf = self.ldf.copy()
        ldf["adv20"] = ts_mean(ldf, "volume", 20)
        corr = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["adv20"].rolling(5).corr(x["low"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )
        return corr + ((ldf["high"] + ldf["low"]) / 2) - ldf["close"]


    def get_alpha_001(self):
        # 12-1 month momentum: cumulative return over past 12 months, skipping the
        # most recent month. Skipping last month avoids short-term reversal contamination.
        # Most robust single factor in cross-sectional equity literature.
        ret_12m = ts_delta(self.ldf, "close", 252)
        ret_1m  = ts_delta(self.ldf, "close", 21)
        return ret_12m - ret_1m

    def get_alpha_101(self):
        # BAB proxy (Betting Against Beta):
        # Stocks with low rolling beta to the equal-weighted market tend to outperform
        # in risk-adjusted terms (Frazzini & Pedersen 2014). Works in VN because
        # retail-driven momentum inflates high-beta stocks.
        # Beta = cov(stock_ret, mkt_ret) / var(mkt_ret) over 60 days, using ldf returns.
        ldf = self.ldf.copy()
        ldf["daily_ret"] = ldf.groupby("Symbol")["close"].pct_change()

        # Equal-weighted market return per date (cross-sectional mean)
        mkt_ret = ldf.groupby("date")["daily_ret"].transform("mean")
        ldf["mkt_ret"] = mkt_ret

        def rolling_beta(x):
            ret   = x["daily_ret"]
            mkt   = x["mkt_ret"]
            cov   = ret.rolling(60, min_periods=20).cov(mkt)
            var   = mkt.rolling(60, min_periods=20).var()
            return cov / (var + 1e-8)

        beta = (
            ldf.groupby("Symbol")
            .apply(rolling_beta, include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )
        # BAB signal: negative beta → bet against high-beta stocks
        return -1 * beta

    def get_alpha_103(self):
        # Price Acceleration (monthly):
        # Second derivative of price at the monthly horizon.
        # ret_21d_{T} - ret_21d_{T-21}: positive = monthly momentum is accelerating.
        # Window extended from 5d to 21d so the signal persists across a full
        # monthly holding period rather than decaying within the first week.
        ret_now  = ts_delta(self.ldf, "close", 21)
        ret_prev = ts_delay(self.ldf, ret_now, 21)
        return ret_now - ret_prev
    def get_alpha_002(self):
        # -corr(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
        # Volume momentum vs intraday return rank correlation
        ldf = self.ldf.copy()
        delta_logvol = ldf.groupby("Symbol")["volume"].transform(
            lambda x: np.log(x + 1).diff(2)
        )
        intraday = (ldf["close"] - ldf["open"]) / (ldf["open"] + 0.001)
        # Cross-sectional ranks per date
        rank_dvol = delta_logvol.groupby(ldf["date"]).rank(pct=True)
        rank_intra = intraday.groupby(ldf["date"]).rank(pct=True)
        ldf["_rdv"] = rank_dvol
        ldf["_ri"] = rank_intra
        return (
            -1
            * ldf.groupby("Symbol")
            .apply(lambda x: x["_rdv"].rolling(6).corr(x["_ri"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )

    def get_alpha_007(self):
        # Conditional momentum: only fires on above-average volume days.
        # Uses 21-day window throughout for monthly-horizon consistency.
        ldf = self.ldf.copy()
        adv20 = ts_mean(ldf, "volume", 20)
        ts_r = ldf.groupby("Symbol")["close"].transform(
            lambda x: x.diff(21).abs().rolling(60, min_periods=10).rank(pct=True)
        )
        sign_delta = np.sign(ts_delta(ldf, "close", 21))
        cond = ldf["volume"] > adv20
        return np.where(cond, -1 * ts_r * sign_delta, -1.0)

    def get_alpha_013(self):
        # -rank(covariance(rank(close), rank(volume), 5))
        # Close-volume co-movement: stocks where price and volume co-move get shorted
        ldf = self.ldf.copy()
        rank_c = ldf.groupby("date")["close"].rank(pct=True)
        rank_v = ldf.groupby("date")["volume"].rank(pct=True)
        ldf["_rc"] = rank_c
        ldf["_rv"] = rank_v
        cov5 = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["_rc"].rolling(21).cov(x["_rv"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )
        return -1 * cov5.groupby(ldf["date"]).rank(pct=True)


    def get_alpha_016(self):
        # -rank(covariance(rank(high), rank(volume), 5))
        # Same idea as 013 but using high instead of close — captures breakout/blowoff tops
        ldf = self.ldf.copy()
        rank_h = ldf.groupby("date")["high"].rank(pct=True)
        rank_v = ldf.groupby("date")["volume"].rank(pct=True)
        ldf["_rh"] = rank_h
        ldf["_rv"] = rank_v
        cov5 = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["_rh"].rolling(21).cov(x["_rv"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )
        return -1 * cov5.groupby(ldf["date"]).rank(pct=True)

    
    def get_alpha_040(self):
        # -rank(stddev(high, 21)) * corr(high, volume, 21)
        # High volatility stocks that also have high-volume correlation get penalized.
        # Relevant for VN: avoids stocks in blow-off tops with volume confirmation.
        # Both windows set to 21 for monthly-horizon consistency.
        ldf = self.ldf.copy()
        std_high = ldf.groupby("Symbol")["high"].transform(
            lambda x: x.rolling(21, min_periods=5).std()
        )
        rank_std = std_high.groupby(ldf["date"]).rank(pct=True)
        corr_hv = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["high"].rolling(21).corr(x["volume"]),
                   include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
            .fillna(0)
        )
        return -1 * rank_std * corr_hv
    def get_alpha_200(self):
        # Earnings-free value proxy: book-to-market via price distance from 52-week high.
        # Stocks far below their 52w high are relatively "cheap" — captures value + distress premium.
        # Works well in VN where P/B data is often stale; price-based value is cleaner.
        high_252 = self.ldf.groupby("Symbol")["close"].transform(
            lambda x: x.rolling(252, min_periods=60).max()
        )
        return -1 * (self.ldf["close"] / (high_252 + 0.001) - 1)  # negative = far from high = cheap

    def get_alpha_201(self):
        # Volume trend confirmation: 1-month price momentum × 1-month volume momentum.
        # Rising price on rising volume = institutional accumulation. Strong in VN where
        # volume is the primary signal of informed order flow.
        ldf = self.ldf.copy()
        price_mom = ldf.groupby("Symbol")["close"].transform(
            lambda x: x.diff(21) / (x.shift(21).abs() + 0.001)
        )
        vol_mom = ldf.groupby("Symbol")["volume"].transform(
            lambda x: x.diff(21) / (x.shift(21).abs() + 0.001)
        )
        return price_mom * vol_mom

    def get_alpha_202(self):
        # Residual momentum (idiosyncratic momentum):
        # Remove the equal-weighted market return from each stock's 12-1m momentum,
        # keeping only the stock-specific component. Less crowded than raw momentum
        # and more robust through factor rotation periods.
        ldf = self.ldf.copy()
        ldf["ret"] = ldf.groupby("Symbol")["close"].pct_change()
        mkt_ret    = ldf.groupby("date")["ret"].transform("mean")
        ldf["excess_ret"] = ldf["ret"] - mkt_ret

        # Cumulative excess return over 12m skipping last 1m
        cum_12m = ldf.groupby("Symbol")["excess_ret"].transform(
            lambda x: x.rolling(252, min_periods=60).sum()
        )
        cum_1m  = ldf.groupby("Symbol")["excess_ret"].transform(
            lambda x: x.rolling(21, min_periods=10).sum()
        )
        return cum_12m - cum_1m
    def generate_all(self):
        """Computes all WQ alpha factors and returns a DataFrame of WQ_Alpha_* columns.

        Each column at row i (date T) is computed exclusively from OHLCV data on
        or before date T-1, matching the look-ahead-free contract of build_features().
        """
        print("Generating WorldQuant Alphas...")
        # --- Original alphas ---
        self.df["WQ_Alpha_001"] = self.get_alpha_001()
        self.df["WQ_Alpha_002"] = self.get_alpha_002()
        self.df["WQ_Alpha_006"] = self.get_alpha_006()
        self.df["WQ_Alpha_007"] = self.get_alpha_007()

        self.df["WQ_Alpha_013"] = self.get_alpha_013()

        self.df["WQ_Alpha_016"] = self.get_alpha_016()
        self.df["WQ_Alpha_024"] = self.get_alpha_024()
        self.df["WQ_Alpha_028"] = self.get_alpha_028()

        self.df["WQ_Alpha_040"] = self.get_alpha_040()
        self.df["WQ_Alpha_101"] = self.get_alpha_101()    # BAB: negative beta proxy
        self.df["WQ_Alpha_103"] = self.get_alpha_103()    # monthly price acceleration
        self.df["WQ_Alpha_200"] = self.get_alpha_200()
        self.df["WQ_Alpha_201"] = self.get_alpha_201()
        self.df["WQ_Alpha_202"] = self.get_alpha_202()

        # NOTE: WQ_Turnover_12m, WQ_Limit_Bias, WQ_Herding_Disp, and Amihud
        # illiquidity are structural/context features — not return signals.
        # They are computed in features.py alongside other XGBoost-only features.

        alpha_cols = [c for c in self.df.columns if c.startswith("WQ_Alpha_")]
        self.df[alpha_cols] = self.df[alpha_cols].replace([np.inf, -np.inf], np.nan)
        return self.df[alpha_cols]
    


def add_and_filter_alphas(gp_model, original_df, input_features):
    df_result = original_df.copy()
    
    # 1. Prepare input data (fill NaN temporarily with 0)
    X_full = df_result[input_features].values
    X_filled = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Generate 10 alpha expressions
    
    alpha_values = gp_model.transform(X_filled)

    _, idx = np.unique(alpha_values, axis=1, return_index=True)
    alpha_values_unique = alpha_values[:, np.sort(idx)]

    alpha_df_unique = pd.DataFrame(
    alpha_values_unique,
    columns=[f'Mega_Alpha_{i}' for i in range(alpha_values_unique.shape[1])])
    
    # 4. Append unique alpha columns to the original DataFrame
    new_cols = []
    for col in alpha_df_unique.columns:
        df_result[col] = alpha_df_unique[col].values
        new_cols.append(col)
        
    print(f"Filtered {len(new_cols)} independent Mega-Alphas from 10 candidates: {new_cols}")
    return df_result, new_cols



def rank_ic_fitness(y_true, y_pred):
    """
    Custom fitness function for GP optimisation. Returns Spearman rank IC.
    y_true: Actual forward returns (next_1m_ret)
    y_pred: Alpha signal values generated by the GP expression
    """
    # Compute Spearman rank correlation
    rho, _ = spearmanr(y_true, y_pred)
    return rho