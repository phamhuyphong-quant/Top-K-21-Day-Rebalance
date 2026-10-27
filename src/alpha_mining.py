import numpy as np
import pandas as pd

from scipy.stats import spearmanr
# --- CÁC TOÁN TỬ WORLDQUANT CƠ BẢN ---
# --- WORLDQUANT BASE OPERATORS ---
# All rolling operators use min_periods=d: a row is NaN until a full window of
# d observations exists, so warm-up rows are never filled with a partial-window value.
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
        lambda x: x.rolling(window=d, min_periods=d).mean()
    )

def ts_min(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=d).min()
    )

def ts_rank(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=d).rank(pct=True)
    )

def ts_argmax(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=d).apply(np.argmax) + 1
    )

# --- CROSS-SECTIONAL HELPERS (mask-aware, không xoá dòng) ---
def cs_mask(series, mask):
    """Đưa các dòng không đủ điều kiện (kém thanh khoản) về NaN để chúng bị
    loại khỏi thống kê cross-sectional, nhưng KHÔNG xoá dòng khỏi frame -->
    các phép rolling per-symbol phía sau vẫn thấy chuỗi thời gian liên tục."""
    if mask is None:
        return series
    return series.where(mask)

def cs_rank(series, date, mask=None):
    """Percentile rank cross-sectional theo ngày, chỉ tính trên các dòng
    đủ điều kiện (mask=True)."""
    return cs_mask(series, mask).groupby(date).rank(pct=True)

def cs_mean(series, date, mask=None):
    """Trung bình cross-sectional theo ngày, chỉ tính trên các dòng đủ điều
    kiện, rồi broadcast giá trị đó về mọi dòng (kể cả dòng không đủ điều kiện)."""
    return cs_mask(series, mask).groupby(date).transform("mean")


class WorldQuantAlphas:
    def __init__(self, df, liquidity_mask=None):
        """
        liquidity_mask: pd.Series boolean cùng index với `df`, đánh dấu dòng nào
        đủ thanh khoản (vd. adtv >= adtv_limit). Nếu None, mọi dòng được coi là
        đủ điều kiện (không lọc cross-sectional). Mask KHÔNG làm xoá dòng nào —
        nó chỉ loại các dòng kém thanh khoản khỏi các phép rank()/mean() theo
        ngày, để các phép rolling theo Symbol vẫn chạy trên chuỗi liên tục.
        """
        self.df = df.copy()
        self.df["_is_liquid"] = (
            liquidity_mask.reindex(self.df.index) if liquidity_mask is not None else True
        )
        self.df = self.df.sort_values(by=["Symbol", "date"])

        # Build a lagged view of all OHLCV columns so that every alpha method
        # at row i (date T) only ever sees data from T-1 and earlier.
        # All ts_* operators applied to self.ldf are therefore look-ahead-free
        # without any additional shifting inside the individual alpha methods.
        ohlcv = ["open", "high", "low", "close", "volume"]
        self.ldf = self.df.copy()
        for col in ohlcv:
            self.ldf[col] = self.df.groupby("Symbol")[col].shift(1)

    # ------------------------------------------------------------------
    # Helpers (methods, so alpha functions can call self._xxx)
    # ------------------------------------------------------------------
    def _dlogret(self):
        """Daily log return of the lagged close."""
        return np.log(self.ldf["close"]).groupby(self.ldf["Symbol"]).diff()

    def _roll(self, s, w, fn, minp=None):
        """Per-symbol rolling statistic. minp defaults to w (full window)."""
        return s.groupby(self.ldf["Symbol"]).transform(
            lambda x: getattr(x.rolling(w, min_periods=minp or w), fn)()
        )

    def _cs(self, s):
        """Cross-sectional percentile rank per date, liquid stocks only."""
        return cs_rank(s, self.ldf["date"], self.ldf["_is_liquid"])

    # ------------------------------------------------------------------
    # Alphas
    # ------------------------------------------------------------------
    def get_alpha_006(self):
        # Alpha#6: -corr(open, volume, 21)
        # ldf already has open_{T-1} and volume_{T-1}, so the 21-day rolling
        # correlation at row T uses days T-1 ... T-21 -- no leakage.
        # Warm-up rows (first ~21 per symbol) stay NaN on purpose.
        g = self.ldf.groupby("Symbol")
        return (
            -1
            * g.apply(lambda x: x["open"].rolling(21).corr(x["volume"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )

    def get_alpha_019(self):
        # -sign(1m ret) * (1 + rank(12m ret)); original used 7d.
        ldf, c = self.ldf, self.ldf["close"]
        r1  = np.log(c / ts_delay(ldf, c, 21))
        r12 = np.log(c / ts_delay(ldf, c, 252))
        return -np.sign(r1) * (1 + self._cs(r12))

    def get_alpha_024(self):
        # Alpha#24: conditional mean-reversion on the 100-day price trend.
        # If the 100-day mean has not risen more than 5% over the last 100 days
        # (weak/flat trend), bet against the distance above the 100-day low;
        # otherwise bet against the 21-day return (short-term reversal).
        # Both branches are log ratios (return units), not VND price differences,
        # so the cross-sectional rank reflects the signal, not share-price level.
        # All inputs come from ldf (lagged 1 day), so no look-ahead.
        # Rows where the 100-day trend is not yet defined (warm-up) stay NaN
        # instead of silently falling into the else-branch.
        # NOTE: assumes close is adjusted for splits/bonus shares.
        c = self.ldf["close"]
        mean_100   = ts_mean(self.ldf, "close", 100)
        delta_mean = ts_delta(self.ldf, mean_100, 100)
        cond = (delta_mean / (ts_delay(self.ldf, "close", 100) + 0.001)) <= 0.05

        dist_low = np.log(c / ts_min(self.ldf, "close", 100))
        ret_1m   = np.log(c / ts_delay(self.ldf, "close", 21))

        out = np.where(cond, -dist_low, -ret_1m)
        return pd.Series(out, index=self.ldf.index).where(delta_mean.notna())

    def get_alpha_028(self):
        # Alpha#28: corr(adv20, low, 21) + (high + low)/2 - close, made scale-consistent.
        # Term 1: 21-day rolling correlation of 20-day avg volume with low (unit-free).
        # Term 2: log(midpoint / close), the relative intraday gap, in return units
        #         instead of VND so it does not sort stocks by share price.
        # Each term is cross-sectionally ranked per date (liquid stocks only) before
        # summing, so neither dominates the other.
        # All inputs come from ldf (lagged 1 day), so no look-ahead.
        # Warm-up rows stay NaN (no fillna(0)).
        ldf = self.ldf.copy()
        ldf["adv20"] = ts_mean(ldf, "volume", 20)
        corr = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["adv20"].rolling(21).corr(x["low"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )
        gap = np.log(((ldf["high"] + ldf["low"]) / 2) / ldf["close"])
        return (
            cs_rank(corr, ldf["date"], ldf["_is_liquid"])
            + cs_rank(gap, ldf["date"], ldf["_is_liquid"])
        )

    def get_alpha_039(self):
        # -rank(ret * (1 - rank(rel volume))) * (1 + rank(12m ret)); decay_linear -> mean.
        ldf, c, v = self.ldf, self.ldf["close"], self.ldf["volume"]
        ret = np.log(c / ts_delay(ldf, c, 21))
        rv  = self._roll(v / (self._roll(v, 20, "mean", 10) + 1), 21, "mean", 10)
        r12 = np.log(c / ts_delay(ldf, c, 252))
        return -self._cs(ret * (1 - self._cs(rv))) * (1 + self._cs(r12))

    def get_alpha_052(self):
        # (prior-month low vs current-month low) * rank(12-1m mom) * ts_rank(volume); 5d -> 21d.
        ldf, c = self.ldf, self.ldf["close"]
        mn   = self._roll(ldf["low"], 21, "min")
        move = np.log(ts_delay(ldf, mn, 21) / mn)
        mom  = np.log(ts_delay(ldf, c, 21) / ts_delay(ldf, c, 252))
        return move * self._cs(mom) * ts_rank(ldf, ldf["volume"], 21)

    def get_alpha_035(self):
        # ts_rank(volume,32) * (1-ts_rank(c+h-l,16)) * (1-ts_rank(ret,32)); ts_rank is scale-free.
        ldf = self.ldf
        return (ts_rank(ldf, ldf["volume"], 32)
                * (1 - ts_rank(ldf, ldf["close"] + ldf["high"] - ldf["low"], 16))
                * (1 - ts_rank(ldf, self._dlogret(), 32)))

    def get_alpha_043(self):
        # ts_rank(vol/adv20, 20) * ts_rank(-delta(close,7), 8); stretched to 63 / 21 / 63.
        ldf, c, v = self.ldf, self.ldf["close"], self.ldf["volume"]
        rel_vol = v / (ts_mean(ldf, v, 20) + 1)
        return ts_rank(ldf, rel_vol, 63) * ts_rank(ldf, -ts_delta(ldf, c, 21), 63)

    def get_alpha_055(self):
        # -corr(rank(stochastic %K), rank(volume)); 12d/6d -> 63d/21d.
        ldf = self.ldf.copy()
        lo = self._roll(ldf["low"], 63, "min", 30)
        hi = self._roll(ldf["high"], 63, "max", 30)
        ldf["_k"]  = self._cs((ldf["close"] - lo) / (hi - lo + 1e-8))
        ldf["_rv"] = self._cs(ldf["volume"])
        return -1 * (
            ldf.groupby("Symbol")
            .apply(lambda x: x["_k"].rolling(21).corr(x["_rv"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )

    def get_alpha_105(self):
        # 12-1 month momentum: log return from T-253 to T-22 (past ~12 months,
        # skipping the most recent month to avoid short-term reversal contamination).
        # Measured in return units, not price units, so the cross-sectional rank
        # reflects momentum rather than share price level.
        # ldf is already lagged by 1 day, so shifts of 21 and 252 on it give
        # close_{T-22} and close_{T-253}; no look-ahead.
        # Warm-up rows (first ~253 per symbol) stay NaN on purpose.
        # NOTE: assumes close is adjusted for splits/bonus shares; otherwise a
        # corporate action inside the window shows up as a fake crash.
        c = self.ldf["close"]
        return np.log(ts_delay(self.ldf, c, 21) / ts_delay(self.ldf, c, 252))

    def get_alpha_104(self):
        # BAB proxy (Betting Against Beta):
        # Stocks with low rolling beta to the equal-weighted market tend to outperform
        # in risk-adjusted terms (Frazzini & Pedersen 2014). Works in VN because
        # retail-driven momentum inflates high-beta stocks.
        # Beta = cov(stock_ret, mkt_ret) / var(mkt_ret) over 60 days, using ldf returns.
        #
        # mkt_ret is a cross-sectional (per-date) mean, computed only over liquid
        # stocks via cs_mean(). The rolling(60) beta itself still runs per-Symbol
        # on the FULL, continuous ldf — liquidity filtering never removes rows,
        # so this rolling window is never distorted by gaps.
        ldf = self.ldf.copy()
        ldf["daily_ret"] = ldf.groupby("Symbol")["close"].pct_change(fill_method=None)
        ldf["mkt_ret"] = cs_mean(ldf["daily_ret"], ldf["date"], ldf["_is_liquid"])

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

    def get_alpha_102(self):
        # Price acceleration (monthly): change in 21-day log return,
        #   log_ret(T-22 -> T-1) - log_ret(T-43 -> T-22).
        # Positive = monthly momentum is accelerating.
        # Equivalent to log(c_{T-1}) - 2*log(c_{T-22}) + log(c_{T-43}), computed on
        # log prices so the signal is in return units, not share-price units.
        # ldf is already lagged by 1 day, so shifts of 21 and 42 give close_{T-22}
        # and close_{T-43}; no look-ahead.
        # Warm-up rows (first ~43 per symbol) stay NaN on purpose.
        # NOTE: assumes close is adjusted for splits/bonus shares; a corporate
        # action inside the 43-day window shows up as a fake acceleration.
        lc = np.log(self.ldf["close"])
        return lc - 2 * ts_delay(self.ldf, lc, 21) + ts_delay(self.ldf, lc, 42)

    def get_alpha_002(self):
        # -corr(rank(delta(log(volume), 7)), rank((close-open)/open), 21)
        # Volume momentum vs intraday return rank correlation.
        # Ranks are cross-sectional (per-date), over liquid stocks only.
        # Warm-up rows (first ~28 per symbol) stay NaN on purpose.
        ldf = self.ldf.copy()
        delta_logvol = ldf.groupby("Symbol")["volume"].transform(
            lambda x: np.log(x + 1).diff(7)
        )
        intraday = (ldf["close"] - ldf["open"]) / (ldf["open"] + 0.001)
        ldf["_rdv"] = cs_rank(delta_logvol, ldf["date"], ldf["_is_liquid"])
        ldf["_ri"]  = cs_rank(intraday,     ldf["date"], ldf["_is_liquid"])
        return (
            -1
            * ldf.groupby("Symbol")
            .apply(lambda x: x["_rdv"].rolling(21).corr(x["_ri"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )

    def get_alpha_007(self):
        # Volume-conditioned reversal: on above-average-volume days, bet against the
        # 21-day log return, scaled by how extreme |return| is vs its own last 60 days.
        # Below-average-volume days get 0.0 (neutral, middle of the [-1, 1] range),
        # not -1.0, which was the minimum and tied half the universe at the bottom.
        # Do not use NaN there: build_features drops rows with any NaN feature and
        # would delete about half of all rows.
        # Warm-up rows (ts_r undefined) stay NaN.
        ldf = self.ldf.copy()
        adv20 = ts_mean(ldf, "volume", 20)
        ret21 = np.log(ldf["close"] / ts_delay(ldf, "close", 21))
        ts_r = ret21.abs().groupby(ldf["Symbol"]).transform(
            lambda x: x.rolling(60, min_periods=10).rank(pct=True)
        )
        out = np.where(ldf["volume"] > adv20, -ts_r * np.sign(ret21), 0.0)
        return pd.Series(out, index=ldf.index).where(ts_r.notna())

    def get_alpha_013(self):
        # -rank(covariance(rank(rel_close), rank(rel_vol), 21))
        # Close-volume co-movement: stocks where price and volume co-move get shorted.
        # Deviates from WQ Alpha#13 on purpose: the inner ranks are over
        #   rel_close = log(close / 20d mean close)
        #   rel_vol   = log(volume / 20d mean volume)
        # instead of raw close/volume, so the ranks compare relative behavior and
        # do not sort stocks by share price (VND) or tick-size density.
        # All ranks are cross-sectional (per-date), over liquid stocks only.
        # Warm-up rows stay NaN on purpose.
        ldf = self.ldf.copy()
        rel_close = np.log(ldf["close"] / ts_mean(ldf, "close", 20))
        rel_vol   = np.log((ldf["volume"] + 1) / (ts_mean(ldf, "volume", 20) + 1))
        ldf["_rc"] = cs_rank(rel_close, ldf["date"], ldf["_is_liquid"])
        ldf["_rv"] = cs_rank(rel_vol,   ldf["date"], ldf["_is_liquid"])
        cov21 = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["_rc"].rolling(21).cov(x["_rv"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )
        return -1 * cs_rank(cov21, ldf["date"], ldf["_is_liquid"])

    def get_alpha_016(self):
        # -rank(covariance(rank(rel_high), rank(rel_vol), 21))
        # Same idea as 013 but using high -- captures breakout/blowoff tops.
        # Deviates from WQ Alpha#16 on purpose: inner ranks are over
        #   rel_high = log(high / 20d mean close)
        #   rel_vol  = log(volume / 20d mean volume)
        # instead of raw high/volume, to make the signal price-scale invariant.
        # Warm-up rows stay NaN on purpose.
        ldf = self.ldf.copy()
        rel_high = np.log(ldf["high"] / ts_mean(ldf, "close", 20))
        rel_vol  = np.log((ldf["volume"] + 1) / (ts_mean(ldf, "volume", 20) + 1))
        ldf["_rh"] = cs_rank(rel_high, ldf["date"], ldf["_is_liquid"])
        ldf["_rv"] = cs_rank(rel_vol,  ldf["date"], ldf["_is_liquid"])
        cov21 = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["_rh"].rolling(21).cov(x["_rv"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )
        return -1 * cs_rank(cov21, ldf["date"], ldf["_is_liquid"])

    def get_alpha_040(self):
        # -rank(stddev(log(high), 21)) * corr(high, volume, 21)
        # High-volatility stocks whose high moves with volume get penalized.
        # Relevant for VN: avoids blow-off tops with volume confirmation.
        # Warm-up rows (first ~21 per symbol) stay NaN on purpose.
        ldf = self.ldf.copy()
        std_high = ldf.groupby("Symbol")["high"].transform(
            lambda x: np.log(x).rolling(21, min_periods=5).std()
        )
        rank_std = cs_rank(std_high, ldf["date"], ldf["_is_liquid"])
        corr_hv = (
            ldf.groupby("Symbol")
            .apply(lambda x: x["high"].rolling(21).corr(x["volume"]), include_groups=False)
            .reset_index(level=0, drop=True)
            .reindex(self.df.index)
        )
        return -1 * rank_std * corr_hv

    def get_alpha_103(self):
        # 52-week-high proximity (George & Hwang): close / max(close, 252d) - 1.
        # 0 = at the high, negative = far below. A momentum-type signal, NOT value.
        # Ratio, so it is scale-free. min_periods=60 means early rows use a shorter window.
        high_252 = self.ldf.groupby("Symbol")["close"].transform(
            lambda x: x.rolling(252, min_periods=60).max()
        )
        return (self.ldf["close"] / (high_252 + 0.001) - 1)  # 0 = at the 252d high, negative = far below it

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
        #
        # mkt_ret is a cross-sectional (per-date) mean over liquid stocks only
        # (cs_mean). The rolling(252)/(21) sums still run per-Symbol on the full,
        # continuous ldf — unaffected by liquidity filtering.
        ldf = self.ldf.copy()
        ldf["ret"] = ldf.groupby("Symbol")["close"].pct_change(fill_method=None)
        ldf["mkt_ret"] = cs_mean(ldf["ret"], ldf["date"], ldf["_is_liquid"])
        ldf["excess_ret"] = ldf["ret"] - ldf["mkt_ret"]

        # Cumulative excess return over 12m skipping last 1m
        cum_12m = ldf.groupby("Symbol")["excess_ret"].transform(
            lambda x: x.rolling(252, min_periods=60).sum()
        )
        cum_1m  = ldf.groupby("Symbol")["excess_ret"].transform(
            lambda x: x.rolling(21, min_periods=10).sum()
        )
        return cum_12m - cum_1m

    def get_alpha_110(self):
        # MAX effect (lottery demand): -max daily return over 21d. Strong in retail-heavy markets.
        return -1 * self._roll(self._dlogret(), 21, "max")

    def get_alpha_111(self):
        # Negative 63d return skewness (lottery / crash-risk proxy).
        return -1 * self._roll(self._dlogret(), 63, "skew")

    def get_alpha_112(self):
        # Intermediate momentum (Novy-Marx): log return T-127 -> T-253, i.e. months 7-12.
        c = self.ldf["close"]
        return np.log(ts_delay(self.ldf, c, 126) / ts_delay(self.ldf, c, 252))

    def get_alpha_113(self):
        # Risk-adjusted 6-1m momentum: return (T-22 -> T-127) divided by 126d daily vol.
        c = self.ldf["close"]
        ret = np.log(ts_delay(self.ldf, c, 21) / ts_delay(self.ldf, c, 126))
        vol = self._roll(self._dlogret(), 126, "std", minp=63)
        return ret / (vol * np.sqrt(105) + 1e-8)

    def get_alpha_114(self):
        # Information discreteness (Da, Gurun, Warachka): smooth winners beat jumpy ones.
        # signal = -sign(12m ret) * (%down days - %up days) over 252d.
        r = self._dlogret()
        neg = (r < 0).astype(float).where(r.notna())
        pos = (r > 0).astype(float).where(r.notna())
        pct_neg = self._roll(neg, 252, "mean", minp=126)
        pct_pos = self._roll(pos, 252, "mean", minp=126)
        c = self.ldf["close"]
        sgn = np.sign(np.log(c / ts_delay(self.ldf, c, 252)))
        return -1 * sgn * (pct_neg - pct_pos)

    def get_alpha_115(self):
        # Overnight drift: mean log(open_t / close_{t-1}) over 21d (institutional vs retail flow).
        ov = np.log(self.ldf["open"] / self.ldf.groupby("Symbol")["close"].shift(1))
        return self._roll(ov, 21, "mean", minp=10)

    def get_alpha_116(self):
        # Close-location value: mean of ((c-l)-(h-c))/(h-l) over 21d = persistent buying pressure.
        h, l, c = self.ldf["high"], self.ldf["low"], self.ldf["close"]
        clv = ((c - l) - (h - c)) / (h - l + 1e-8)
        return self._roll(clv, 21, "mean", minp=10)

    def get_alpha_117(self):
        # Abnormal volume (attention): log(adv21 / adv252).
        v = self.ldf["volume"]
        a21  = self._roll(v, 21, "mean", minp=10)
        a252 = self._roll(v, 252, "mean", minp=126)
        return np.log((a21 + 1) / (a252 + 1))

    def get_alpha_118(self):
        # Low-vol anomaly via Parkinson range volatility (21d); less noisy than close-to-close.
        hl2 = np.log(self.ldf["high"] / self.ldf["low"]) ** 2
        return -1 * np.sqrt(self._roll(hl2, 21, "mean", minp=10) / (4 * np.log(2)))

    def generate_all(self):
        """Computes all WQ alpha factors and returns a DataFrame of WQ_Alpha_* columns.

        Each column at row i (date T) is computed exclusively from OHLCV data on
        or before date T-1, matching the look-ahead-free contract of build_features().

        Cross-sectional pieces (ranks/means computed per date) only aggregate over
        rows flagged liquid in `liquidity_mask` (see cs_rank/cs_mean), while every
        rolling/lag computation per Symbol still runs on the full, continuous
        row history — liquidity filtering never removes rows before this point.
        """
        print("Generating WorldQuant Alphas...")
        # --- Original alphas ---
        self.df["WQ_Alpha_002"] = self.get_alpha_002()
        self.df["WQ_Alpha_006"] = self.get_alpha_006()
        self.df["WQ_Alpha_007"] = self.get_alpha_007()    # volume-conditioned reversal (log-return based)

        self.df["WQ_Alpha_013"] = self.get_alpha_013()    # rank(rel close/volume) cov, price-scale invariant
        self.df["WQ_Alpha_016"] = self.get_alpha_016()    # rank(rel high/volume) cov, price-scale invariant
        self.df["WQ_Alpha_024"] = self.get_alpha_024()    # conditional reversal (log-ratio based)
        self.df["WQ_Alpha_028"] = self.get_alpha_028()    # rank(vol-low corr) + rank(log midpoint/close gap)

        self.df["WQ_Alpha_040"] = self.get_alpha_040()

        self.df["WQ_Alpha_104"] = self.get_alpha_104()    # BAB: negative beta proxy
        self.df["WQ_Alpha_102"] = self.get_alpha_102()    # monthly price acceleration (log-return based)
        self.df["WQ_Alpha_103"] = self.get_alpha_103()    # 52w-high proximity (momentum-type)
        self.df["WQ_Alpha_105"] = self.get_alpha_105()    # 12-1m momentum (log return, T-253 to T-22)

        # --- Slow-moving additions (monthly horizon) ---
        for n in (110, 111, 112, 113, 114, 115, 116, 117, 118):
            self.df[f"WQ_Alpha_{n:03d}"] = getattr(self, f"get_alpha_{n:03d}")()
        # --- Alpha101 adaptations (windows stretched to monthly) ---
        for n in (19, 35, 39, 43, 52, 55):
            self.df[f"WQ_Alpha_{n:03d}"] = getattr(self, f"get_alpha_{n:03d}")()
        # self.df["WQ_Alpha_201"] = self.get_alpha_201()
        # self.df["WQ_Alpha_202"] = self.get_alpha_202()

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

    # 2. Generate alpha expressions
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