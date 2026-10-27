import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _rolling_rank_ic(factor_series: pd.Series, ret_series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling RankIC (Spearman correlation) between one factor and forward
    returns, grouped by date (cross-sectional), then rolled over `window` periods.

    At each date t, RankIC_t = Spearman(factor values across stocks, returns across stocks).
    The rolling mean is then taken over the past `window` dates to get a smoothed IC.

    Returns a Series indexed by date.
    """
    # Cross-sectional Spearman per date
    df = pd.DataFrame({"factor": factor_series, "ret": ret_series})
    daily_ic = (
        df.groupby(df.index)
        .apply(lambda x: spearmanr(x["factor"], x["ret"])[0] if len(x) > 2 else np.nan,
               include_groups=False)
    )
    rolling_ic   = daily_ic.rolling(window, min_periods=window // 2).mean()
    rolling_icir = rolling_ic / (daily_ic.rolling(window, min_periods=window // 2).std() + 1e-8)
    return rolling_ic, rolling_icir


class AlphaForgeCombiner:
    """
    Faithful implementation of AlphaForge's Algorithm 2 (factor combination model).

    Usage per walk-forward fold:
      1. combiner.precompute(all_df, alpha_cols, ret_col)   # once per fold
      2. combiner.fit_and_predict(current_df, last_date)    # once per test date

    At each rebalance date t, given a factor zoo Z = {f1, ..., fk}:
      1. Read rolling RankIC and ICIR for each factor over the past `ic_window`
         dates (up to `last_date`, the last date whose label is fully realized).
      2. Gate: drop factors where |RankIC| < ic_threshold OR |ICIR| < icir_threshold.
      3. Sort survivors by |ICIR|, keep top `max_active_factors`.
      4. Fit Ridge regression of those factors against recent returns → dynamic weights.
      5. Apply weights to current-date factor values → Mega-Alpha scalar per stock.

    This replaces the LSTM-attention combiner entirely. The paper specifically chose
    a linear model here for two reasons:
      - Interpretability: weights are directly readable
      - Overfitting resistance: nonlinear combiners overfit on financial noise

    Parameters
    ----------
    ic_window : int or None
        Number of past dates used for the rolling RankIC/ICIR and for the Ridge fit.
        None = expanding window (all history up to last_date), as in the authors' default.
    ic_threshold : float
        Minimum |RankIC| for a factor to be considered active. The authors'
        reference code uses 0.02.
    icir_threshold : float
        Minimum |ICIR| for a factor to be considered active. The authors'
        reference code uses 0.2.
    max_active_factors : int
        Top-N active factors to use in the linear combination. Paper found N=10 optimal.
    ridge_alpha : float
        Regularisation strength for the Ridge regression combiner.
    """

    def __init__(
        self,
        ic_window: int | None = 40,   # None = expanding (all available history)
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.2,
        max_active_factors: int = 10, 
        ridge_alpha: float = 1.0,
    ):
        self.ic_window          = ic_window
        self.ic_threshold       = ic_threshold
        self.icir_threshold     = icir_threshold
        self.max_active_factors = max_active_factors
        self.ridge_alpha        = ridge_alpha

        # State updated at each call to fit_and_predict()
        self.active_factors_   = []   # names of factors selected at last rebalance
        self.weights_          = {}   # {factor_name: coefficient}
        self.factor_ic_history_= {}   # {factor_name: (rolling_ic, rolling_icir)} Series

        # Cache filled by precompute()
        self._alpha_cols = None
        self._ret_col    = None
        self._ranked     = None       # per-date percentile ranks of factors + label
        self._daily_ic   = None       # dates x factors rank-IC table

    # ------------------------------------------------------------------
    # Once per fold: rank within each date and cache the daily rank-IC
    # ------------------------------------------------------------------
    def precompute(self, all_df: pd.DataFrame, alpha_cols: list, ret_col: str):
        """Call once per fold. Ranks factors and label within each date and caches
        the per-date rank-IC table (dates x factors)."""
        cols = alpha_cols + [ret_col]
        d = all_df.dropna(subset=cols)
        r = d.groupby("date")[cols].rank(pct=True)
        r["date"] = d["date"]
        self._alpha_cols, self._ret_col = alpha_cols, ret_col
        self._ranked = r
        self._daily_ic = r.groupby("date").apply(
            lambda x: x[alpha_cols].corrwith(x[ret_col]), include_groups=False
        )

    # ------------------------------------------------------------------
    # Core: select factors and fit weights from the cache, predict current date
    # ------------------------------------------------------------------
    def fit_and_predict(self, current_df: pd.DataFrame, last_date) -> pd.Series:
        """
        current_df : rows of the date being predicted.
        last_date  : last date whose label is fully realized (no later date is used).
        """
        alpha_cols, ret_col, w = self._alpha_cols, self._ret_col, self.ic_window

        # IC/ICIR from the cache; only dates <= last_date (their labels are realized)
        ic  = self._daily_ic.loc[:last_date]
        ic  = ic if w is None else ic.tail(w)
        win = len(ic) if w is None else w
        ok  = ic.count() >= max(1, win // 2)
        mu, sd = ic.mean().where(ok), ic.std().where(ok)
        ic_records   = mu.fillna(0).to_dict()
        icir_records = (mu / (sd + 1e-8)).fillna(0).to_dict()

        # --- Step 2: gate — keep only factors meeting both thresholds ---
        survivors = [
            f for f in alpha_cols
            if abs(ic_records.get(f, 0)) >= self.ic_threshold
            and abs(icir_records.get(f, 0)) >= self.icir_threshold
        ]

        # --- Step 3: top-N by |ICIR| ---
        survivors_sorted = sorted(survivors, key=lambda f: abs(icir_records[f]), reverse=True)
        active = survivors_sorted[: self.max_active_factors]
        self.active_factors_ = active

        if len(active) == 0:
            # No factor passed the gate: fall back to the single most stable factor
            # (highest |ICIR|), as in the authors' reference code, instead of
            # blending all factors, which would defeat the purpose of the gate.
            active = [max(alpha_cols, key=lambda f: abs(icir_records.get(f, 0)))]
            self.active_factors_ = active

        # --- Step 4: fit Ridge on the cached cross-sectional ranks of the same window ---
        r = self._ranked[self._ranked["date"].isin(ic.index)]

        if r.empty:
            self.weights_ = {f: 1.0 / len(active) for f in active}
        else:
            X_fit = r[active].values - 0.5          # per-date ranks, centered at 0
            y_fit = r[ret_col].values - 0.5

            sign_vector = np.array([np.sign(ic_records[f]) if ic_records[f] != 0 else 1.0 for f in active])
            X_fit = X_fit * sign_vector

            reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            reg.fit(X_fit, y_fit)
            self.weights_ = {f: coef * sign_vector[i] for i, (f, coef) in enumerate(zip(active, reg.coef_))}

        # --- Step 5: predict on current date (same ranking) ---
        cur = current_df.copy()
        cur_rank = cur[list(self.weights_)].rank(pct=True) - 0.5
        mega_alpha = pd.Series(0.0, index=cur.index)
        for f, wt in self.weights_.items():
            mega_alpha += wt * cur_rank[f].fillna(0)
        return mega_alpha

    # ------------------------------------------------------------------
    # Convenience: print which factors are active and their weights
    # ------------------------------------------------------------------
    def report(self):
        print(f"Active factors ({len(self.active_factors_)}):")
        for f, wt in sorted(self.weights_.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {f:30s}  weight={wt:+.6f}")