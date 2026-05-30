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

    At each rebalance date t, given a factor zoo Z = {f1, ..., fk}:
      1. Compute rolling RankIC and ICIR for each factor over the past `ic_window` periods.
      2. Gate: drop factors where |RankIC| < ic_threshold OR |ICIR| < icir_threshold.
      3. Sort survivors by |RankIC|, keep top `max_active_factors`.
      4. Fit Ridge regression of those factors against recent returns → dynamic weights.
      5. Apply weights to current-date factor values → Mega-Alpha scalar per stock.

    This replaces the LSTM-attention combiner entirely. The paper specifically chose
    a linear model here for two reasons:
      - Interpretability: weights are directly readable
      - Overfitting resistance: nonlinear combiners overfit on financial noise

    Parameters
    ----------
    ic_window : int
        Number of past periods (dates) used to compute rolling RankIC. Default 12
        (roughly 12 months when called monthly).
    ic_threshold : float
        Minimum |RankIC| for a factor to be considered active. Paper uses ~0.02.
    icir_threshold : float
        Minimum |ICIR| for a factor to be considered active. Paper uses ~0.3.
    max_active_factors : int
        Top-N active factors to use in the linear combination. Paper found N=10 optimal.
    ridge_alpha : float
        Regularisation strength for the Ridge regression combiner.
    """

    def __init__(
        self,
        ic_window: int = 40,
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.2,
        max_active_factors: int = 13,   # matches your current zoo size
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

    # ------------------------------------------------------------------
    # Core: fit on history, predict on current date
    # ------------------------------------------------------------------
    def fit_and_predict(
        self,
        hist_df: pd.DataFrame,
        current_df: pd.DataFrame,
        alpha_cols: list,
        ret_col: str = "next_1m_ret",
    ) -> pd.Series:
        
        # 1. TRUNCATE HISTORY IMMEDIATELY
        # We only need enough history for rolling IC (ic_window) + a small buffer.
        # This prevents the loop from slowing down as the walk-forward fold progresses.
        dates = hist_df["date"].sort_values().unique()
        required_dates = dates[-(self.ic_window * 2):] 
        hist = hist_df[hist_df["date"].isin(required_dates)].copy().dropna(subset=alpha_cols + [ret_col])
        
        # 2. VECTORIZED SPEARMAN RANK CORRELATION
        # Rank the features and target cross-sectionally per date
        ranked_hist = hist.groupby("date")[alpha_cols + [ret_col]].rank(pct=True)
        ranked_hist["date"] = hist["date"] # Restore grouping key
        
        # Compute daily Pearson correlation on the ranks (which equals Spearman)
        daily_ic = ranked_hist.groupby("date").apply(
            lambda x: x[alpha_cols].corrwith(x[ret_col]), 
            include_groups=False
        )

        # 3. CALCULATE ROLLING METRICS
        roll_ic = daily_ic.rolling(self.ic_window, min_periods=self.ic_window // 2).mean()
        roll_icir = roll_ic / (daily_ic.rolling(self.ic_window, min_periods=self.ic_window // 2).std() + 1e-8)

        # Extract the most recent valid values
        ic_records = roll_ic.iloc[-1].fillna(0).to_dict() if not roll_ic.empty else {f: 0.0 for f in alpha_cols}
        icir_records = roll_icir.iloc[-1].fillna(0).to_dict() if not roll_icir.empty else {f: 0.0 for f in alpha_cols}

        # --- Step 2: gate — keep only factors meeting both thresholds ---
        survivors = [
            f for f in alpha_cols
            if abs(ic_records.get(f, 0)) >= self.ic_threshold
            and abs(icir_records.get(f, 0)) >= self.icir_threshold
        ]

        # --- Step 3: top-N by |RankIC| ---
        survivors_sorted = sorted(survivors, key=lambda f: abs(ic_records[f]), reverse=True)
        active = survivors_sorted[: self.max_active_factors]
        self.active_factors_ = active

        if len(active) == 0:
            active = alpha_cols
            self.active_factors_ = active

        # --- Step 4: fit Ridge on recent history ---
        recent_dates = dates[-self.ic_window:]
        recent = hist[hist["date"].isin(recent_dates)].dropna(subset=active + [ret_col])

        if recent.empty:
            # Fallback if history is completely missing
            self.weights_ = {f: 1.0 / len(active) for f in active}
        else:
            X_fit = recent[active].values
            y_fit = recent[ret_col].values

            # Sign-flip logic
            sign_vector = np.array([np.sign(ic_records[f]) if ic_records[f] != 0 else 1.0 for f in active])
            X_fit = X_fit * sign_vector

            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            reg.fit(X_fit, y_fit)

            self.weights_ = {f: coef * sign_vector[i] for i, (f, coef) in enumerate(zip(active, reg.coef_))}

        # --- Step 5: predict on current date ---
        cur = current_df.copy()
        mega_alpha = pd.Series(0.0, index=cur.index)
        for f, w in self.weights_.items():
            mega_alpha += w * cur[f].fillna(0)

        return mega_alpha
   
    # ------------------------------------------------------------------
    # Convenience: print which factors are active and their weights
    # ------------------------------------------------------------------
    def report(self):
        print(f"Active factors ({len(self.active_factors_)}):")
        for f, w in sorted(self.weights_.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {f:30s}  weight={w:+.6f}")