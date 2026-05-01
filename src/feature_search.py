"""
feature_search.py
-----------------
Searches every combination of feature *groups* (as defined in features.py)
to find which combination produces the highest final ROI when evaluated via
walk_forward_cv + run_xgboost_backtest.

Feature groups
--------------
  G0 - Returns        : log_ret_1w, log_ret_1m, log_ret_3m, log_ret_6m, log_ret_1y
  G1 - Volatility     : volatility_1w/1m/3m/6m, volatility_shock_monthly/weekly, log_ret_daily
  G2 - Moving Average : SMA/EMA 14/20/50/100 + dist_SMA_100/14/50
  G3 - Volume         : vol_5d/21d/3m_avg, volume_surge_monthly/weekly
  G4 - RSI            : RSI_14

Why groups and not individual features?
  29 individual features → 2^29 ≈ 500 M combinations (infeasible).
  5 groups → 2^5 - 1 = 31 combinations (very fast).
  Each group represents a coherent information source, so group-level
  ablation is also the most interpretable result.

Usage
-----
    from src.features import build_features, target_generating_ranking
    from feature_search import find_best_feature_combo

    df_raw = ...  # your raw OHLCV dataframe
    df = build_features(df_raw)
    df = target_generating_ranking(df, freq='M')

    best_features, best_roi, results_df = find_best_feature_combo(
        df,
        initial_capital=10_000,
        walk_forward_kwargs=dict(initial_train_months=24, test_months=6, gap_days=21),
        backtest_kwargs=dict(buy_fraction=0.05, hold_fraction=0.15,
                             trend_filter_col='dist_SMA_100'),
        min_groups=1,          # minimum number of groups in a combo
        verbose=True,
    )
"""

import itertools
import traceback
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Feature group definitions (mirrors the functions in features.py)
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "returns": [
        "log_ret_1w",
        "log_ret_1m",
        "log_ret_3m",
        "log_ret_6m",
        "log_ret_1y",
    ],
    "volatility": [
        "log_ret_daily",
        "volatility_1w",
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_shock_monthly",
        "volatility_shock_weekly",
    ],
    "moving_average": [
        "SMA_14", "SMA_20", "SMA_50", "SMA_100",
        "EMA_14", "EMA_20", "EMA_50", "EMA_100",
        "dist_SMA_100", "dist_SMA_14", "dist_SMA_50",
    ],
    "volume": [
        "vol_5d_avg",
        "vol_21d_avg",
        "vol_3m_avg",
        "volume_surge_monthly",
        "volume_surge_weekly",
    ],
    "rsi": [
        "RSI_14",
    ],
}

ALL_GROUP_NAMES: list[str] = list(FEATURE_GROUPS.keys())


# ---------------------------------------------------------------------------
# Helper: flatten a list of group names → feature column list
# ---------------------------------------------------------------------------
def groups_to_features(group_names: list[str]) -> list[str]:
    features: list[str] = []
    for g in group_names:
        features.extend(FEATURE_GROUPS[g])
    return features


# ---------------------------------------------------------------------------
# Helper: compute final ROI from a backtest history DataFrame
# ---------------------------------------------------------------------------
def _final_roi(history: pd.DataFrame) -> float:
    """
    Returns the final ROI (%) from a `run_xgboost_backtest` result.

    ROI = (final_value / initial_value - 1) * 100
    """
    if history is None or history.empty:
        return float("-inf")
    initial = history["total_value"].iloc[0]
    final   = history["total_value"].iloc[-1]
    if initial == 0:
        return float("-inf")
    return (final / initial - 1) * 100.0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def find_best_feature_combo(
    df: pd.DataFrame,
    *,
    initial_capital: float = 10_000,
    walk_forward_kwargs: Optional[dict] = None,
    backtest_kwargs: Optional[dict] = None,
    model_params: Optional[dict] = None,
    min_groups: int = 1,
    verbose: bool = True,
) -> tuple[list[str], float, pd.DataFrame]:
    """
    Try every non-empty combination of feature groups, run walk_forward_cv and
    run_xgboost_backtest for each, and return the combination that achieves the
    highest final ROI.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame produced by build_features() and
        target_generating_ranking().  Must already contain all feature columns,
        'target_quintile', 'qid', 'date', 'Symbol', and 'close'.

    initial_capital : float
        Starting cash passed to run_xgboost_backtest (default 10 000).

    walk_forward_kwargs : dict, optional
        Extra keyword arguments forwarded to walk_forward_cv(), e.g.
        {'initial_train_months': 24, 'test_months': 6, 'gap_days': 21}.

    backtest_kwargs : dict, optional
        Extra keyword arguments forwarded to run_xgboost_backtest(), e.g.
        {'buy_fraction': 0.05, 'hold_fraction': 0.15,
         'trend_filter_col': 'dist_SMA_100'}.

    model_params : dict, optional
        XGBRanker hyper-parameters passed to walk_forward_cv(model_params=…).
        Defaults to the base_model() params defined in models.py.

    min_groups : int
        Minimum number of groups that must be present in a combination.
        Set to 2 if you want to skip single-group experiments.

    verbose : bool
        Print progress and results for every combination.

    Returns
    -------
    best_features : list[str]
        The flat list of feature column names in the winning combination.

    best_roi : float
        The final ROI (%) achieved by the winning combination.

    results_df : pd.DataFrame
        Summary table with columns:
            combo_name | groups | features | roi_pct | error
        sorted by roi_pct descending.
    """
    # --- lazy imports to avoid loading heavy libs at module import time ----
    from src.evaluation import run_xgboost_backtest
    from src.models import walk_forward_cv

    wf_kwargs = dict(
        initial_train_months=24,
        test_months=6,
        gap_days=21,
        use_mega=False,
        use_gp=False,
    )
    if walk_forward_kwargs:
        wf_kwargs.update(walk_forward_kwargs)

    bt_kwargs = dict(
        buy_fraction=0.05,
        hold_fraction=0.15,
        trailing_stop=-0.10,
        take_profit=0.50,
        time_of_rebalance="M",
        trend_filter_col="dist_SMA_100",
        settlement_delay=3,
    )
    if backtest_kwargs:
        bt_kwargs.update(backtest_kwargs)

    # --- build all combinations of groups ---------------------------------
    combos: list[tuple[str, ...]] = []
    for r in range(min_groups, len(ALL_GROUP_NAMES) + 1):
        combos.extend(itertools.combinations(ALL_GROUP_NAMES, r))

    total = len(combos)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature-Combo Search: {total} combinations to evaluate")
        print(f"Groups: {ALL_GROUP_NAMES}")
        print(f"{'='*60}\n")

    records: list[dict] = []
    best_features: list[str] = []
    best_roi: float = float("-inf")

    for idx, combo in enumerate(combos, start=1):
        combo_name = " + ".join(combo)
        features   = groups_to_features(list(combo))

        # Skip if any required column is missing in df
        missing = [f for f in features if f not in df.columns]
        if missing:
            if verbose:
                print(f"[{idx}/{total}] SKIP '{combo_name}': missing cols {missing}")
            records.append({
                "combo_name": combo_name,
                "groups":     list(combo),
                "features":   features,
                "roi_pct":    None,
                "error":      f"Missing columns: {missing}",
            })
            continue

        if verbose:
            print(f"[{idx}/{total}] Testing: {combo_name}")
            print(f"         Features ({len(features)}): {features}")

        try:
            # --- Walk-forward CV ------------------------------------------
            oos_df = walk_forward_cv(
                df=df,
                features=features,
                model_params=model_params,
                **wf_kwargs,
            )

            # --- Backtest on OOS predictions ------------------------------
            history = run_xgboost_backtest(
                df=oos_df,
                model=None,        # predictions already in 'pred_score' column
                features=None,
                initial_capital=initial_capital,
                **bt_kwargs,
            )

            roi = _final_roi(history)

            if verbose:
                print(f"         → Final ROI: {roi:+.2f}%\n")

            records.append({
                "combo_name": combo_name,
                "groups":     list(combo),
                "features":   features,
                "roi_pct":    roi,
                "error":      None,
            })

            if roi > best_roi:
                best_roi      = roi
                best_features = features
                if verbose:
                    print(f"  🏆 New best! ROI = {best_roi:+.2f}%  [{combo_name}]\n")

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"         ⚠️  ERROR — {err_msg}")
                traceback.print_exc()
            records.append({
                "combo_name": combo_name,
                "groups":     list(combo),
                "features":   features,
                "roi_pct":    None,
                "error":      err_msg,
            })

    # --- Summary table ----------------------------------------------------
    results_df = (
        pd.DataFrame(records)
        .sort_values("roi_pct", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    if verbose:
        print("\n" + "="*60)
        print("SEARCH COMPLETE — Top 5 combinations by ROI:")
        print("="*60)
        top5 = results_df.dropna(subset=["roi_pct"]).head(5)
        for _, row in top5.iterrows():
            print(f"  {row['roi_pct']:+8.2f}%  |  {row['combo_name']}")
        print()
        if best_features:
            print(f"🏆 Best combo : {results_df.iloc[0]['combo_name']}")
            print(f"   Final ROI  : {best_roi:+.2f}%")
            print(f"   Features   : {best_features}")

    return best_features, best_roi, results_df


# ---------------------------------------------------------------------------
# Optional: narrow the search to a specific subset of groups
# ---------------------------------------------------------------------------
def find_best_feature_combo_subset(
    df: pd.DataFrame,
    candidate_groups: list[str],
    **kwargs,
) -> tuple[list[str], float, pd.DataFrame]:
    """
    Like find_best_feature_combo() but only considers combinations built from
    `candidate_groups` (a subset of ALL_GROUP_NAMES).

    Example
    -------
        best_feat, best_roi, tbl = find_best_feature_combo_subset(
            df,
            candidate_groups=["returns", "volatility", "rsi"],
        )
    """
    # Temporarily restrict the global group list
    invalid = [g for g in candidate_groups if g not in FEATURE_GROUPS]
    if invalid:
        raise ValueError(f"Unknown group(s): {invalid}. Valid: {ALL_GROUP_NAMES}")

    # Monkey-patch locally then restore
    original = list(ALL_GROUP_NAMES)
    ALL_GROUP_NAMES.clear()
    ALL_GROUP_NAMES.extend(candidate_groups)
    try:
        return find_best_feature_combo(df, **kwargs)
    finally:
        ALL_GROUP_NAMES.clear()
        ALL_GROUP_NAMES.extend(original)


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python feature_search.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available feature groups:")
    for name, cols in FEATURE_GROUPS.items():
        print(f"  {name:20s} ({len(cols):2d} cols): {cols}")
    print()
    print(f"Total combinations to evaluate (all groups, min_groups=1): "
          f"{sum(1 for r in range(1, len(FEATURE_GROUPS)+1) for _ in itertools.combinations(ALL_GROUP_NAMES, r))}")
