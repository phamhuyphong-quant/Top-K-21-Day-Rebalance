"""
feature_search.py
-----------------
Searches every combination of feature *groups* (as defined in features.py)
to find which combination produces the highest final ROI when evaluated via
walk_forward_cv + simulate_portfolio.

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
        "volatility_1w",
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_shock_monthly",
        "volatility_shock_weekly",
    ],
    "moving_average": [
        "dist_SMA_9", "dist_SMA_21", "dist_SMA_50","dist_SMA_100","dist_SMA_200",
        "dist_EMA_9", "dist_EMA_21", "dist_EMA_50","dist_EMA_100","dist_EMA_200",
    ],
    "volume": [
        "volume_surge_monthly",
        "volume_surge_weekly",
        "obv_trend",
        "price_vol_divergence",
    ],
    "rsi": [
        "RSI_14",
    ],
    "wq_features":["WQ_Alpha_012","WQ_Alpha_024","WQ_Alpha_028","WQ_Alpha_053","WQ_Alpha_060"],
    
    "price_structure": [
    "dist_52w_high",
    "log_ret_skip1m",]

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
# Metric helpers
# ---------------------------------------------------------------------------
def _final_roi(history: pd.DataFrame) -> float:
    if history is None or history.empty:
        return float("-inf")
    initial = history["total_value"].iloc[0]
    final   = history["total_value"].iloc[-1]
    if initial == 0:
        return float("-inf")
    return (final / initial - 1) * 100.0


def _sharpe_ratio(history: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    if history is None or history.empty:
        return float("-inf")
    returns = history["total_value"].pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return float("-inf")
    # Annualise assuming monthly rows → factor = √12
    excess = returns - risk_free_rate / 12
    return (excess.mean() / excess.std()) * (12 ** 0.5)


# ---------------------------------------------------------------------------
# Core search loop (shared by both public functions)
# ---------------------------------------------------------------------------
def _run_combo_search(
    df: pd.DataFrame,
    metric_fn,
    metric_name: str,
    *,
    initial_capital: float = 10_000,
    walk_forward_kwargs: Optional[dict] = None,
    backtest_kwargs: Optional[dict] = None,
    model_params: Optional[dict] = None,
    min_groups: int = 1,
    verbose: bool = True,
) -> tuple[list[str], float, pd.DataFrame]:

    from src.evaluation import simulate_portfolio
    from src.models import walk_forward_cv

    wf_kwargs = dict(
        initial_train_months=24, test_months=6,
        gap_days=21, use_mega=False, use_gp=False,
    )
    if walk_forward_kwargs:
        wf_kwargs.update(walk_forward_kwargs)

    bt_kwargs = dict(
        buy_fraction=0.05, hold_fraction=0.15,
        trailing_stop=-0.10, take_profit=0.50,
        time_of_rebalance="M", trend_filter_col="dist_SMA_100",
        settlement_delay=3,
    )
    if backtest_kwargs:
        bt_kwargs.update(backtest_kwargs)

    combos: list[tuple[str, ...]] = []
    for r in range(min_groups, len(ALL_GROUP_NAMES) + 1):
        combos.extend(itertools.combinations(ALL_GROUP_NAMES, r))

    total = len(combos)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature-Combo Search ({metric_name}): {total} combinations")
        print(f"Groups: {ALL_GROUP_NAMES}")
        print(f"{'='*60}\n")

    records: list[dict] = []
    best_features: list[str] = []
    best_score: float = float("-inf")

    for idx, combo in enumerate(combos, start=1):
        combo_name = " + ".join(combo)
        features   = groups_to_features(list(combo))

        missing = [f for f in features if f not in df.columns]
        if missing:
            if verbose:
                print(f"[{idx}/{total}] SKIP '{combo_name}': missing cols {missing}")
            records.append(dict(combo_name=combo_name, groups=list(combo),
                                features=features, score=None,
                                error=f"Missing columns: {missing}"))
            continue

        if verbose:
            print(f"[{idx}/{total}] Testing: {combo_name}")

        try:
            oos_df = walk_forward_cv(
                df=df, features=features,
                model_params=model_params, **wf_kwargs,
            )
            history = simulate_portfolio(
                df=oos_df, model=None, features=None,
                initial_capital=initial_capital, **bt_kwargs,
            )
            score = metric_fn(history)

            if verbose:
                print(f"         → {metric_name}: {score:+.4f}\n")

            records.append(dict(combo_name=combo_name, groups=list(combo),
                                features=features, score=score, error=None))

            if score > best_score:
                best_score    = score
                best_features = features
                if verbose:
                    print(f"  🏆 New best! {metric_name} = {best_score:+.4f}  [{combo_name}]\n")

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"         ⚠️  ERROR — {err_msg}")
                traceback.print_exc()
            records.append(dict(combo_name=combo_name, groups=list(combo),
                                features=features, score=None, error=err_msg))

    results_df = (
        pd.DataFrame(records)
        .sort_values("score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    if verbose:
        print("\n" + "="*60)
        print(f"SEARCH COMPLETE — Top 5 by {metric_name}:")
        print("="*60)
        for _, row in results_df.dropna(subset=["score"]).head(5).iterrows():
            print(f"  {row['score']:+10.4f}  |  {row['combo_name']}")
        if best_features:
            print(f"\n🏆 Best combo : {results_df.iloc[0]['combo_name']}")
            print(f"   {metric_name:12s}: {best_score:+.4f}")
            print(f"   Features    : {best_features}")

    return best_features, best_score, results_df


# ---------------------------------------------------------------------------
# Public functions
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
    """Optimises for final ROI (%)."""
    return _run_combo_search(
        df, _final_roi, "ROI (%)",
        initial_capital=initial_capital,
        walk_forward_kwargs=walk_forward_kwargs,
        backtest_kwargs=backtest_kwargs,
        model_params=model_params,
        min_groups=min_groups,
        verbose=verbose,
    )


def find_best_feature_combo_sharpe(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    initial_capital: float = 10_000,
    walk_forward_kwargs: Optional[dict] = None,
    backtest_kwargs: Optional[dict] = None,
    model_params: Optional[dict] = None,
    min_groups: int = 1,
    verbose: bool = True,
) -> tuple[list[str], float, pd.DataFrame]:
    """Optimises for annualised Sharpe ratio (assumes monthly history rows)."""
    metric_fn = lambda h: _sharpe_ratio(h, risk_free_rate)
    return _run_combo_search(
        df, metric_fn, "Sharpe",
        initial_capital=initial_capital,
        walk_forward_kwargs=walk_forward_kwargs,
        backtest_kwargs=backtest_kwargs,
        model_params=model_params,
        min_groups=min_groups,
        verbose=verbose,
    )


def find_best_feature_combo_subset(
    df: pd.DataFrame,
    candidate_groups: list[str],
    metric: str = "roi",
    **kwargs,
) -> tuple[list[str], float, pd.DataFrame]:
    """
    Subset search — pass metric='roi' or metric='sharpe'.
    Any extra kwargs (including risk_free_rate) are forwarded.
    """
    invalid = [g for g in candidate_groups if g not in FEATURE_GROUPS]
    if invalid:
        raise ValueError(f"Unknown group(s): {invalid}. Valid: {ALL_GROUP_NAMES}")

    fn = find_best_feature_combo_sharpe if metric == "sharpe" else find_best_feature_combo

    original = list(ALL_GROUP_NAMES)
    ALL_GROUP_NAMES.clear()
    ALL_GROUP_NAMES.extend(candidate_groups)
    try:
        return fn(df, **kwargs)
    finally:
        ALL_GROUP_NAMES.clear()
        ALL_GROUP_NAMES.extend(original)



def search_best_roi_and_sharpe(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.045,
    initial_capital: float = 10_000,
    walk_forward_kwargs: Optional[dict] = None,
    backtest_kwargs: Optional[dict] = None,
    model_params: Optional[dict] = None,
    min_groups: int = 1,
    verbose: bool = True,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Runs the search once and identifies the best combo for ROI 
    and the best combo for Sharpe Ratio separately.
    """
    from src.evaluation import simulate_portfolio
    from src.models import walk_forward_cv

    # 1. Setup default parameters
    wf_kwargs = dict(initial_train_months=24, test_months=6, gap_days=21)
    if walk_forward_kwargs: wf_kwargs.update(walk_forward_kwargs)

    bt_kwargs = dict(buy_fraction=0.05, hold_fraction=0.15, trend_filter_col="dist_SMA_100")
    if backtest_kwargs: bt_kwargs.update(backtest_kwargs)

    # 2. Generate all combinations
    combos = []
    for r in range(min_groups, len(ALL_GROUP_NAMES) + 1):
        combos.extend(itertools.combinations(ALL_GROUP_NAMES, r))

    records = []
    total = len(combos)

    for idx, combo in enumerate(combos, start=1):
        combo_name = " + ".join(combo)
        features = groups_to_features(list(combo))

        if verbose:
            print(f"[{idx}/{total}] Testing: {combo_name}")

        try:
            # Execute simulation (the expensive part)
            oos_df = walk_forward_cv(df=df, features=features, model_params=model_params, **wf_kwargs)
            history = simulate_portfolio(df=oos_df, model=None, features=None, initial_capital=initial_capital, **bt_kwargs)

            # Calculate both metrics from the SAME history object
            roi_val = _final_roi(history)
            sharpe_val = _sharpe_ratio(history, risk_free_rate=risk_free_rate)

            if verbose:
                print(f"         >> Result: ROI = {roi_val:>8.2f}% | Sharpe = {sharpe_val:>6.2f}")
                
                # Optional: Highlight if it's a "leader" in either category
                if not records: # First successful run
                    print("         🏆 Current Leader (First Run)")
                else:
                    is_best_roi = roi_val > max([r['roi_%'] for r in records if r['roi_%'] is not None], default=float('-inf'))
                    is_best_sharpe = sharpe_val > max([r['sharpe'] for r in records if r['sharpe'] is not None], default=float('-inf'))
                    if is_best_roi or is_best_sharpe:
                        lead_msg = " + ".join([m for c, m in [(is_best_roi, "ROI"), (is_best_sharpe, "Sharpe")] if c])
                        print(f"         🏆 New Best {lead_msg}!")


            records.append({
                "combo_name": combo_name,
                "roi_%": roi_val,
                "sharpe": sharpe_val,
                "features": features
            })

        except Exception as exc:
            if verbose: print(f" Error: {exc}")
            continue

    # 3. Create results dataframe and find bests
    results_df = pd.DataFrame(records)
    
    # Identify the best for each metric independently
    best_roi_row = results_df.loc[results_df["roi_%"].idxmax()].to_dict()
    best_sharpe_row = results_df.loc[results_df["sharpe"].idxmax()].to_dict()

    if verbose:
        print("\n" + "="*40)
        print(f"BEST ROI: {best_roi_row['roi_%']:.2f}% ({best_roi_row['combo_name']})")
        print(f"BEST SHARPE: {best_sharpe_row['sharpe']:.2f} ({best_sharpe_row['combo_name']})")
        print("="*40)

    return best_roi_row, best_sharpe_row, results_df


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
