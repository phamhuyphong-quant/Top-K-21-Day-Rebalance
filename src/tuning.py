import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import models as md
from src import evaluation as ev
from src import significance_test as st
from src.simulation import build_data_engines, OrderManager

ID = ["date", "Symbol"]


# ------------------------------------------------------------------ ensembling
def ensemble_predictions(preds):
    """
    preds: list of df_pred (one per seed, same rows). Returns one df whose pred_score is the
    mean of per-date rank percentiles across seeds (rank-averaging, scale-free).
    """
    ranks = []
    for p in preds:
        s = p.set_index(ID)["pred_score"]
        ranks.append(s.groupby(level="date").rank(pct=True))
    avg = pd.concat(ranks, axis=1).mean(axis=1).rename("pred_score")
    base = preds[0].drop(columns="pred_score").set_index(ID)
    out = base.join(avg).reset_index()
    return out


# ------------------------------------------------------------------ metrics
def _daily_ic(df, target_col):
    d = df[["date", "pred_score", target_col]].dropna()
    return d.groupby("date").apply(
        lambda g: spearmanr(g["pred_score"], g[target_col]).statistic, include_groups=False)


def model_metrics(df, k=10, ret_col="next_1m_ret", target_col="target_magnitude"):
    out = {}
    out["topk_avg"] = st.topk_daily_avg(df, k=k, target_col=target_col).mean()
    out["topk_vs_univ"] = st.compute_delta_vs_universe(df, k=k, target_col=target_col)["delta"].mean()

    ic = _daily_ic(df, target_col)
    out["ic_mean"] = ic.mean()
    out["icir"] = ic.mean() / ic.std() if ic.std() > 0 else 0.0

    if ret_col in df.columns:
        ex = md._daily_topk_excess_return(df, k, ret_col)
        out["topk_excess_ret"] = float(np.mean(ex)) if ex else np.nan
        prec = md._daily_precision_at_k(df, k, ret_col)
        out["precision_at_k"] = float(np.mean(prec)) if prec else np.nan
    return out


def simulate(df_pred, df_raw, trade_k, initial_capital, price_unit):
    eng_raw, eng_pred, _ = build_data_engines(df_raw, df_pred)
    om = OrderManager(initial_capital, trade_k, eng_raw, eng_pred, price_unit=price_unit)
    om.run_strategy()
    hist = pd.DataFrame(om.portfolio.history, columns=["date", "total_value"])
    hist["date"] = pd.to_datetime(hist["date"])
    perf = ev.print_performance_report(hist, initial_capital=initial_capital, verbose=False)
    return {"sharpe": perf["sharpe"], "cagr": perf["cagr"], "max_drawdown": perf["max_drawdown"]}


# ------------------------------------------------------------------ main
def select_lambda(df, features, base_params, lambda_pairs, seeds=tuple(range(8)),
                  k=10, df_raw=None, initial_capital=50_000_000, price_unit=1.0,
                  rank_metrics=("topk_avg", "ic_mean", "icir", "topk_excess_ret"),
                  block=21, alpha=0.05, save_prefix=None):
    """
    For each lambda: train len(seeds) models, rank-average them into ONE model, compute metrics at top-k.
    Then rank lambdas by the average rank over `rank_metrics`, test each vs the best on topk_daily_avg
    (block permutation, Holm), and report the plateau (lambdas statistically tied with the best).

    Returns dict(metrics, ranking, sig, plateau, choice, ensembles)
    """
    df_raw = df if df_raw is None else df_raw
    ensembles, rows = {}, []

    for lp in lambda_pairs:
        preds = []
        for seed in seeds:
            print(f"--- lambda={lp} seed={seed} ---")
            params = copy.deepcopy(base_params)
            params["lambdarank_num_pair_per_sample"] = lp
            params["random_state"] = seed
            p, _ = md.walk_forward_cv(
                df=df, model_params=params, features=features,
                icir_filter=False, corr_prune=True, window_mode="expanding",
                initial_train_dates=252 * 3, train_percent=0.8, top_k=k)
            preds.append(p)
        ens = ensemble_predictions(preds)
        ensembles[lp] = ens
        rows.append({"lambda_pairs": lp, **model_metrics(ens, k=k),
                     **simulate(ens, df_raw, k, initial_capital, price_unit)})

    metrics = pd.DataFrame(rows).set_index("lambda_pairs")

    # ---- rank: higher is better for every ranking metric
    ranks = metrics[list(rank_metrics)].rank(ascending=False)
    metrics["avg_rank"] = ranks.mean(axis=1)
    ranking = metrics.sort_values("avg_rank")
    best = ranking.index[0]

    # ---- is each lambda statistically worse than the best?
    sig_rows = []
    for lp in lambda_pairs:
        if lp == best:
            continue
        d = st.compute_delta_base_vs_model(ensembles[best], ensembles[lp], k=k)["delta"]
        sig_rows.append(st.run_comparison(f"{best} vs {lp}", d, block=block))
    sig = st.add_holm_correction(pd.DataFrame(sig_rows)) if sig_rows else pd.DataFrame()

    tied = [best]
    if not sig.empty:
        for _, r in sig.iterrows():
            lp = int(r["comparison"].split(" vs ")[1])
            if not (r["mean_delta"] > 0 and r["p_holm"] < alpha):   # best is NOT significantly better
                tied.append(lp)
    plateau = sorted(tied)
    choice = plateau[(len(plateau) - 1) // 2]   # middle of the plateau

    pd.set_option("display.width", 200)
    print("\n=== Metrics per lambda (seed-ensembled, k=%d) ===" % k)
    print(ranking.round(4).to_string())
    print(f"\n=== Best by avg rank: {best} ; tests vs best on topk_daily_avg (delta>0 => best better) ===")
    if not sig.empty:
        st.print_table(sig)
    print(f"\nPlateau (not significantly worse than best): {plateau}")
    print(f"Suggested choice (middle of plateau): {choice}")

    if save_prefix:
        ranking.to_csv(f"{save_prefix}_metrics.csv")
        sig.to_csv(f"{save_prefix}_sig.csv", index=False)

    return {"metrics": metrics, "ranking": ranking, "sig": sig, "plateau": plateau,
            "choice": choice, "ensembles": ensembles}


# res = select_lambda(df, BASIC_FEATURES, MODEL_PARAMS,
#                     lambda_pairs=[10, 25, 40, 50, 75, 100], seeds=range(8), k=10,
#                     initial_capital=50_000_000, price_unit=1.0, save_prefix="lambda_select")