import numpy as np
import optuna
import pandas as pd
import os
from gplearn.genetic import SymbolicTransformer
import xgboost as xgb
from config import BASE_MODEL_PARAMS
from sklearn.metrics import ndcg_score
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.deep_combiner import AlphaForgeCombiner
def select_features_by_icir(
    df: pd.DataFrame,
    candidate_features: list[str],
    icir_threshold: float = 0.02,
    target_col: str = "next_1m_ret",
) -> tuple[list[str], dict[str, float]]:
    """
    Computes IC/IR for each candidate feature on the given df (which should be
    all data available up to — but NOT including — the test window), then returns
    only the features whose |IC IR| exceeds the threshold.

    IC  = per-date Spearman rank correlation between feature and forward return.
    IR  = IC_mean / IC_std  (risk-adjusted signal quality).

    Vectorized implementation: ranks all features and the target within each
    date in a single groupby pass, then computes Pearson correlation on those
    ranks (== Spearman) via corrwith — no per-feature scipy loop.

    Parameters
    ----------
    df                  : Training-window data (no future rows).
    candidate_features  : Full list of features to evaluate.
    icir_threshold      : Minimum |IC IR| to keep a feature (default 0.02).
    target_col          : Forward-return column used as the prediction target.

    Returns
    -------
    selected  : List of features that survive the |IC IR| > icir_threshold filter.
    ic_ir_map : Dict of {feature: ic_ir_value} for ALL valid features — returned
                as a free byproduct so callers (e.g. corr_prune) never need to
                recompute it.
    """
    df_ic  = df.dropna(subset=[target_col]).copy()
    valid  = [f for f in candidate_features if f in df_ic.columns]

    if not valid:
        return [], {}

    cols        = valid + [target_col]
    # Rank within each date in one vectorized pass (Pearson on ranks == Spearman)
    ranked      = df_ic[["date"] + cols].copy()
    ranked[cols] = (
        ranked.groupby("date")[cols]
        .rank(method="average")
    )

    # Per-date IC for every feature at once via corrwith
    ic_by_date = (
        ranked.groupby("date")[cols]
        .apply(lambda g: g.drop(columns="date", errors="ignore")
                          .corrwith(g[target_col]),
               include_groups=False)
        .drop(columns=target_col, errors="ignore")
        .dropna(how="all")
    )

    ic_mean   = ic_by_date.mean()
    ic_std    = ic_by_date.std()
    ic_ir_s   = (ic_mean / ic_std.replace(0.0, float("nan"))).fillna(0.0)
    ic_ir_map = ic_ir_s.to_dict()

    # A feature needs >= 2 valid IC dates to have a stable IR
    valid_counts = ic_by_date.count()
    selected = [
        f for f in valid
        if valid_counts.get(f, 0) >= 2 and abs(ic_ir_map.get(f, 0.0)) > icir_threshold
    ]

    return selected, ic_ir_map


def _compute_icir_map(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "next_1m_ret",
) -> dict[str, float]:
    """
    Helper: compute IC IR for each feature on the given df.
    Returns {feature: ic_ir_value}.  Features with fewer than 2 valid
    IC dates get ic_ir = 0.0 (they will be pruned first if correlated).

    Vectorized implementation: ranks all features and the target within each
    date in a single groupby pass, then computes Pearson correlation on those
    ranks (== Spearman) via corrwith — no per-feature scipy loop.
    """
    df_ic = df.dropna(subset=[target_col]).copy()
    valid = [f for f in features if f in df_ic.columns]

    ic_ir_map: dict[str, float] = {f: 0.0 for f in features}
    if not valid:
        return ic_ir_map

    cols         = valid + [target_col]
    ranked       = df_ic[["date"] + cols].copy()
    ranked[cols] = ranked.groupby("date")[cols].rank(method="average")

    ic_by_date = (
        ranked.groupby("date")[cols]
        .apply(lambda g: g.drop(columns="date", errors="ignore")
                          .corrwith(g[target_col]),
               include_groups=False)
        .drop(columns=target_col, errors="ignore")
        .dropna(how="all")
    )

    ic_mean        = ic_by_date.mean()
    ic_std         = ic_by_date.std()
    ic_ir_s        = (ic_mean / ic_std.replace(0.0, float("nan"))).fillna(0.0)
    valid_counts   = ic_by_date.count()

    for f in valid:
        ic_ir_map[f] = ic_ir_s.get(f, 0.0) if valid_counts.get(f, 0) >= 2 else 0.0

    return ic_ir_map


def prune_correlated_features(
    df: pd.DataFrame,
    candidate_features: list[str],
    ic_ir_map: dict[str, float],
    feature_groups: dict[str, list[str]] | None = None,
    correlation_threshold: float = 0.75,
) -> list[str]:
    """
    Removes redundant features using Spearman correlation pruning — mirrors the
    notebook-02 logic but operates only on training-window data so there is
    zero look-ahead bias.

    For every pair of features within the same group whose |Spearman correlation|
    >= correlation_threshold, the feature with the lower |IC IR| is dropped.
    Features that do not belong to any group are treated as their own singleton
    group (never pruned against each other).

    Parameters
    ----------
    df                    : Training-window data used to compute the correlation matrix.
    candidate_features    : Features to evaluate (already IC/IR-filtered is fine).
    ic_ir_map             : Dict mapping feature name → IC IR value (from
                            select_features_by_icir or pre-computed).
    feature_groups        : Optional dict of {group_name: [feat, ...]} that defines
                            which features compete against each other.  If None,
                            all candidate_features are treated as one group.
    correlation_threshold : |Spearman corr| >= this → drop the weaker feature.

    Returns
    -------
    List of features that survive pruning, preserving input order.
    """
    from scipy.stats import spearmanr

    present = [f for f in candidate_features if f in df.columns]
    if len(present) < 2:
        return present

    # Compute Spearman correlation matrix on the training window (drop NaN rows
    # to keep it well-defined; use the same approach as the notebook).
    df_clean = df[present].dropna()
    if df_clean.shape[0] < 2:
        return present

    corr_matrix, _ = spearmanr(df_clean)
    if corr_matrix.ndim == 0:
        # Only one feature after dropna — nothing to prune
        return present

    # Use a numpy array + integer index map for fast inner-loop lookups
    # (avoids repeated pandas label-based .loc[] on every pair)
    corr_arr = np.abs(np.asarray(corr_matrix))
    feat_idx = {f: i for i, f in enumerate(present)}

    # Build group membership: each feature belongs to exactly one group.
    # Features absent from every group get a private singleton group.
    if feature_groups is None:
        groups_to_check = {"all": present}
    else:
        feat_to_group: dict[str, str] = {}
        for gname, gfeats in feature_groups.items():
            for f in gfeats:
                if f in feat_idx:
                    feat_to_group[f] = gname

        groups_to_check: dict[str, list[str]] = {}
        for f in present:
            g = feat_to_group.get(f, f"__singleton_{f}")
            groups_to_check.setdefault(g, []).append(f)

    # Pre-build IC/IR array indexed the same way as corr_arr
    ir_arr  = np.array([abs(ic_ir_map.get(f, 0.0)) for f in present])
    to_drop: set[str] = set()

    for group_name, group_feats in groups_to_check.items():
        group_feats = [f for f in group_feats if f in feat_idx]
        for i in range(len(group_feats)):
            if group_feats[i] in to_drop:
                continue
            ii = feat_idx[group_feats[i]]
            for j in range(i + 1, len(group_feats)):
                feat_j = group_feats[j]
                if feat_j in to_drop:
                    continue
                jj = feat_idx[feat_j]
                if corr_arr[ii, jj] >= correlation_threshold:
                    drop = feat_j if ir_arr[ii] >= ir_arr[jj] else group_feats[i]
                    to_drop.add(drop)

    pruned = [f for f in present if f not in to_drop]
    return pruned


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

def build_mega_combiner(
    ic_window: int = 40,
    ic_threshold: float = 0.02,
    icir_threshold: float = 0.2,
    max_active_factors: int = 13,
    ridge_alpha: float = 1.0,
):
    """
    Constructs an AlphaForgeCombiner instance (AlphaForge Algorithm 2).

    No training happens here — the combiner recomputes dynamic weights at every
    rebalance date using rolling RankIC/ICIR from the available history.

    Parameters mirror AlphaForgeCombiner.__init__; see deep_combiner.py for details.
    """
    from src.deep_combiner import AlphaForgeCombiner
    return AlphaForgeCombiner(
        ic_window=ic_window,
        ic_threshold=ic_threshold,
        icir_threshold=icir_threshold,
        max_active_factors=max_active_factors,
        ridge_alpha=ridge_alpha,
    )

from dataclasses import dataclass

@dataclass
class Fold:
    fold_number:  int
    train_df:     pd.DataFrame
    test_df:      pd.DataFrame
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp

def generate_folds(
    df: pd.DataFrame,
    initial_train_months: int = 12,
    test_months: int = 6,
    gap_days: int = 21,
    # --- per-fold liquidity filter ---
    liquidity_filter: bool = False,
    capital: float = 100_000,
    buy_fraction: float = 0.10,
    adtv_lookback: int = 20,
    adtv_participation: float = 0.10,
    min_adtv: float = 0.0,
) -> list[Fold]:
    """
    Produces walk-forward fold splits. Completely model-agnostic.
    gap_days: buffer between train cutoff and test start to avoid leakage.

    Liquidity filter (optional)
    ---------------------------
    When liquidity_filter=True, each fold's train_df and test_df are
    independently filtered by _filter_fold_by_liquidity().  ADTV is
    computed using the rolling window over all history available up to
    each date within that fold, so:
      - No future volume leaks into an earlier fold's filter decision.
      - A stock illiquid in fold N can still appear in fold N+k if its
        volume recovers by then (e.g. post-IPO lock-up, re-listing, etc.).
      - The initial training window rows are filtered with the same logic,
        so the model is never trained on stocks it couldn't trade at that
        point in time.
    """
    min_date = df["date"].min()
    max_date = df["date"].max()
    current_train_end = min_date + pd.DateOffset(months=initial_train_months)
    folds, fold_num = [], 1

    while current_train_end < max_date:
        train_cutoff = current_train_end - pd.Timedelta(days=gap_days)
        test_start   = current_train_end
        test_end     = test_start + pd.DateOffset(months=test_months)

        train_df = df[df["date"] <= train_cutoff].copy()
        test_df  = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()

        # ── Per-fold liquidity filter (before qid assignment) ─────────────
        # Applied before qid so group numbers are contiguous after filtering.
        if liquidity_filter and not test_df.empty:
            train_df, test_df = _filter_fold_by_liquidity(
                train_df, test_df,
                fold_number=fold_num,
                capital=capital,
                buy_fraction=buy_fraction,
                adtv_lookback=adtv_lookback,
                adtv_participation=adtv_participation,
                min_adtv=min_adtv,
            )

        # qid assignment lives here, not in each model
        train_df["qid"] = train_df.groupby("date").ngroup()
        test_df["qid"]  = test_df.groupby("date").ngroup()

        if not test_df.empty:
            folds.append(Fold(fold_num, train_df, test_df, test_start, test_end))

        current_train_end = test_end
        fold_num += 1

    return folds
def score_and_evaluate(
    test_df: pd.DataFrame,
    fold: Fold,
    total_folds: int,
    path_label: str,
) -> pd.DataFrame:
    """
    Assigns pred_quintile and computes mean daily NDCG.
    Expects test_df to have a 'pred_score' column already set.
    """
    test_df = test_df.copy()
    test_df["pred_quintile"] = test_df.groupby("date")["pred_score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    )

    daily_ndcg = [
        ndcg_score([g["target_quintile"].values], [g["pred_score"].values])
        for _, g in test_df.groupby("date")
        if len(g) > 1
    ]
    mean_ndcg = np.mean(daily_ndcg) if daily_ndcg else 0.0

    print(
        f"Fold {fold.fold_number}/{total_folds} [{path_label}] "
        f"({fold.test_start.strftime('%Y-%m')}): NDCG = {mean_ndcg:.4f}"
    )
    return test_df

def mine_gp_factors(
    df: pd.DataFrame,
    features: list[str],
    initial_train_end: pd.Timestamp,
    n_components: int = 15,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fits SymbolicTransformer on the initial training window only (no leakage),
    then transforms the full df. Returns df with GP_Alpha_* columns appended,
    plus the list of new column names.

    Fitting on the initial window only mirrors live deployment: the factor
    grammar is fixed at launch and never re-learned on future returns.
    """
    from gplearn.genetic import SymbolicTransformer
    print("🧬 Mining GP alpha factors (fit on initial window only)...")

    init_df = df[df["date"] < initial_train_end]
    X_init  = np.nan_to_num(init_df[features].values)
    y_init  = np.nan_to_num(init_df["next_1m_ret"].values)

    gp_model = SymbolicTransformer(
        generations=20, population_size=1000, n_components=n_components,
        metric="pearson", n_jobs=-1, random_state=42, max_samples=0.5,
    )
    gp_model.fit(X_init, y_init)

    X_full     = np.nan_to_num(df[features].values)
    gp_values  = gp_model.transform(X_full)
    gp_cols    = [f"GP_Alpha_{i}" for i in range(gp_values.shape[1])]

    df = df.copy()
    for i, col in enumerate(gp_cols):
        df[col] = gp_values[:, i]

    print(f"   ✅ {len(gp_cols)} GP factors added.")
    return df, gp_cols

def _filter_fold_by_liquidity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_number: int,
    capital: float,
    buy_fraction: float,
    adtv_lookback: int,
    adtv_participation: float,
    min_adtv: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the ADTV liquidity filter independently to one fold.

    ADTV is computed using a rolling window over the *full history up to each
    date* (train + test combined, sorted by date), but the decision to keep or
    drop a row is made using only that row's own ADTV value — so no future
    information crosses the train/test boundary.

    Why per-fold?
    - A stock illiquid in fold N may be perfectly liquid in fold N+2 (e.g.
      after an IPO lock-up or a sustained volume increase).  Filtering globally
      would silently remove it from all folds including the future ones where
      it is tradeable, making CV metrics inconsistent with the live simulator.
    - ADTV for the very first training dates must be estimated from whatever
      history is available; computing it on the full df would use future volume.

    Position-size estimate
    ----------------------
    cash_per_stock ≈ capital / max(1, n_stocks_in_universe × buy_fraction)
    This matches the denominator simulate_portfolio uses when splitting cash.
    The estimate is computed per date so universe-size changes across the fold
    are respected.

    Parameters
    ----------
    train_df / test_df   : Split DataFrames for this fold (must have 'close', 'volume').
    fold_number          : Used only for the diagnostic print.
    capital              : Approximate portfolio size (match simulate_portfolio).
    buy_fraction         : Fraction of universe targeted each period.
    adtv_lookback        : Rolling window for ADTV (default 20, same as simulator).
    adtv_participation   : Max order size as fraction of ADTV (default 0.10).
    min_adtv             : Hard VND floor; 0 = disabled.

    Returns
    -------
    (filtered_train_df, filtered_test_df) — temp ADTV columns are dropped.
    """
    # ── 1. Stack train + test so rolling ADTV sees the full history ───────
    #    We tag each row so we can split them back out after computing ADTV.
    train_tagged          = train_df.copy()
    test_tagged           = test_df.copy()
    train_tagged["_split"] = "train"
    test_tagged["_split"]  = "test"

    combined = (
        pd.concat([train_tagged, test_tagged])
        .sort_values(["Symbol", "date"])
    )

    # ── 2. Rolling ADTV — same formula as simulate_portfolio ─────────────
    combined["_traded_value"] = combined["close"] * combined["volume"]
    combined["_adtv"] = (
        combined.groupby("Symbol")["_traded_value"]
        .transform(lambda x: x.rolling(adtv_lookback, min_periods=5).mean())
    )

    # ── 3. Per-date position-size estimate ────────────────────────────────
    n_per_date              = combined.groupby("date")["Symbol"].transform("count")
    n_targets               = (n_per_date * buy_fraction).clip(lower=1)
    combined["_pos_size"]   = capital / n_targets

    # ── 4. Apply filter masks ─────────────────────────────────────────────
    adtv_ok = combined["_adtv"].isna() | (
        combined["_pos_size"] <= adtv_participation * combined["_adtv"]
    )
    min_ok  = (min_adtv == 0) | combined["_adtv"].isna() | (combined["_adtv"] >= min_adtv)
    mask    = adtv_ok & min_ok

    # ── 5. Diagnostic ─────────────────────────────────────────────────────
    n_before = len(combined)
    n_after  = mask.sum()
    print(
        f"   💧 Fold {fold_number} liquidity filter: "
        f"{n_before - n_after:,} rows removed "
        f"({(n_before - n_after) / n_before:.1%}), "
        f"{n_after:,} kept."
    )

    # ── 6. Drop temp cols and re-split ────────────────────────────────────
    tmp_cols = ["_traded_value", "_adtv", "_pos_size", "_split"]
    filtered = combined[mask].drop(columns=tmp_cols)

    out_train = filtered[filtered.index.isin(train_df.index)].copy()
    out_test  = filtered[filtered.index.isin(test_df.index)].copy()

    return out_train, out_test

def predict_alphaforge(
    fold: Fold,
    alpha_pool: list[str],
    combiner_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    AlphaForge Algorithm 2 (arXiv:2406.18394).
    Mega-Alpha is the sole ranking signal — no XGBoost involved.

    At each test date d, the combiner sees only:
      - the full training history, plus
      - test rows from dates strictly before d
    This preserves the paper's walk-forward guarantee within the fold.

    Returns fold.test_df with 'pred_score' column set to Mega_Alpha.
    """
    kwargs   = combiner_kwargs or {}
    combiner = AlphaForgeCombiner(
        max_active_factors=min(13, len(alpha_pool)), **kwargs
    )

    test_df     = fold.test_df.copy()
    accumulated = fold.train_df.copy()

    for d in sorted(test_df["date"].unique()):
        cur    = test_df[test_df["date"] == d]
        scores = combiner.fit_and_predict(accumulated, cur, alpha_pool)
        test_df.loc[cur.index, "Mega_Alpha"] = scores.values
        accumulated = pd.concat([accumulated, cur])

    test_df["pred_score"] = test_df["Mega_Alpha"]
    return test_df
def predict_xgboost(
    fold: Fold,
    features: list[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """
    XGBRanker baseline. GP columns (if any) should already be included in
    `features` before calling — this function is unaware of GP.

    Returns fold.test_df with 'pred_score' column set.
    """
    params = model_params or BASE_MODEL_PARAMS
    train_df = fold.train_df.sort_values("qid").reset_index(drop=True)
    test_df  = fold.test_df.copy().sort_values("qid").reset_index(drop=True)

    ranker = xgb.XGBRanker(**params)
    ranker.fit(
        train_df[features], train_df["target_quintile"],
        qid=train_df["qid"],
        #eval_set=[(test_df[features], test_df["target_quintile"])],
        #eval_qid=[test_df["qid"]],
        verbose=False,
    )
    test_df["pred_score"] = ranker.predict(test_df[features])
    return test_df
def walk_forward_cv(
    df: pd.DataFrame,
    features: list[str],
    model: str = "xgboost",          # "xgboost" | "alphaforge"
    model_params: dict | None = None,
    initial_train_months: int = 12,
    test_months: int = 6,
    gap_days: int = 21,
    use_gp: bool = False,
    # --- Per-fold IC/IR feature filter (no look-ahead bias) ---
    icir_filter: bool = False,
    icir_threshold: float = 0.02,
    icir_target_col: str = "next_1m_ret",
    # --- Per-fold correlation pruning (applied after IC/IR filter) ---
    corr_prune: bool = False,
    corr_threshold: float = 0.75,
    # --- Liquidity filter (mirrors simulate_portfolio) ---
    liquidity_filter: bool = False,
    capital: float = 100_000,
    buy_fraction: float = 0.10,
    adtv_lookback: int = 20,
    adtv_participation: float = 0.10,
    min_adtv: float = 0.0,
) -> pd.DataFrame:
    """
    Orchestrates walk-forward CV. model= selects the prediction path:
      "xgboost"    → XGBRanker on hand-crafted + optional GP features
      "alphaforge" → AlphaForgeCombiner Mega-Alpha as sole ranking signal

    Per-fold IC/IR feature filter (optional, zero look-ahead)
    ----------------------------------------------------------
    When icir_filter=True, feature selection is re-run at the start of
    every fold using only the data in fold.train_df (i.e. all rows up to
    the train cutoff for that fold).  This guarantees:
      - No future return information leaks into feature selection.
      - The feature set adapts over time as signal quality evolves.
      - Early folds with less history use whatever data is available;
        features with fewer than 2 valid IC dates are excluded.

    The filter evaluates every feature in `features` (+ GP columns if
    use_gp=True) and keeps only those with |IC IR| > icir_threshold.
    At least one feature is always kept (the highest |IC IR| feature)
    to avoid passing an empty feature list to the model.

    Per-fold correlation pruning (optional, zero look-ahead)
    ---------------------------------------------------------
    When corr_prune=True, a second pruning step runs after the IC/IR
    filter (or directly on the full candidate pool if icir_filter=False).
    For every pair of features within the same FEATURE_GROUPS group whose
    |Spearman correlation| >= corr_threshold (default 0.75), the feature
    with the lower |IC IR| is dropped.  This mirrors the notebook-02
    logic but is computed purely on fold.train_df so there is zero
    look-ahead bias.  corr_prune=True automatically requires the IC IR
    values to be computed; if icir_filter=False they are computed
    internally just for tie-breaking.

    Liquidity filter (optional, mirrors simulate_portfolio)
    -------------------------------------------------------
    When liquidity_filter=True, the ADTV filter is applied *per fold*
    inside generate_folds(), not upfront on the whole dataframe.

    Adding a new model: implement predict_<name>(fold, ...) → pd.DataFrame
    and add an elif branch below.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "Symbol"])

    min_date           = df["date"].min()
    initial_train_end  = min_date + pd.DateOffset(months=initial_train_months)

    # --- One-time GP mining (fit on initial window, transform full df) ---
    gp_cols = []
    if use_gp:
        df, gp_cols = mine_gp_factors(df, list(features), initial_train_end)

    # --- Collect WQ factor zoo (AlphaForge path only) ---
    wq_cols = [c for c in df.columns if c.startswith("WQ_Alpha_")]
    if model == "alphaforge":
        alpha_pool = wq_cols + gp_cols
        if not alpha_pool:
            raise ValueError("alphaforge requires WQ_Alpha_* columns. Call build_features() first.")

    # --- Fold generation (liquidity filter applied per-fold inside) -------
    folds = generate_folds(
        df,
        initial_train_months=initial_train_months,
        test_months=test_months,
        gap_days=gap_days,
        liquidity_filter=liquidity_filter,
        capital=capital,
        buy_fraction=buy_fraction,
        adtv_lookback=adtv_lookback,
        adtv_participation=adtv_participation,
        min_adtv=min_adtv,
    )
    total_folds = len(folds)
    results     = []

    print("🚀 Starting walk-forward fold loop...")
    for fold in folds:

        # ── Per-fold IC/IR feature selection (bias-free) ──────────────────
        # Uses only fold.train_df — all rows from the start of history up
        # to this fold's train cutoff.  No test-window data is ever seen.
        #
        # select_features_by_icir returns (selected, ic_ir_map) together as
        # a free byproduct of the same vectorized pass, so corr_prune can
        # reuse it directly — no second compute regardless of which filters
        # are active.
        candidate_pool = list(features) + gp_cols

        if icir_filter:
            fold_selected, _ic_ir_map = select_features_by_icir(
                df=fold.train_df,
                candidate_features=candidate_pool,
                icir_threshold=icir_threshold,
                target_col=icir_target_col,
            )
            # Safety: never pass an empty list to the model
            if not fold_selected:
                fold_selected = candidate_pool[:1]
                print(
                    f"   ⚠️  Fold {fold.fold_number}: no feature passed |IC IR| > "
                    f"{icir_threshold}; keeping {fold_selected[0]!r} as fallback."
                )
            else:
                print(
                    f"   🔍 Fold {fold.fold_number}: {len(fold_selected)}/{len(candidate_pool)} "
                    f"features selected by IC/IR filter."
                )
            fold_features = fold_selected
        elif corr_prune:
            # corr_prune=True but icir_filter=False: keep all features but
            # still need ic_ir_map for tie-breaking — compute it once here.
            fold_features = candidate_pool
            _ic_ir_map    = _compute_icir_map(
                fold.train_df, fold_features, icir_target_col
            )
        else:
            fold_features = candidate_pool

        # ── Per-fold correlation pruning (bias-free) ───────────────────────
        # Removes redundant features whose |Spearman corr| >= corr_threshold
        # within the same feature group, keeping the higher |IC IR| one.
        # Computed on fold.train_df only — zero look-ahead.
        # _ic_ir_map is guaranteed to exist here whenever corr_prune=True
        # (set by whichever branch above was active).
        if corr_prune and len(fold_features) > 1:
            from config import FEATURE_GROUPS as _FG

            pruned = prune_correlated_features(
                df=fold.train_df,
                candidate_features=fold_features,
                ic_ir_map=_ic_ir_map,
                feature_groups=_FG,
                correlation_threshold=corr_threshold,
            )
            n_before = len(fold_features)
            fold_features = pruned if pruned else fold_features[:1]
            print(
                f"   ✂️  Fold {fold.fold_number}: correlation pruning "
                f"{n_before} → {len(fold_features)} features "
                f"(threshold={corr_threshold})."
            )

        if model == "alphaforge":
            test_df    = predict_alphaforge(fold, alpha_pool)
            path_label = "AlphaForge (Mega)"
        elif model == "xgboost":
            test_df    = predict_xgboost(fold, fold_features, model_params)
            path_label = "XGBoost"
        # elif model == "lightgbm":
        #     test_df = predict_lightgbm(fold, fold_features, model_params)
        else:
            raise ValueError(f"Unknown model: {model!r}")

        test_df = score_and_evaluate(test_df, fold, total_folds, path_label)
        results.append(test_df)
        gc.collect()

    print("\n✅ Walk-forward CV complete.")
    return pd.concat(results)
def optimize_xgboost_ranker(df, features, n_trials=50):
    """
    Uses Optuna to find the mathematically perfect XGBoost parameters.
    """
    print("Preparing data for Optuna...")
    
    # 1. Create a recent Train/Validation split (e.g., train on 2022-2023, validate on 2024)
    # We do NOT use the 2025-2026 test set here to prevent look-ahead bias!
    df = df.sort_values(by=['date', 'Symbol']).copy()
    
    val_start = pd.Timestamp('2024-01-01')
    val_end = pd.Timestamp('2025-01-01')
    train_cutoff = val_start - pd.Timedelta(days=21)
    
    train_df = df[df['date'] <= train_cutoff]
    val_df = df[(df['date'] >= val_start) & (df['date'] < val_end)]
    
    X_train = train_df[features]
    y_train = train_df['target_quintile']
    qids_train = train_df['qid']
    
    X_val = val_df[features]
    y_val = val_df['target_quintile']
    qids_val = val_df['qid']

    # 2. Define the Optuna Objective Function
    def objective(trial):
        # Define the Search Space (Optuna will guess values within these ranges)
        param = {
            'tree_method': 'hist',
            'objective': 'rank:ndcg',
            'random_state': 42,
            # Let Optuna explore tree complexity
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            # Let Optuna explore learning speed
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            # Let Optuna explore the number of trees
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            # Let Optuna explore row and column sampling (prevents overfitting)
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            # Let Optuna explore regularization (penalizes overly complex trees)
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        }
        
        # Initialize and Train
        model = xgb.XGBRanker(**param)
        model.fit(X_train, y_train, qid=qids_train, verbose=False)
        
        # Predict on the Validation Set
        val_df_copy = val_df.copy()
        val_df_copy['pred_score'] = model.predict(X_val)
        
        # Calculate Average NDCG across all validation dates
        ndcg_scores = []
        for date, group in val_df_copy.groupby('date'):
            if len(group) > 1: # NDCG requires at least 2 items to rank
                # We want to see how well the predicted scores rank the actual target quintiles
                true_relevance = np.asarray([group['target_quintile'].values])
                predicted_scores = np.asarray([group['pred_score'].values])
                score = ndcg_score(true_relevance, predicted_scores)
                ndcg_scores.append(score)
                
        # Return the mean score for Optuna to maximize
        return np.mean(ndcg_scores)

    # 3. Create and run the Optuna Study
    print(f"Starting Optuna search for {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n--- Optuna Optimization Complete ---")
    print(f"Best Validation NDCG Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
        
    return study.best_params