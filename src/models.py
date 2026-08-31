import numpy as np
import pandas as pd
import os
from gplearn.genetic import SymbolicTransformer
import xgboost as xgb
from config import BASE_MODEL_PARAMS
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
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
    adequate_adtv: int = 2_500_000,
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
                adequate_adtv=adequate_adtv,
            )

        # qid assignment lives here, not in each model
        train_df["qid"] = train_df.groupby("date").ngroup()
        test_df["qid"]  = test_df.groupby("date").ngroup()

        if not test_df.empty:
            folds.append(Fold(fold_num, train_df, test_df, test_start, test_end))

        current_train_end = test_end
        fold_num += 1

    return folds
def _daily_precision_at_k(
    test_df: pd.DataFrame,
    top_k: int,
    return_col: str = "next_1m_ret",
    id_col: str = "Symbol",
) -> list[float]:
    """
    Per date: overlap between the model's top_k picks (by pred_score) and
    the date's ACTUAL top_k performers (by realized return_col) — same
    top_k, same yardstick, on both sides. This replaces the old version,
    which compared each model's top_k picks against that model's own
    eval_target_col "best group" — a group whose size differs wildly
    across models (target_bucket ≈ 20 stocks vs target_quintile ≈ 20% of
    the universe, e.g. ~100 stocks), making precision@k numbers look
    directly comparable across models when they weren't: a quintile-scored
    model's precision@20 partly reflects that its "top" bucket is ~5x
    easier to land in by chance, not that it's a better model.

    Basing it on realized forward return instead removes the dependency on
    any model-specific bucket/quintile scheme entirely, so precision@k is
    now apples-to-apples across NDCG / MSE / Linear.

    Dates with fewer than top_k rows, or where return_col is missing
    (e.g. alphaforge's test_df), are skipped/return an empty list.
    """
    if return_col not in test_df.columns:
        return []
    precisions = []
    for _, g in test_df.groupby("date"):
        if len(g) < top_k:
            continue
        predicted_top = set(g.nlargest(top_k, "pred_score")[id_col])
        actual_top    = set(g.nlargest(top_k, return_col)[id_col])
        precisions.append(len(predicted_top & actual_top) / top_k)
    return precisions

def _daily_topk_excess_return(
    test_df: pd.DataFrame,
    top_k: int,
    return_col: str = "next_1m_ret",
) -> list[float]:
    """
    Per date: mean forward return of the model's top_k picks minus the mean
    forward return of the full universe that day (equal-weighted). This is
    the closest proxy in this function to "would the top-k buy list have
    actually made money," independent of any bucketing/quintile scheme.

    Silently skipped if return_col isn't present (e.g. alphaforge's test_df).
    """
    if return_col not in test_df.columns:
        return []
    excess = []
    for _, g in test_df.groupby("date"):
        if len(g) < top_k:
            continue
        picks = g.nlargest(top_k, "pred_score")
        excess.append(picks[return_col].mean() - g[return_col].mean())
    return excess


def score_and_evaluate(
    test_df: pd.DataFrame,
    fold: Fold,
    total_folds: int,
    path_label: str,
    eval_target_col: str = "target_quintile",
    top_k: int = 20,
) -> tuple[pd.DataFrame, float]:
    """
    Assigns pred_quintile and computes mean daily NDCG, precision@top_k, and
    top_k excess return. Expects test_df to have a 'pred_score' column already set.

    eval_target_col: the column treated as ground-truth relevance for NDCG
    and for precision@top_k's "best group" check. Defaults to 'target_quintile'
    (used by mse/linear/alphaforge). Each model declares its own value via
    MODEL_REGISTRY / the alphaforge branch in walk_forward_cv, so this default
    only matters for direct/manual calls.

    top_k: size of the buy-only book precision@k and excess-return are
    evaluated against. Aggregate NDCG rewards getting the *whole* daily
    ranking right; precision@k and excess return only care about the exact
    slice you'd actually trade, which is what a top-k strategy lives or dies
    on — a model can look fine on NDCG while being mediocre right at the
    top_k/top_k+1 boundary.

    Returns
    -------
    (test_df, mean_ndcg) : the scored test_df (unchanged from before, plus
    'pred_quintile'), and this fold's mean daily NDCG as a float — the same
    number that was previously only printed, now also returned so callers
    (e.g. walk_forward_cv) can record it per fold instead of losing it.
    """
    test_df = test_df.copy()
    test_df["pred_quintile"] = test_df.groupby("date")["pred_score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    )

    daily_ndcg = [
        ndcg_score([g[eval_target_col].values], [g["pred_score"].values])
        for _, g in test_df.groupby("date")
        if len(g) > 1
    ]
    mean_ndcg = np.mean(daily_ndcg) if daily_ndcg else 0.0

    daily_precision = _daily_precision_at_k(test_df, top_k)
    mean_precision   = np.mean(daily_precision) if daily_precision else float("nan")

    daily_excess = _daily_topk_excess_return(test_df, top_k)
    mean_excess  = np.mean(daily_excess) if daily_excess else float("nan")

    excess_str = f", top{top_k} excess ret = {mean_excess:+.4%}" if daily_excess else ""
    print(
        f"Fold {fold.fold_number}/{total_folds} [{path_label}] "
        f"({fold.test_start.strftime('%Y-%m')}): NDCG = {mean_ndcg:.4f}, "
        f"precision@{top_k} = {mean_precision:.2%}{excess_str}"
    )
    return test_df, mean_ndcg

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
    adequate_adtv:int,
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
    This matches the denominator simulate_portfolio uses when splitting cash.
    The estimate is computed per date so universe-size changes across the fold
    are respected.

    Parameters
    ----------
    train_df / test_df   : Split DataFrames for this fold (must have 'close', 'volume').
    fold_number          : Used only for the diagnostic print.
    adequate_adtv        : Minimum ADTV threshold for inclusion.

    Returns
    -------
    (filtered_train_df, filtered_test_df) — temp ADTV columns are dropped.
    """
    def get_liquidity_mask(df_target):
        return df_target["adtv"].isna() | (df_target["adtv"] >= adequate_adtv)
    

    train_mask = get_liquidity_mask(train_df)
    test_mask = get_liquidity_mask(test_df)

    total_before = len(train_df) + len(test_df)
    total_after = train_mask.sum() + test_mask.sum()

    print(
        f"   💧 Fold {fold_number} liquidity filter (Static Optimized): "
        f"{total_before - total_after:,} rows removed "
        f"({(total_before - total_after) / total_before:.1%}), "
        f"{total_after:,} kept."
    )

    return train_df[train_mask].copy(), test_df[test_mask].copy()

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
def predict_xgboost_ndcg(
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
        train_df[features], train_df["target_magnitude"],
        qid=train_df["qid"],
        #eval_set=[(test_df[features], test_df["target_quintile"])],
        #eval_qid=[test_df["qid"]],
        verbose=False,
    )
    test_df["pred_score"] = ranker.predict(test_df[features])
    return test_df

def predict_xgboost_mse(
    fold: Fold,
    features: list[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """
    XGBRanker baseline. GP columns (if any) should already be included in
    `features` before calling — this function is unaware of GP.

    Returns fold.test_df with 'pred_score' column set.
    """
    params = (model_params or BASE_MODEL_PARAMS).copy()
    params['objective'] = 'reg:squarederror'
    train_df = fold.train_df.sort_values("qid").reset_index(drop=True)
    test_df  = fold.test_df.copy().sort_values("qid").reset_index(drop=True)

    model = xgb.XGBRegressor(**params)
    model.fit(
        train_df[features], train_df["target_magnitude"],
        #qid=train_df["qid"],
        #eval_set=[(test_df[features], test_df["target_quintile"])],
        #eval_qid=[test_df["qid"]],
        verbose=False,
    )
    test_df["pred_score"] = model.predict(test_df[features])
    return test_df


def predict_linear(
    fold: Fold,
    features: list[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """
    Linear regression baseline. GP columns (if any) should already be included in
    `features` before calling — this function is unaware of GP.

    Returns fold.test_df with 'pred_score' column set.
    """
    train_df = fold.train_df.sort_values("qid").reset_index(drop=True)
    test_df  = fold.test_df.copy().sort_values("qid").reset_index(drop=True)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[features])
    test_features = scaler.transform(test_df[features])

    model = LassoCV(cv=5, random_state=42)
    model.fit(train_features, train_df["target_magnitude"])
    test_df["pred_score"] = model.predict(test_features)
    return test_df

def predict_lstm_mse(
    fold: Fold,
    features: list[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """
    LSTM regression baseline (trained with MSE loss).

    Unlike the tree/linear baselines, an LSTM needs a *sequence* of
    per-symbol history to produce one prediction, so this function
    reshapes the panel into (num_samples, seq_len, num_features) sliding
    windows per Symbol before handing off to a small PyTorch
    nn.LSTM + Linear head trained with MSELoss.

    Sequence construction (zero look-ahead):
      - Training windows are built purely from fold.train_df, sorted by
        date within each Symbol. A window's target is target_magnitude
        at the window's LAST row — no test-window data is ever touched.
      - Test predictions for a given test row use the seq_len-1 rows
        immediately preceding it (trailing train history the first time
        a symbol appears, then its own earlier test rows), so nothing
        from the future leaks into a prediction.
      - Symbols with fewer than seq_len rows of trailing history (e.g.
        brand-new listings) can't form a full window; their pred_score
        falls back to the median training prediction as a neutral value.

    model_params (all optional; read from a plain dict for consistency
    with the xgboost_*/linear signatures):
      seq_len      : int   — timesteps per window (default 10)
      hidden_size  : int   — LSTM hidden units (default 32)
      num_layers   : int   — stacked LSTM layers (default 1)
      dropout      : float — dropout between LSTM layers (default 0.0)
      epochs       : int   — training epochs (default 20)
      batch_size   : int   — default 256
      lr           : float — Adam learning rate (default 1e-3)
      device       : str   — "cpu" or "cuda" (default: cuda if available)

    Returns fold.test_df with 'pred_score' column set.
    """
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    params      = model_params or {}
    seq_len     = params.get("seq_len", 10)
    hidden_size = params.get("hidden_size", 32)
    num_layers  = params.get("num_layers", 1)
    dropout     = params.get("dropout", 0.0)
    epochs      = params.get("epochs", 20)
    batch_size  = params.get("batch_size", 256)
    lr          = params.get("lr", 1e-3)
    device      = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    target_col = "target_magnitude"

    train_df = fold.train_df.sort_values(["Symbol", "date"]).reset_index(drop=True)
    test_df  = fold.test_df.copy().sort_values(["Symbol", "date"]).reset_index(drop=True)

    # --- Scale features on train only (no look-ahead) ---
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    def _windows_from_group(feat_arr, target_arr):
        """Sliding windows of length seq_len from one symbol's
        chronologically-sorted feature/target arrays."""
        n = len(feat_arr)
        if n < seq_len:
            return [], []
        Xs, ys = [], []
        for end in range(seq_len, n + 1):
            Xs.append(feat_arr[end - seq_len:end])
            ys.append(target_arr[end - 1])
        return Xs, ys

    # --- Build training windows, one symbol at a time ---
    X_train, y_train = [], []
    for _, g in train_df.groupby("Symbol"):
        feat_arr   = scaler.transform(g[features])
        target_arr = g[target_col].values
        Xs, ys = _windows_from_group(feat_arr, target_arr)
        X_train.extend(Xs)
        y_train.extend(ys)

    if not X_train:
        # Not enough per-symbol history anywhere in this fold — fall back
        # to a neutral prediction rather than crashing the CV loop.
        test_df["pred_score"] = 0.0
        return test_df

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    finite_mask = np.isfinite(X_train).all(axis=(1, 2)) & np.isfinite(y_train)
    X_train, y_train = X_train[finite_mask], y_train[finite_mask]
    if len(X_train) == 0:
        test_df["pred_score"] = 0.0
        return test_df

    class _LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    torch.manual_seed(42)
    model     = _LSTMRegressor(len(features), hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # --- Test predictions: walk each symbol's rows in date order, seeding
    # the window with its trailing train-history so every test row gets a
    # full seq_len window without ever looking at a future row ---
    model.eval()
    preds = np.full(len(test_df), np.nan, dtype=np.float32)
    train_by_symbol = {sym: g.sort_values("date") for sym, g in train_df.groupby("Symbol")}

    with torch.no_grad():
        for sym, g in test_df.groupby("Symbol"):
            g = g.sort_values("date")
            hist = train_by_symbol.get(sym)
            if hist is not None and len(hist) > 0:
                window = list(scaler.transform(hist[features]))[-(seq_len - 1):]
            else:
                window = []

            for idx, row in g.iterrows():
                row_feat = scaler.transform(row[features].to_frame().T)[0]
                window.append(row_feat)
                window = window[-seq_len:]
                if len(window) == seq_len:
                    x = torch.from_numpy(
                        np.asarray(window, dtype=np.float32)
                    ).unsqueeze(0).to(device)
                    preds[test_df.index.get_loc(idx)] = model(x).item()

    # Rows that never accumulated a full window (e.g. brand-new symbols
    # with no trailing history) fall back to the median training prediction.
    # NOTE: X_train grows every fold (expanding walk-forward window), so
    # this must run through the model in mini-batches — a single forward
    # pass over the whole array can blow up GPU memory in later folds.
    if np.isnan(preds).any():
        train_preds_chunks = []
        with torch.no_grad():
            for i in range(0, len(X_train), batch_size):
                xb = torch.from_numpy(X_train[i:i + batch_size]).to(device)
                train_preds_chunks.append(model(xb).cpu().numpy())
        train_preds = np.concatenate(train_preds_chunks) if train_preds_chunks else np.array([])
        fallback = float(np.median(train_preds)) if len(train_preds) else 0.0
        preds = np.where(np.isnan(preds), fallback, preds)

    test_df["pred_score"] = preds

    # Release GPU memory before the next (larger) fold trains a fresh model.
    # The reserved-but-unallocated fragmentation PyTorch's CUDA allocator
    # accumulates across folds is exactly what the OOM traceback flagged.
    del model, optimizer, train_loader
    if device == "cuda":
        torch.cuda.empty_cache()

    return test_df


@dataclass
class ModelSpec:
    """
    Declares, per model, which column NDCG/scoring should treat as ground
    truth relevance (eval_target_col). Training targets are left inside each
    predict_* function for now (they aren't diverging yet) — only eval_target_col
    is centralized here, since that's the thing walk_forward_cv currently
    hardcodes per-call instead of per-model.

    To make a model's *training* target configurable too, later, add a
    train_target_col field here and thread it into the corresponding
    predict_* function instead of that function's hardcoded literal.
    """
    name: str
    predict_fn: callable
    eval_target_col: str = "target_quintile"


# Single source of truth for "which model scores against which column".
# xgboost_ndcg is the only one that diverges today (trains AND scores on
# target_bucket); mse/linear/alphaforge all score on target_quintile.
# alphaforge is intentionally NOT here — its predict_fn takes an alpha_pool
# argument instead of (fold, features, model_params), so it keeps its own
# branch in walk_forward_cv below.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost_ndcg": ModelSpec("XGBoost (NDCG)",    predict_xgboost_ndcg, eval_target_col="target_magnitude"),
    "xgboost_mse":  ModelSpec("XGBoost (MSE)",     predict_xgboost_mse,  eval_target_col="target_magnitude"),
    "linear":       ModelSpec("Linear Regression", predict_linear,       eval_target_col="target_magnitude"),
    "lstm_mse":     ModelSpec("LSTM (MSE)",        predict_lstm_mse,     eval_target_col="target_magnitude"),
}


def walk_forward_cv(
    df: pd.DataFrame,
    features: list[str],
    model: str = "xgboost_ndcg",          # "xgboost" | "alphaforge"
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
    adequate_adtv: int = 2_500_000,
    #Cannot skip features
    nonskip_features: list[str] | None = None,
    # --- Buy-only book size for precision@k / excess-return diagnostics ---
    top_k: int = 20,
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

    Returns
    -------
    (final_df, fold_ndcg_df) : tuple[pd.DataFrame, pd.DataFrame]
        final_df     — unchanged: every fold's scored test_df, concatenated.
        fold_ndcg_df — NEW: one row per fold with columns
                       [fold_index, test_start, test_end, ndcg], for plotting
                       NDCG across the walk-forward timeline or comparing
                       across multiple runs (e.g. a threshold sweep).
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
        adequate_adtv=adequate_adtv,
    )
    total_folds  = len(folds)
    results      = []
    fold_records = []  # one dict per fold: fold_index, test_start, test_end, ndcg

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
        fold_features = fold_features + [f for f in (nonskip_features or []) if f not in fold_features]
        if model == "alphaforge":
            # Kept separate: predict_alphaforge's signature (fold, alpha_pool, ...)
            # doesn't match the (fold, features, model_params) shape every
            # MODEL_REGISTRY entry uses, so it can't sit in the registry as-is.
            test_df         = predict_alphaforge(fold, alpha_pool)
            path_label      = "AlphaForge (Mega)"
            eval_target_col = "target_quintile"
        else:
            spec = MODEL_REGISTRY.get(model)
            if spec is None:
                raise ValueError(f"Unknown model: {model!r}")
            test_df         = spec.predict_fn(fold, fold_features, model_params)
            path_label      = spec.name
            eval_target_col = spec.eval_target_col
        # elif model == "lightgbm":
        #     test_df = predict_lightgbm(fold, fold_features, model_params)

        test_df, fold_ndcg = score_and_evaluate(
            test_df, fold, total_folds, path_label,
            eval_target_col=eval_target_col, top_k=top_k,
        )
        results.append(test_df)

        # One row per fold: what you need to plot NDCG across the walk-forward
        # timeline (or across multiple runs at different icir_threshold values).
        fold_records.append({
            "fold_index": fold.fold_number,
            "test_start": fold.test_start,
            "test_end":   fold.test_end,
            "ndcg":       fold_ndcg,
        })

        gc.collect()

    final_df = pd.concat(results)
    overall_ndcg = np.mean([
        ndcg_score([g[eval_target_col].values], [g["pred_score"].values])
        for _, g in final_df.groupby("date") if len(g) > 1
    ])
    overall_precision = np.mean(_daily_precision_at_k(final_df, top_k))
    overall_excess_list = _daily_topk_excess_return(final_df, top_k)

    print(f"📊 Overall NDCG (all folds, all dates): {overall_ndcg:.4f}")
    print(f"📊 Overall precision@{top_k} (all folds, all dates): {overall_precision:.2%}")
    if overall_excess_list:
        print(f"📊 Overall top{top_k} excess return vs universe (all folds, all dates): "
              f"{np.mean(overall_excess_list):+.4%}")

    print("\n✅ Walk-forward CV complete.")

    # One row per fold — test_start, test_end, fold_index, ndcg — ready to
    # plot directly (e.g. x=test_start or x=fold_index, y=ndcg), or to
    # concatenate across multiple icir_threshold sweep runs by adding your
    # own "icir_threshold" column to the result before combining.
    fold_ndcg_df = pd.DataFrame(fold_records)

    return final_df, fold_ndcg_df