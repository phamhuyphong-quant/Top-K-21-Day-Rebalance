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
from src.features import seed_everything
seed_everything(42)
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

    Liquidity filter (optional, mirrors simulate_portfolio)
    -------------------------------------------------------
    When liquidity_filter=True, the ADTV filter is applied *per fold*
    inside generate_folds(), not upfront on the whole dataframe.

    This means:
      - A stock illiquid in fold N can still appear in fold N+k once its
        volume recovers (IPO lock-up expiry, re-listing, sustained growth).
      - ADTV for each fold is computed from history available up to that
        fold's dates only — no future volume leaks backward.
      - The initial training window is also filtered, so the model is
        never trained on stocks it couldn't have traded at that time.

    Pass the same capital / buy_fraction / adtv_participation values
    you use in simulate_portfolio so the filter is identical in both places.

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
        if model == "alphaforge":
            test_df    = predict_alphaforge(fold, alpha_pool)
            path_label = "AlphaForge (Mega)"
        elif model == "xgboost":
            xgb_features = list(features) + gp_cols
            test_df      = predict_xgboost(fold, xgb_features, model_params)
            path_label   = "XGBoost"
        # elif model == "lightgbm":
        #     test_df = predict_lightgbm(fold, features, model_params)
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