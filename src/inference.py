import pandas as pd
import xgboost as xgb
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_MODEL_PARAMS
from src.models import select_features_by_icir, prune_correlated_features, _compute_icir_map


def generate_paper_trade_signals(
    df: pd.DataFrame,
    current_portfolio: list,
    features: list,
    use_mega: bool = False,
    model=None,
    buy_n: int = 30,
    trend_filter_col: str = 'dist_SMA_100',
    trend_filter_threshold: float = 1.0,
    target_col: str = 'target_magnitude',
    icir_filter: bool = False,
    icir_threshold: float = 0.02,
    icir_target_col: str = 'next_1m_ret',
    corr_prune: bool = False,
    corr_threshold: float = 0.75,
    nonskip_features: list | None = None,
):
    """
    Generates Buy, Hold, and Sell signals.
    Applies a grace band for existing positions (kept while rank <= buy_n)
    and a trend filter for new buys.

    Parameters:
    - df: Full DataFrame containing features, dates, and the VN universe.
    - current_portfolio: List of stock symbols currently held (e.g., ['VNM', 'FPT']).
    - features: List of feature column names.
    - use_mega: Not implemented yet; raises NotImplementedError if True.
    - model: Optional pretrained XGBRanker. If given, its own feature names are
      used and icir_filter/corr_prune/nonskip_features are not applied.
    - buy_n: Top N stocks targeted for new entries. Also the hold threshold:
      existing positions are kept as long as they rank within the top buy_n.
    - trend_filter_col: Column name for the trend filter (e.g., dist_SMA_100).
    - trend_filter_threshold: Stock must have a trend value > this to be bought.
    - target_col: The column used for training the ranker.
    - nonskip_features: Features always appended after IC/IR and correlation
      filtering (only when training a new model).

    Returns:
    - buy_list: Symbols to buy.
    - hold_list: Symbols to keep holding.
    - sell_list: Symbols to sell.
    - not_in_universe_list: Symbols held but no longer in today's VN dataset.
    - ranked_today: DataFrame containing today's scores and ranks.
    """
    if use_mega:
        raise NotImplementedError("use_mega=True is not yet implemented.")

    if model is not None:
        if icir_filter or corr_prune:
            raise ValueError("icir_filter/corr_prune can't be used with a pretrained model.")
        model_features = model.get_booster().feature_names
        if model_features is None:
            raise ValueError(
                "Pretrained model has no feature names (trained on a numpy array?). "
                "Retrain on a DataFrame or pass features explicitly."
            )
        features = list(model_features)

    # 1. Identify "Today"
    latest_date = df['date'].max()

    # 2. Split Data
    LOOKBACK_DAYS = 735

    all_dates = sorted(df.loc[df['date'] < latest_date, 'date'].unique())
    window_dates = all_dates[-LOOKBACK_DAYS:]  # last N trading dates before today
    if not window_dates:
        raise ValueError(f"No training history before {latest_date}.")
    cutoff_date = window_dates[0]

    train_df = df[
        (df['date'] < latest_date) &
        (df['date'] >= cutoff_date) &
        (df[target_col].notna())
    ].copy()
    inference_df = df[df['date'] == latest_date].copy()

    if inference_df.empty:
        raise ValueError(f"No data available for inference on {latest_date}")

    current_universe = inference_df['Symbol'].unique().tolist()

    _ic_ir_map: dict = {}

    if icir_filter:
        features, _ic_ir_map = select_features_by_icir(
            train_df, features,
            icir_threshold=icir_threshold,
            target_col=icir_target_col,
        )
        print(f"   🔍 IC/IR filter: {len(features)} features selected.")

    if corr_prune and len(features) > 1:
        from config import FEATURE_GROUPS
        # Reuse the ic_ir_map already computed by the icir filter if available;
        # otherwise compute it now (corr_prune=True but icir_filter=False).
        if not _ic_ir_map:
            _ic_ir_map = _compute_icir_map(train_df, features, icir_target_col)
        features = prune_correlated_features(
            df=train_df,
            candidate_features=features,
            ic_ir_map=_ic_ir_map,
            feature_groups=FEATURE_GROUPS,
            correlation_threshold=corr_threshold,
        ) or features[:1]
        print(f"   ✂️  Correlation pruning: {len(features)} features kept (threshold={corr_threshold}).")

    # Only when training a new model: a pretrained model's feature set is fixed.
    if model is None:
        features = features + [f for f in (nonskip_features or []) if f not in features]

    # 3. Model Training & Scoring
    if model is not None:
        print(f"Using pretrained model for inference on {latest_date.date()}...")
    else:
        print(f"Training Baseline XGBoost up to {train_df['date'].max().date()}...")
        train_df = train_df.sort_values(['date', 'Symbol']).reset_index(drop=True)
        train_df['qid'] = train_df.groupby('date').ngroup()   # contiguous, 0..n_dates-1

        model = xgb.XGBRanker(**BASE_MODEL_PARAMS)
        model.fit(
            train_df[features], train_df[target_col],
            qid=train_df['qid'],
            verbose=False,
        )

    inference_df['live_score'] = model.predict(inference_df[features])

    # 4. Rank Today's Stocks
    ranked_today = inference_df.sort_values(by='live_score', ascending=False).copy()
    # Explicitly calculate integer ranks (1 = best score)
    ranked_today['rank'] = ranked_today['live_score'].rank(ascending=False, method='first').astype(int)

    # Create quick lookups for our logic
    rank_dict = dict(zip(ranked_today['Symbol'], ranked_today['rank']))

    has_trend_filter = trend_filter_col and trend_filter_col in ranked_today.columns
    if has_trend_filter:
        trend_dict = dict(zip(ranked_today['Symbol'], ranked_today[trend_filter_col]))
    else:
        trend_dict = {}

    sell_list = []
    hold_list = []
    buy_list = []
    not_in_universe_list = []

    # 5. Evaluate Current Portfolio (SELL vs HOLD based on grace band)
    for sym in current_portfolio:
        if sym not in current_universe:
            not_in_universe_list.append(sym)
            continue

        stock_rank = rank_dict.get(sym)

        if stock_rank is None:
            sell_list.append(sym)
        elif stock_rank > buy_n:
            # Fell out of the grace band (rank worse than buy_n)
            sell_list.append(sym)
        else:
            # Still within the top buy_n, keep it
            hold_list.append(sym)

    # 6. Determine Target Buys (top buy_n + trend filter)
    top_candidates = ranked_today.head(buy_n)['Symbol'].tolist()

    for sym in top_candidates:
        # Skip if we already own it (it's already in the hold_list)
        if sym in current_portfolio:
            continue

        # Apply the trend filter (e.g., must be trading above its 100-day SMA)
        if has_trend_filter:
            trend_val = trend_dict.get(sym, 0)
            if trend_val <= trend_filter_threshold:
                continue  # Fails the trend filter, skip buying

        # If it passes all checks, it's a valid new buy
        buy_list.append(sym)

    # 7. Print Summary
    print(f"\n--- Paper Trading Signals for {latest_date.date()} ---")
    print("Strategy: XGBoost (Baseline)")
    print(f"SELL ({len(sell_list)}): {sell_list}")
    print(f"HOLD ({len(hold_list)}): {hold_list}")
    print(f"BUY  ({len(buy_list)}): {buy_list} (Strict top <= {buy_n} + Trend > {trend_filter_threshold})")
    if not_in_universe_list:
        print(f"NOT IN UNIVERSE ({len(not_in_universe_list)}): {not_in_universe_list} (Held but missing from today's data)")

    return buy_list, hold_list, sell_list, not_in_universe_list, ranked_today[['Symbol', 'live_score', 'rank']]


def get_actionable_portfolio_lists(
    df: pd.DataFrame,
    current_portfolio: list,
    features: list,
    **kwargs
) -> dict:
    """
    Wrapper function that calls generate_paper_trade_signals and formats
    the output into a clean dictionary for the user or UI.
    """
    buys, holds, sells, non_universe, ranks_df = generate_paper_trade_signals(
        df=df,
        current_portfolio=current_portfolio,
        features=features,
        **kwargs
    )

    return {
        "BUY": buys,
        "HOLD": holds,
        "SELL": sells,
        "NOT_IN_UNIVERSE": non_universe,
        "Rankings": ranks_df
    }