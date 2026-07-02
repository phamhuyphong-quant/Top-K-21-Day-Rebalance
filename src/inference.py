import pandas as pd
import xgboost as xgb
import sys,os
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
    target_col: str = 'target_quintile',
    icir_filter: bool = False,
    icir_threshold: float = 0.02,
    icir_target_col: str = 'next_1m_ret',
    corr_prune: bool = False,
    corr_threshold: float = 0.75,
):
    """
    Generates Buy, Hold, and Sell signals mirroring walk-forward logic.
    Applies a grace band (hold_n) for existing positions and a trend filter for new buys.
    
    Parameters:
    - df: Full DataFrame containing features, dates, and the VN universe.
    - current_portfolio: List of stock symbols currently held (e.g., ['VNM', 'FPT']).
    - features: List of feature column names.
    - use_mega: Boolean flag to switch between XGBoost (False) and LSTM (True).
    - buy_n: Top N stocks targeted for new entries. Also used as the hold threshold — existing positions are kept as long as they rank within the top buy_n.
    - trend_filter_col: Column name for the trend filter (e.g., dist_SMA_50).
    - trend_filter_threshold: Stock must have a trend value > this to be bought.
    - target_col: The column used for training the ranker.
    
    Returns:
    - buy_list: Symbols to buy.
    - hold_list: Symbols to keep holding.
    - sell_list: Symbols to sell.
    - not_in_universe_list: Symbols held but no longer in today's VN dataset.
    - ranked_today: DataFrame containing today's scores and ranks.
    """
    
    # 1. Identify "Today"
    latest_date = df['date'].max()
    
    # 2. Split Data
    train_df = df[(df['date'] < latest_date) & (df[target_col].notna())].copy()
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
    # 3. Model Training & Scoring
    if not use_mega:
        if model is not None:
            print(f"Using pretrained model for inference on {latest_date.date()}...")
        else:
            print(f"Training Baseline XGBoost up to {train_df['date'].max().date()}...")
            X_train = train_df[features]
            y_train = train_df[target_col]
            qid_train = train_df['qid']
            model = xgb.XGBRanker(**BASE_MODEL_PARAMS)
            model.fit(X_train, y_train, qid=qid_train)
        
        X_inference = inference_df[features]
        inference_df['live_score'] = model.predict(X_inference)
        
    else:
        raise NotImplementedError(
        "use_mega=True (LSTM path) is not yet implemented in generate_paper_trade_signals."
    )

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
            continue  # Skip further rank checking, move to the next stock
            
        stock_rank = rank_dict.get(sym)
        
        # We technically won't hit this None check often now because we already filtered 
        # out symbols not in current_universe, but it's good defensive programming.
        if stock_rank is None:
            sell_list.append(sym)
        elif stock_rank > buy_n:
            # Fell out of the grace band (e.g., ranked 16th, threshold is 15)
            sell_list.append(sym)
        else:
            # Still in the top hold_n, keep it
            hold_list.append(sym)

    # 6. Determine Target Buys (Based on top_n and trend filter)
    top_candidates = ranked_today.head(buy_n)['Symbol'].tolist()
    
    for sym in top_candidates:
        # Skip if we already own it (it's already in the hold_list)
        if sym in current_portfolio:
            continue
            
        # Apply the trend filter (e.g., must be trading above its 50-day SMA)
        if has_trend_filter:
            trend_val = trend_dict.get(sym, 0)
            if trend_val <= trend_filter_threshold:
                continue # Fails the trend filter, skip buying
                
        # If it passes all checks, it's a valid new buy
        buy_list.append(sym)
        
    # 7. Print Summary
    print(f"\n--- Paper Trading Signals for {latest_date.date()} ---")
    print(f"Strategy: {'LSTM (Mega)' if use_mega else 'XGBoost (Baseline)'}")
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