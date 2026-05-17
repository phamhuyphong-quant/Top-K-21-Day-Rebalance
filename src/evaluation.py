import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import xgboost as xgb
import numpy as np
from scipy.stats import spearmanr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import seed_everything
from src.models import base_model
seed_everything(42)
from src.models import walk_forward_cv
def plot_feature_importances(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', title='What drives the ranking?')
    plt.show()

def compute_model_ic(test_df, pred_col='pred_score', ret_col='next_1m_ret'):
    daily_ic = test_df.groupby('date').apply(
        lambda x: spearmanr(x[pred_col], x[ret_col]).statistic,
        include_groups=False
    )

    ic_mean = daily_ic.mean()
    ic_std  = daily_ic.std()
    ic_ir   = ic_mean / ic_std if ic_std > 0 else 0

    print(f"IC Mean : {ic_mean:.4f}   (target: > 0.05)")
    print(f"IC Std  : {ic_std:.4f}")
    print(f"IC IR   : {ic_ir:.4f}   (target: > 0.5)")

    return {'ic_mean': ic_mean, 'ic_std': ic_std, 'ic_ir': ic_ir, 'daily_ic': daily_ic}

def compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col='next_1m_ret'):
    """
    Evaluates the ranker. Supports two modes:
    1. Static mode: pass X_test and ranker to score predictions on the fly.
    2. Dynamic mode: leave X_test and ranker as None to read 'pred_score' from test_df (output of walk_forward_cv).
    """
    eval_df = test_df.copy()
    
    # --- Determine the source of scores ---
    if ranker is not None and X_test is not None:
        # Static mode: run predictions now
        eval_df['score'] = ranker.predict(X_test)
    elif 'pred_score' in eval_df.columns:
        # Dynamic mode: use OOS scores from walk-forward CV
        eval_df['score'] = eval_df['pred_score']
    elif 'score' in eval_df.columns:
        pass  # Score column already present, nothing to do
    else:
        raise ValueError("No score column found. Pass X_test and ranker, or provide a DataFrame with a 'pred_score' column.")
    
    # Identify top picks per date using the predicted score
    score_rank_pct = eval_df.groupby('date')['score'].rank(ascending=False, pct=True)
    top_picks = eval_df[score_rank_pct <= top_quantile]
    
    # Calculate Win Rates
    top_win_rate = (top_picks[ret_col] > 0).mean()
    market_win_rate = (eval_df[ret_col] > 0).mean()
    lift = top_win_rate - market_win_rate
    
    print(f"--- Backtest Results (Top {top_quantile*100:.0f}% Picks) ---")
    print(f"Top Picks Win Rate:   {top_win_rate:.2%}")
    print(f"Market Baseline:      {market_win_rate:.2%}")
    print(f"Excess Win Rate:      {lift:+.2%}")
    
    return {
        'top_win_rate': top_win_rate,
        'market_win_rate': market_win_rate,
        'lift': lift,
        'top_picks_df': top_picks
    }

def plot_feature_ic(test_df, features, target_col='next_1m_ret'):
    """
    Evaluates feature predictive power via Information Coefficient (IC).
    Computes Spearman rank correlation between each feature and the forward return target.
    """
    ic_dict = {}
    for feat in features:
        # Spearman correlation between the feature and the future return
        ic = test_df[feat].corr(test_df[target_col], method='spearman')
        ic_dict[feat] = ic
        
    # Sort and plot
    ic_series = pd.Series(ic_dict).sort_values()
    
    plt.figure(figsize=(10, 6))
    # Blue for positive correlation, red for negative
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in ic_series]
    
    ic_series.plot(kind='barh', color=colors, edgecolor='black')
    plt.title('Feature Influence via Information Coefficient (OOS Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Spearman Rank Correlation (IC)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def plot_feature_ir(test_df,features,target_col='next_1m_ret'):
    #Calculates IC Mean, IC Std, and IC IR for each feature.
    #IC IR = Mean(Daily IC) / Std(Daily IC)

    ic_ir_results = {}

    for feat in features:
        # 1. Calculate IC for each date (Time-series of ICs)
        # We group by 'date' and correlate the feature with the target for that specific day
        daily_ic = test_df.groupby('date').apply(
    lambda x: x[feat].corr(x[target_col], method='spearman'),
    include_groups=False
)
        
        # 2. Calculate Metrics
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        
        # Avoid division by zero if std is 0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        ic_ir_results[feat] = {
            'IC Mean': ic_mean,
            'IC Std': ic_std,
            'IC IR': ic_ir
        }

    # Convert to DataFrame for easy viewing and plotting
    ir_df = pd.DataFrame(ic_ir_results).T.sort_values(by='IC IR', ascending=True)
    
    # Plotting IC IR
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in ir_df['IC IR']]
    ir_df['IC IR'].plot(kind='barh', color=colors, edgecolor='black')
    
    plt.title('Feature Consistency (IC IR)', fontsize=14, fontweight='bold')
    plt.xlabel('IC IR (Mean IC / Std IC)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return ir_df

def plot_feature_rolling_ir(test_df, feature, target_col='next_1m_ret', window=6):
    # Calculate monthly ICs
    monthly_ic = test_df.groupby(test_df['date'].dt.to_period('M')).apply(
    lambda x: x[feature].corr(x[target_col], method='spearman'),
    include_groups=False
)
    
    # Calculate rolling IR
    rolling_ir = monthly_ic.rolling(window=window).mean() / monthly_ic.rolling(window=window).std()
    
    rolling_ir.plot(title=f'Rolling {window}-Month IC IR for {feature}')
    plt.axhline(0, color='black', linestyle='--')
    plt.show()
def plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None):
    """
    Plots average forward return by predicted quintile.
    Supports both static mode (pass ranker) and dynamic mode (read pred_score from test_df).
    """
    df_plot = test_df.copy()
    
    # --- Determine the source of scores ---
    if ranker is not None and X_test is not None:
        df_plot['score'] = ranker.predict(X_test)
    elif 'pred_score' in df_plot.columns:
        df_plot['score'] = df_plot['pred_score']
    elif 'score' in df_plot.columns:
        pass
    else:
        raise ValueError("No score column found. Pass X_test and ranker, or provide a DataFrame with a 'pred_score' column.")
    
    # Use rank(method='first') to handle ties; label quintiles 1–5 for the x-axis
    df_plot['pred_quintile'] = df_plot.groupby('date')['score'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    )

    performance = df_plot.groupby('pred_quintile')['next_1m_ret'].mean()
    
    plt.figure(figsize=(8, 5))
    performance.plot(kind='bar', title='Future Return by Predicted Quintile', color='#1f77b4', edgecolor='black')
    plt.ylabel('Average 1-Month Forward Return')
    plt.xlabel('Quintile (1 = Worst, 5 = Best)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_equity_curves(*results, labels=None, normalize=False):
    plt.figure(figsize=(12, 6))

    for i, result in enumerate(results):
        result = result.copy()
        result['date'] = pd.to_datetime(result['date'])
        label = labels[i] if labels and i < len(labels) else f'Strategy {i+1}'

        y = result['total_value']
        if normalize:
            y = (y / y.iloc[0] - 1) * 100

        plt.plot(result['date'], y, marker='o', linewidth=2, label=label)

    plt.title('Equity Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)' if normalize else 'Total Value (VND)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    
def print_performance_report(result, initial_capital=None, rf_annual=0.045):
    """
    Computes and prints risk-adjusted performance metrics from a backtest result DataFrame.
    
    Parameters
    ----------
    result          : DataFrame returned by simulate_portfolio (columns: date, total_value)
    initial_capital : Starting capital. If None, uses result['total_value'].iloc[0]
    rf_annual       : Annual risk-free rate. Default 4.5% (approx Vietnam T-bill rate)
    """
    import numpy as np

    nav = result['total_value'].copy()
    
    if initial_capital is None:
        initial_capital = nav.iloc[0]

    # --- Monthly returns (your backtest rebalances monthly) ---
    monthly_returns = nav.pct_change().dropna()

    # --- Core metrics ---
    n_months = len(monthly_returns)
    n_years  = n_months / 12

    total_return = (nav.iloc[-1] / initial_capital) - 1
    cagr         = (nav.iloc[-1] / initial_capital) ** (1 / n_years) - 1

    # Annualized Sharpe (monthly rf = annual rf / 12)
    rf_monthly   = rf_annual / 12
    excess_ret   = monthly_returns - rf_monthly
    sharpe       = (excess_ret.mean() / excess_ret.std()) * np.sqrt(12)

    # Sortino — only penalizes downside volatility
    downside     = monthly_returns[monthly_returns < rf_monthly]
    downside_std = downside.std() * np.sqrt(12)
    sortino      = (cagr - rf_annual) / downside_std if downside_std > 0 else np.nan

    # Max drawdown
    rolling_max  = nav.cummax()
    drawdown     = (nav - rolling_max) / rolling_max
    max_dd       = drawdown.min()

    # Calmar = CAGR / abs(Max Drawdown)
    calmar       = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate (months with positive return)
    win_rate     = (monthly_returns > 0).mean()

    # Profit factor = sum of gains / sum of losses
    gains        = monthly_returns[monthly_returns > 0].sum()
    losses       = abs(monthly_returns[monthly_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.nan

    # --- Print ---
    print("=" * 40)
    print("       STRATEGY PERFORMANCE REPORT")
    print("=" * 40)
    print(f"  Period          : {result['date'].iloc[0].strftime('%Y-%m')} → {result['date'].iloc[-1].strftime('%Y-%m')} ({n_months} months)")
    print(f"  Total Return    : {total_return:>+.2%}")
    print(f"  CAGR            : {cagr:>+.2%}")
    print("-" * 40)
    print(f"  Sharpe Ratio    : {sharpe:>6.2f}   (>1 good, >2 great)")
    print(f"  Sortino Ratio   : {sortino:>6.2f}   (like Sharpe, downside only)")
    print(f"  Calmar Ratio    : {calmar:>6.2f}   (CAGR / Max Drawdown)")
    print("-" * 40)
    print(f"  Max Drawdown    : {max_dd:>+.2%}")
    print(f"  Monthly Win Rate: {win_rate:>6.2%}")
    print(f"  Profit Factor   : {profit_factor:>6.2f}   (gains / losses)")
    print("=" * 40)

    return {
        'total_return'  : total_return,
        'cagr'          : cagr,
        'sharpe'        : sharpe,
        'sortino'       : sortino,
        'calmar'        : calmar,
        'max_drawdown'  : max_dd,
        'win_rate'      : win_rate,
        'profit_factor' : profit_factor,
    }


def simulate_portfolio(
    df,
    model,
    features,
    initial_capital=10000,
    buy_fraction=0.05,
    time_of_rebalance='M',
    trend_filter_col='dist_SMA_100',
    settlement_delay=3,
    vnindex_df=None,
    vol_lookback=21,          # days for realized vol calculation
    vol_percentile=0.80,      # percentile threshold — above this = high vol = skip
    vol_window=252,
):
    """
    Simulates a portfolio with the correct VN-market timing:

        REBALANCE MORNING (first trading day of month)
        ├── Step 1: Model predicts scores → ranks all stocks
        ├── Step 2: SELL — stocks that no longer qualify exit at today's price
        │           └── proceeds go into pending_cash (available after T+settlement_delay)
        └── Step 3: Schedule a DEFERRED BUY on the settlement date
                    └── On that date, use ALL available cash (settled proceeds + any
                        existing free cash) to buy the new target stocks at that day's price

    This means:
      - Sells happen on rebalance morning (day 0)
      - Cash is available on day +settlement_delay (T+3 by default)
      - Buys execute on that settlement date at settlement-date prices
      - No free-riding: you cannot buy with money from the same-day sell

    Parameters
    ----------
    df               : DataFrame with columns [date, Symbol, close, pred_score, ...]
    model            : trained XGBRanker; pass None to use existing 'pred_score' column
    features         : feature columns for model.predict() (ignored when model=None)
    initial_capital  : starting cash
    buy_fraction     : top X% of ranked stocks are buy targets
    hold_fraction    : top Y% of ranked stocks are hold targets (grace band, Y > X)
    trailing_stop    : sell if drawdown from peak <= this (e.g. -0.10 = -10%)
    take_profit      : sell if return since buy >= this (e.g. 0.50 = +50%)
    time_of_rebalance: pandas period alias ('M' = monthly, 'W' = weekly)
    trend_filter_col : column name for trend filter; stock must have value > 1.0 to buy.
                       Pass None to disable.
    settlement_delay : trading days between sell and cash availability (default 3, VN T+2.5)
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year_time'] = df['date'].dt.to_period(time_of_rebalance)
    rebalance_dates = sorted(df.groupby('year_time')['date'].min().unique())
     # --- VNINDEX VOLATILITY REGIME FILTER ---
    # regime_ok[date] = True  → normal market, proceed with rebalance
    # regime_ok[date] = False → high vol, skip rebalance and stay in cash
    regime_ok = {}
    if vnindex_df is not None:
        vn = vnindex_df.copy()
        vn['date'] = pd.to_datetime(vn['date'])
        vn = vn.sort_values('date').set_index('date')
        vn['daily_ret'] = vn['close'].pct_change()
        vn['vol_21d']   = vn['daily_ret'].rolling(vol_lookback).std() * (252 ** 0.5)
        vn['vol_p']     = vn['vol_21d'].rolling(vol_window).quantile(vol_percentile)
        vn['is_normal'] = vn['vol_21d'] <= vn['vol_p']
        # forward-fill so every rebalance date has a value
        vn = vn.reindex(
            pd.date_range(vn.index.min(), vn.index.max(), freq='D'),
            method='ffill'
        )
        for d in rebalance_dates:
            ts = pd.Timestamp(d)
            regime_ok[ts] = bool(vn['is_normal'].get(ts, True))
    # if vnindex_df is None, all dates are treated as normal
    # Trading calendar — used to find the exact settlement date in trading days
    trading_days = np.sort(df['date'].unique())

    def nth_trading_day_after(date, n):
        """Return the nth trading day strictly after `date`."""
        future = trading_days[trading_days > date]
        if len(future) == 0:
            return pd.Timestamp(date)
        idx = min(n - 1, len(future) - 1)
        return pd.Timestamp(future[idx])

    def settle_pending(pending_cash, as_of):
        """Release matured pending cash. Returns (freed_amount, remaining_list)."""
        freed, remaining = 0.0, []
        for entry in pending_cash:
            if pd.Timestamp(entry['available_date']) <= as_of:
                freed += entry['amount']
            else:
                remaining.append(entry)
        return freed, remaining

    cash         = initial_capital
    pending_cash = []   # [{'amount': float, 'available_date': Timestamp}]
    portfolio    = {}   # symbol -> {'shares', 'buy_price', 'highest_price'}
    history      = []

    # Scheduled buy orders waiting for their settlement date to arrive
    # [{'execute_date': Timestamp, 'targets': [sym, ...], 'prices': {sym: price}}]
    pending_buys = []

    for date in rebalance_dates:
        date = pd.Timestamp(date)
        day_df_preview = df[df['date'] == date]
        n_buy = max(1, int(len(day_df_preview) * buy_fraction))
        # --- REGIME CHECK — skip rebalance on high-vol months ---
        if not regime_ok.get(date, True):
            # Still record NAV so the equity curve has no gaps
            nav = cash
            nav += sum(e['amount'] for e in pending_cash)
            day_prices = df[df['date'] == date].set_index('Symbol')['close'].to_dict()
            for sym, pos_data in portfolio.items():
                if sym in day_prices:
                    nav += pos_data['shares'] * day_prices[sym]
            history.append({
                'date':               date,
                'total_value':        nav,
                'cash':               cash,
                'pending_cash':       sum(e['amount'] for e in pending_cash),
                'number_of_holdings': len(portfolio),
            })
            print(f"  [REGIME FILTER] {date.strftime('%Y-%m')} — high vol, skipping rebalance")
            continue   # skip everything below — no sells, no buys
        # ------------------------------------------------------------------
        # 0. Execute any deferred buys whose settlement date has arrived
        # ------------------------------------------------------------------
        still_pending_buys = []
        for order in pending_buys:
            if order['execute_date'] <= date:
                # Settle cash that matured by this buy's execution date
                freed, pending_cash = settle_pending(pending_cash, order['execute_date'])
                cash += freed

                targets = order['targets']

                # Use settlement-date prices if data is available, else fall back
                # to prices locked in on the original rebalance (sell) day
                settlement_day_df = df[df['date'] == order['execute_date']]
                settlement_prices = settlement_day_df.set_index('Symbol')['close'].to_dict()
                fallback_prices   = order['prices']

                n_total_targets = n_buy   # the full intended portfolio size
                n = len(targets)          # just the new names to actually buy
                if n > 0 and cash > 0:
                    cash_per_stock = cash / n_total_targets 

                    for sym in targets:
                        price = settlement_prices.get(sym) or fallback_prices.get(sym)
                        if price is None or price <= 0:
                            continue

                        max_shares    = cash_per_stock / (price * 1.001)
                        shares_to_buy = int(max_shares // 100) * 100

                        if shares_to_buy > 0:
                            buy_value  = shares_to_buy * price
                            fee        = buy_value * 0.001
                            total_cost = buy_value + fee

                            if sym in portfolio:
                                # Stock was held through the rebalance — average cost
                                old          = portfolio[sym]
                                total_shares = old['shares'] + shares_to_buy
                                avg_cost     = (old['shares'] * old['buy_price'] + buy_value) / total_shares
                                portfolio[sym] = {
                                    'shares':        total_shares,
                                    'buy_price':     avg_cost,
                                    'highest_price': max(old['highest_price'], price),
                                }
                            else:
                                portfolio[sym] = {
                                    'shares':        shares_to_buy,
                                    'buy_price':     price,
                                    'highest_price': price,
                                }
                            cash -= total_cost
            else:
                still_pending_buys.append(order)

        pending_buys = still_pending_buys

        # Settle any remaining cash that matured by today
        freed, pending_cash = settle_pending(pending_cash, date)
        cash += freed

        # ------------------------------------------------------------------
        # Get today's market data
        # ------------------------------------------------------------------
        day_df = df[df['date'] == date].copy()
        if day_df.empty:
            continue

        prices = day_df.set_index('Symbol')['close'].to_dict()

        # ------------------------------------------------------------------
        # Step 1: PREDICT — score and rank all stocks this morning
        # ------------------------------------------------------------------
        if model is not None:
            X_day = day_df[features]
            day_df['pred_score'] = model.predict(X_day)

        day_df = day_df.sort_values(by='pred_score', ascending=False)


        target_buy_stocks  = day_df.head(n_buy)['Symbol'].tolist()
        

        # Update peak prices for trailing stop
        for sym in portfolio:
            if sym in prices and prices[sym] > portfolio[sym]['highest_price']:
                portfolio[sym]['highest_price'] = prices[sym]

        # ------------------------------------------------------------------
        # Step 2: SELL — on rebalance morning at today's price
        # ------------------------------------------------------------------
        symbols_to_sell = []

        for sym, pos_data in list(portfolio.items()):
            if sym not in prices:
                continue

            current_price      = prices[sym]
            return_since_buy   = (current_price - pos_data['buy_price']) / pos_data['buy_price']
            drawdown_from_peak = (current_price - pos_data['highest_price']) / pos_data['highest_price']

            if sym not in target_buy_stocks:        # hard: not in buy targets → sell
                symbols_to_sell.append(sym)

        # Execute sells — proceeds enter pending_cash, available after T+settlement_delay
        settlement_date = nth_trading_day_after(date, settlement_delay)

        for sym in symbols_to_sell:
            pos           = portfolio.pop(sym)
            sell_price    = prices[sym]
            sell_value    = pos['shares'] * sell_price
            fee           = sell_value * 0.001
            tax           = sell_value * 0.001
            per_share_fee = pos['shares'] * 0.3
            net_proceeds  = sell_value - fee - tax - per_share_fee

            pending_cash.append({
                'amount':         net_proceeds,
                'available_date': settlement_date,
            })

        # ------------------------------------------------------------------
        # Step 3: SCHEDULE DEFERRED BUY on settlement_date
        #         Targets = top N stocks not already in portfolio
        # ------------------------------------------------------------------
        currently_held = set(portfolio.keys())
        new_targets    = []

        for sym in target_buy_stocks:
            if sym in currently_held:
                continue
            if trend_filter_col and trend_filter_col in day_df.columns:
                row = day_df[day_df['Symbol'] == sym]
                if not row.empty and row[trend_filter_col].values[0] > 1.0:
                    new_targets.append(sym)
            else:
                new_targets.append(sym)

        if new_targets:
            pending_buys.append({
                'execute_date': settlement_date,
                'targets':      new_targets,
                'prices':       prices,     # locked-in fallback prices from sell day
            })

        # ------------------------------------------------------------------
        # NAV: free cash + pending cash in transit + holdings mark-to-market
        # ------------------------------------------------------------------
        nav = cash
        nav += sum(e['amount'] for e in pending_cash)
        for sym, pos_data in portfolio.items():
            if sym in prices:
                nav += pos_data['shares'] * prices[sym]

        history.append({
            'date':               date,
            'total_value':        nav,
            'cash':               cash,
            'pending_cash':       sum(e['amount'] for e in pending_cash),
            'number_of_holdings': len(portfolio),
        })

    return pd.DataFrame(history)


def pretrain_and_save_artifacts(
    df: pd.DataFrame,
    selected_features: list,
    use_mega_alpha: bool = False,
    output_dir: str = "data/pretrained/",
    vnindex_df: pd.DataFrame | None = None,
) -> tuple[str, str, str]:
   
    """
    Runs the full walk-forward CV and backtest once, then saves all artifacts to disk
    so the Streamlit app can load them instantly without retraining.

    Parameters:
    - df: Fully processed DataFrame (features + targets + qid must already exist).
    - selected_features: List of feature column names to train on.
    - use_mega_alpha: Reserved for future LSTM-based mega-alpha pipeline (currently unused).
    - output_dir: Directory where artifacts are saved.
    - vnindex_df: Reserved for future market-relative metrics (currently unused).

    Returns:
    - predictions_path: Path to pretrained_predictions.parquet
    - equity_curve_path: Path to pretrained_equity_curve.parquet
    - final_model_path: Path to pretrained_model.json
    """
    print("🚀 Starting Pre-training Walk-Forward CV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run the heavy Walk-Forward CV
    honest_test_df = walk_forward_cv(
        df=df, 
        features=selected_features, 
        initial_train_months=24, 
        test_months=6, 
        gap_days=21,
        callback=lambda f, t, m: print(f"Fold {f}/{t}: {m}"), # Simple console callback
        use_mega=False,
        use_gp=False
    )
    
    # 2. Run the Backtest logic to get the Equity Curve
    print("📈 Running Backtest on OOS results...")
    result = simulate_portfolio(
        df=honest_test_df, 
        model=None, 
        features=None,
        initial_capital=100000,
        buy_fraction=0.10,

        time_of_rebalance='M', 
        trend_filter_col='dist_SMA_100'
    )
    final_model = xgb.XGBRanker(**base_model())
    x_train = df[selected_features]
    y_train=df['target_quintile']
    qids_train =df['qid']
    final_model.fit(
            x_train, y_train, qid=qids_train, 
            verbose=False
        )
    # 3. Save the critical artifacts to Parquet (much faster than CSV)
    predictions_path = os.path.join(output_dir, "pretrained_predictions.parquet")
    equity_curve_path = os.path.join(output_dir, "pretrained_equity_curve.parquet")
    final_model_path = os.path.join(output_dir,"pretrained_model.json")
    # We only need to save the columns app.py actually uses to save space!
    #cols_to_save = ['date', 'Symbol', 'next_1m_ret', 'pred_score', 'pred_quintile'] + selected_features
    # Ensure we only try to save columns that actually exist in the dataframe
    #cols_to_save = [c for c in cols_to_save if c in honest_test_df.columns]
    
    honest_test_df.to_parquet(predictions_path, index=False)
    result.to_parquet(equity_curve_path, index=False)
    final_model.save_model(final_model_path)
    print(f"✅ Success! Artifacts saved to {output_dir}")
    return predictions_path, equity_curve_path,final_model_path