import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import xgboost as xgb
import numpy as np
from scipy.stats import spearmanr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_MODEL_PARAMS
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

    print(f"IC Mean : {ic_mean:.4f}   (target: > 0.05 | good: 0.05–0.08 | ref: XGBRanker NDCG baseline ~0.08, regression ~0.044 [LambdaRankIC, Lin et al. 2025])")
    print(f"IC Std  : {ic_std:.4f}   (target: < 0.12 | lower = more stable signal; typical range 0.08–0.15 for 300-stock universe)")
    print(f"IC IR   : {ic_ir:.4f}   (target: > 0.3  | good: 0.3–0.5 | strong: > 0.5 [practitioner consensus]; small universe inflates Std so > 0.3 is realistic here)")

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

def plot_feature_ir(test_df, features, target_col='next_1m_ret'):
    # Calculates IC Mean, IC Std, and IC IR for each feature.
    # IC IR = Mean(Daily IC) / Std(Daily IC)

    ic_ir_results = {}

    for feat in features:
        # 1. Calculate IC for each date (Time-series of ICs)
        daily_ic = test_df.groupby('date').apply(
            lambda x: x[feat].corr(x[target_col], method='spearman'),
            include_groups=False
        )
        
        # 2. Calculate Metrics
        ic_mean = daily_ic.mean()
        ic_std  = daily_ic.std()
        
        # Avoid division by zero if std is 0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        ic_ir_results[feat] = {
            'IC Mean': ic_mean,
            'IC Std':  ic_std,
            'IC IR':   ic_ir
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
    gains         = monthly_returns[monthly_returns > 0].sum()
    losses        = abs(monthly_returns[monthly_returns < 0].sum())
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


def allocate_equal(executable, cash):
    """
    Current behaviour: split cash equally across all executable stocks.
    Returns list of (sym, price, cash_allocated).
    """
    if not executable:
        return []
    cash_per_stock = cash / len(executable)
    return [(sym, price, cash_per_stock) for sym, price in executable]


def allocate_rank_weighted(executable, cash, scores):
    """
    Weight each stock proportionally to its pred_score rank.
    Rank 1 (highest score) gets the most cash, rank N gets the least.
    scores: dict of {sym: pred_score}
    """
    if not executable:
        return []

    syms   = [sym for sym, price in executable]
    ranked = sorted(syms, key=lambda s: scores.get(s, 0), reverse=True)
    n      = len(ranked)

    # Linear rank weights: rank 1 → weight N, rank N → weight 1
    weights = {sym: (n - i) for i, sym in enumerate(ranked)}
    total_w = sum(weights.values())

    return [
        (sym, price, cash * weights[sym] / total_w)
        for sym, price in executable
    ]

def simulate_portfolio(
    df,
    model,
    features,
    initial_capital=10_000_000,
    buy_fraction=0.05,
    time_of_rebalance='M',
    trend_filter_col='dist_SMA_100',
    settlement_delay=3,
    vnindex_df=None,
    liquidity_filter = True,
    vol_lookback=21,
    vol_percentile=0.80,
    vol_window=252,
    adtv_participation=0.10,    # your order must be <= this fraction of ADTV
    allocation='equal',
):
    """
    Hard-rebalance portfolio simulator for VN market with clean monthly timing:

        REBALANCE DAY (first trading day of month)
        ├── Step 1: Model scores & ranks all stocks
        ├── Step 2: HARD SELL — every stock NOT in top-N exits today at today's price
        │           └── proceeds → pending_cash, available on day +settlement_delay
        └── Step 3: Schedule BUY to execute exactly on settlement_date (T+3)
                    └── On that date, available cash is split equally across stocks
                        that pass BOTH the trend filter AND the ADTV liquidity filter,
                        and whose price allows at least 1 lot (100 shares).

    Key design decisions
    --------------------
    - Hard rebalance: no grace band. If a stock falls out of top-N it is sold.
    - Sells happen exactly once per month on rebalance day.
    - Buys happen exactly once per month on rebalance_day + T+settlement_delay.
    - The two events never overlap in the same loop iteration.
    - ADTV filter: position size must be <= adtv_participation * adtv_20d so we
      only buy stocks liquid enough to absorb our order.
    - Two-pass buy execution: Pass 1 finds all stocks that will actually fill
      (have a valid price and can afford >= 1 lot). Pass 2 divides cash only
      among those confirmed-executable stocks so no cash is left stranded.

    Parameters
    ----------
    df                  : DataFrame with [date, Symbol, close, volume, pred_score, ...]
    model               : trained XGBRanker; pass None to use existing 'pred_score' column
    features            : feature columns for model.predict() (ignored when model=None)
    initial_capital     : starting cash (VND)
    buy_fraction        : top X% of ranked stocks are buy targets (e.g. 0.10 = top 10%)
    time_of_rebalance   : pandas period alias ('M' = monthly, 'W' = weekly)
    trend_filter_col    : column name for trend filter; stock must have value > 1.0 to buy.
                          Pass None to disable.
    settlement_delay    : trading days between sell day and cash/buy availability (VN T+2.5 → use 3)
    vnindex_df          : optional DataFrame with [date, close] for VNINDEX vol regime filter
    vol_lookback        : rolling window (days) for realized vol on VNINDEX
    vol_percentile      : vol regime threshold percentile (0.80 = skip rebalance if top-20% vol)
    vol_window          : rolling window for vol percentile baseline
    adtv_participation  : max fraction of ADTV your order can represent (default 10%)
    allocation          : cash allocation strategy for buy orders: 'equal' (default)
                          splits cash evenly across all buy targets; 'rank_weighted' 
                          allocates proportionally to pred_score rank (rank 1 gets the most cash)
    """

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Symbol', 'date'])

    # Pre-compute ADTV (average daily traded value) per symbol per date
    # This is done once before the loop so it's O(n) not O(n * rebalances)
    if 'adtv' not in df.columns:
        raise ValueError("df must have pre-computed 'adtv' column from build_features()")

    # Build rebalance dates by stepping forward a fixed number of trading days,
    # matching the shift(-21) / shift(-5) used in build_targets so the holding
    # period the model was trained on equals the holding period in the backtest.
    # Step size is derived purely from the historical trading calendar — no future
    # dates are referenced, so there is no look-ahead.
    _step = {'M': 21, 'W': 5}.get(time_of_rebalance, 21)
    _all_trading_days = np.sort(df['date'].unique())
    rebalance_dates = []
    _idx = 0
    while _idx < len(_all_trading_days):
        rebalance_dates.append(pd.Timestamp(_all_trading_days[_idx]))
        _idx += _step

    # ------------------------------------------------------------------
    # VNINDEX volatility regime filter
    # regime_ok[date] = True  → normal market, go ahead
    # regime_ok[date] = False → high vol, skip this month entirely
    # ------------------------------------------------------------------
    regime_ok = {}
    if vnindex_df is not None:
        vn = vnindex_df.copy()
        vn['date'] = pd.to_datetime(vn['date'])
        vn = vn.sort_values('date').set_index('date')
        vn['daily_ret'] = vn['close'].pct_change()
        vn['vol_21d']   = vn['daily_ret'].rolling(vol_lookback).std() * (252 ** 0.5)
        vn['vol_p']     = vn['vol_21d'].rolling(vol_window).quantile(vol_percentile)
        vn['is_normal'] = vn['vol_21d'] <= vn['vol_p']
        vn = vn.reindex(
            pd.date_range(vn.index.min(), vn.index.max(), freq='D'),
            method='ffill'
        )
        for d in rebalance_dates:
            ts = pd.Timestamp(d)
            regime_ok[ts] = bool(vn['is_normal'].get(ts, True))

    # ------------------------------------------------------------------
    # Trading calendar helpers
    # ------------------------------------------------------------------
    trading_days = np.sort(df['date'].unique())

    def nth_trading_day_after(date, n):
        """Return the nth trading day strictly after `date`."""
        future = trading_days[trading_days > date]
        if len(future) == 0:
            return pd.Timestamp(date)
        idx = min(n - 1, len(future) - 1)
        return pd.Timestamp(future[idx])

    def settle_pending(pending_cash_list, as_of):
        """Release matured pending cash. Returns (freed_amount, remaining_list)."""
        freed, remaining = 0.0, []
        for entry in pending_cash_list:
            if pd.Timestamp(entry['available_date']) <= as_of:
                freed += entry['amount']
            else:
                remaining.append(entry)
        return freed, remaining

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    cash         = float(initial_capital)
    pending_cash = []   # [{'amount': float, 'available_date': Timestamp}]
    portfolio    = {}   # symbol -> {'shares', 'buy_price', 'highest_price'}
    history      = []

    # Each entry: {'execute_date': Timestamp, 'targets': [sym,...],
    #              'prices': {sym: price}, 'n_total_targets': int}
    pending_buys = []

    # ------------------------------------------------------------------
    # Main loop — one iteration per rebalance date
    # ------------------------------------------------------------------
    for date in rebalance_dates:
        date = pd.Timestamp(date)

        # ── REGIME CHECK ──────────────────────────────────────────────
        if not regime_ok.get(date, True):
            nav = cash + sum(e['amount'] for e in pending_cash)
            day_prices = df[df['date'] == date].set_index('Symbol')['close'].to_dict()
            for sym, pos_data in portfolio.items():
                nav += pos_data['shares'] * day_prices.get(sym, 0)
            history.append({
                'date':               date,
                'total_value':        nav,
                'cash':               cash,
                'pending_cash':       sum(e['amount'] for e in pending_cash),
                'number_of_holdings': len(portfolio),
            })
            print(f"  [REGIME FILTER] {date.strftime('%Y-%m')} — high vol, skipping rebalance")
            continue

        # ── STEP 0: EXECUTE DEFERRED BUYS whose settlement date has arrived ──
        # These are buys scheduled from a PREVIOUS month's rebalance.
        # They execute here, before this month's sell, so the sequence is:
        #   prev-month sell → T+3 buy (now) → this-month sell → T+3 buy (scheduled)
        still_pending = []
        for order in pending_buys:
            if order['execute_date'] <= date:
                # Settle cash that has matured by the buy's execution date
                freed, pending_cash = settle_pending(pending_cash, order['execute_date'])
                cash += freed

                targets           = order['targets']
                settlement_day_df = df[df['date'] == order['execute_date']]
                settlement_prices = settlement_day_df.set_index('Symbol')['close'].to_dict()
                fallback_prices   = order['prices']

                if targets and cash > 0:
                    # ── PASS 1: determine which stocks will actually execute ──
                    # Compute a conservative per-stock estimate using the full
                    # target list as denominator. Any stock that can't afford
                    # even 1 lot at that estimate is dropped before we fix the
                    # real denominator in Pass 2.
                    estimated_per_stock = cash / max(1, len(targets))
                    executable = []
                    for sym in targets:
                        price = settlement_prices.get(sym) or fallback_prices.get(sym)
                        if not price or price <= 0:
                            continue
                        # Can we afford at least 1 VN lot (100 shares)?
                        if int((estimated_per_stock / (price * 1.001)) // 100) * 100 <= 0:
                            continue
                        executable.append((sym, price))

                    # ── PASS 2: buy using the correct denominator ────────────
                    # Build score lookup for rank-weighted allocation
                    if allocation == 'rank_weighted':
                        score_lookup = settlement_day_df.set_index('Symbol')['pred_score'].to_dict() \
                                    if 'pred_score' in settlement_day_df.columns else {}
                        scores = {sym: score_lookup.get(sym, 0) for sym in order['targets']}
                        alloc = allocate_rank_weighted(executable, cash, scores)
                    else:
                        alloc = allocate_equal(executable, cash)

                    remaining_cash = cash
                    for sym, price, cash_allocated in alloc:
                        max_shares    = cash_allocated / (price * 1.001)
                        shares_to_buy = int(max_shares // 100) * 100

                        if shares_to_buy <= 0:
                            continue

                        buy_value  = shares_to_buy * price
                        fee        = buy_value * 0.001
                        total_cost = buy_value + fee

                        if total_cost > remaining_cash:
                            continue

                        if sym in portfolio:
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
                        cash           -= total_cost
                        remaining_cash -= total_cost
            else:
                still_pending.append(order)

        pending_buys = still_pending

        # Settle any remaining cash that has matured by today
        freed, pending_cash = settle_pending(pending_cash, date)
        cash += freed

        # ── GET TODAY'S MARKET DATA ───────────────────────────────────
        day_df = df[df['date'] == date].copy()
        if day_df.empty:
            continue

        prices = day_df.set_index('Symbol')['close'].to_dict()

        # ── STEP 1: PREDICT — score and rank all stocks ───────────────
        if model is not None:
            X_day = day_df[features]
            day_df['pred_score'] = model.predict(X_day)

        day_df = day_df.sort_values('pred_score', ascending=False)

        n_buy             = max(1, int(len(day_df) * buy_fraction))
        target_buy_stocks = day_df.head(n_buy)['Symbol'].tolist()

        # Update peak prices for any trailing-stop use downstream
        for sym in portfolio:
            if sym in prices and prices[sym] > portfolio[sym]['highest_price']:
                portfolio[sym]['highest_price'] = prices[sym]

        # ── STEP 2: HARD SELL ─────────────────────────────────────────
        # Any stock not in target_buy_stocks is sold — no grace band.
        settlement_date = nth_trading_day_after(date, settlement_delay)

        symbols_to_sell = [
            sym for sym in list(portfolio.keys())
            if sym in prices and sym not in target_buy_stocks
        ]

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
                'available_date': settlement_date,   # T+3
            })

        # ── STEP 3: SCHEDULE DEFERRED BUY at T+settlement_delay ──────
        # Only NEW stocks (not already held) that pass BOTH filters:
        #   (a) trend filter: dist_SMA_100 > 1.0  (stock is above its SMA)
        #   (b) ADTV filter:  our position size <= adtv_participation * adtv_20d
        #
        # Note: cash_per_stock is estimated using current cash + ALL pending cash
        # (since it will all be settled by buy day). This avoids under-buying.
        estimated_cash_at_buy = cash + sum(e['amount'] for e in pending_cash)
        cash_per_stock_est    = estimated_cash_at_buy / n_buy

        currently_held = set(portfolio.keys())
        new_targets    = []

        adtv_map = day_df.set_index('Symbol')['adtv'].to_dict() if 'adtv' in day_df.columns else {}

        for sym in target_buy_stocks:
            if sym in currently_held:
                continue

            row = day_df[day_df['Symbol'] == sym]
            if row.empty:
                continue

            # (a) Trend filter
            if trend_filter_col and trend_filter_col in day_df.columns:
                if row[trend_filter_col].values[0] <= 1.0:
                    continue
            if liquidity_filter:
                # (b) ADTV liquidity filter
                adtv_val = adtv_map.get(sym, None)
                if adtv_val and adtv_val > 0:
                    if cash_per_stock_est > adtv_participation * adtv_val:
                        continue   # our order is too large relative to this stock's liquidity

            new_targets.append(sym)

        if new_targets:
            pending_buys.append({
                'execute_date':    settlement_date,
                'targets':         new_targets,
                'prices':          prices,           # fallback prices from sell day
                'n_total_targets': n_buy,            # kept for reference / debugging
            })

        # ── NAV SNAPSHOT ──────────────────────────────────────────────
        nav = cash + sum(e['amount'] for e in pending_cash)
        for sym, pos_data in portfolio.items():
            nav += pos_data['shares'] * prices.get(sym, 0)

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
        use_gp=False
    )
    
    # 2. Run the Backtest logic to get the Equity Curve
    print("📈 Running Backtest on OOS results...")
    result = simulate_portfolio(
        df=honest_test_df, 
        model=None, 
        features=None,
        initial_capital=100000,
        buy_fraction=0.20,
        time_of_rebalance='M', 
        trend_filter_col='dist_SMA_100'
    )

    final_model = xgb.XGBRanker(**BASE_MODEL_PARAMS)
    x_train    = df[selected_features]
    y_train    = df['target_quintile']
    qids_train = df['qid']
    final_model.fit(
        x_train, y_train, qid=qids_train,
        verbose=False
    )

    # 3. Save the critical artifacts to Parquet (much faster than CSV)
    predictions_path  = os.path.join(output_dir, "pretrained_predictions.parquet")
    equity_curve_path = os.path.join(output_dir, "pretrained_equity_curve.parquet")
    final_model_path  = os.path.join(output_dir, "pretrained_model.json")
    keep_cols = ['date', 'Symbol', 'pred_score', 'pred_quintile', 'target_quintile', 'next_1m_ret']
    honest_test_df = honest_test_df[[c for c in keep_cols if c in honest_test_df.columns]]
    honest_test_df.to_parquet(predictions_path, index=False)
    result.to_parquet(equity_curve_path, index=False)
    final_model.save_model(final_model_path)
    print(f"✅ Success! Artifacts saved to {output_dir}")
    return predictions_path, equity_curve_path, final_model_path