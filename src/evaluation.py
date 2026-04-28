
import matplotlib.pyplot as plt
import pandas as pd
def feature_influence(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', title='What drives the ranking?')
    plt.show()

def evaluate_ranking_performance(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col='next_1m_ret'):
    """
    Evaluates the ranker. Hỗ trợ cả 2 chế độ:
    1. Chế độ cũ (Tĩnh): Truyền X_test và ranker để tự tính điểm.
    2. Chế độ mới (Động): Bỏ trống X_test và ranker, tự đọc cột 'pred_score' hoặc 'score' có sẵn.
    """
    eval_df = test_df.copy()
    
    # --- LOGIC ĐỘNG: Xác định nguồn lấy điểm số ---
    if ranker is not None and X_test is not None:
        # Chế độ cũ: Tự chạy dự báo
        eval_df['score'] = ranker.predict(X_test)
    elif 'pred_score' in eval_df.columns:
        # Chế độ mới: Lấy điểm OOS từ Walk-Forward
        eval_df['score'] = eval_df['pred_score']
    elif 'score' in eval_df.columns:
        pass # Nếu cột đã tên là score thì bỏ qua
    else:
        raise ValueError("Không tìm thấy điểm số! Vui lòng truyền X_test và ranker, hoặc cung cấp DataFrame có cột 'pred_score'.")
    
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

def feature_influence_ic(test_df, features, target_col='next_1m_ret'):
    """
    Đánh giá độ ảnh hưởng của Feature bằng Information Coefficient (IC).
    Tính tương quan Spearman trực tiếp từ DataFrame dự báo.
    """
    ic_dict = {}
    for feat in features:
        # Tính tương quan hạng Spearman giữa Feature và Lợi nhuận tương lai
        ic = test_df[feat].corr(test_df[target_col], method='spearman')
        ic_dict[feat] = ic
        
    # Sắp xếp và vẽ biểu đồ
    ic_series = pd.Series(ic_dict).sort_values()
    
    plt.figure(figsize=(10, 6))
    # Màu xanh cho tương quan dương, màu đỏ cho tương quan âm
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in ic_series]
    
    ic_series.plot(kind='barh', color=colors, edgecolor='black')
    plt.title('Feature Influence via Information Coefficient (OOS Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Spearman Rank Correlation (IC)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def predicted_quintile_chart(test_df, X_test=None, ranker=None):
    """
    Vẽ biểu đồ hiệu suất. Hỗ trợ cả mô hình tĩnh (có ranker) và mô hình động (đã có pred_score).
    """
    df_plot = test_df.copy()
    
    # --- LOGIC ĐỘNG: Xác định nguồn lấy điểm số ---
    if ranker is not None and X_test is not None:
        df_plot['score'] = ranker.predict(X_test)
    elif 'pred_score' in df_plot.columns:
        df_plot['score'] = df_plot['pred_score']
    elif 'score' in df_plot.columns:
        pass
    else:
        raise ValueError("Không tìm thấy điểm số! Vui lòng truyền X_test và ranker, hoặc cung cấp DataFrame có cột 'pred_score'.")
    
    # Sử dụng rank(method='first') để tránh lỗi khi có nhiều điểm số trùng nhau
    # Gán nhãn 1, 2, 3, 4, 5 cho đẹp trên trục X
    df_plot['pred_quintile'] = df_plot.groupby('date')['score'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    )

    performance = df_plot.groupby('pred_quintile')['next_1m_ret'].mean()
    
    plt.figure(figsize=(8, 5))
    performance.plot(kind='bar', title='Future Return by Predicted Quintile', color='#1f77b4', edgecolor='black')
    plt.ylabel('Average 1-Month Forward Return')
    plt.xlabel('Quintile (1 = Nhóm tệ nhất, 5 = Nhóm tốt nhất)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def capital_over_time(result):
    # 1. Ensure the 'date' column is in datetime format for proper x-axis scaling
    result['date'] = pd.to_datetime(result['date'])

    # 2. Set up the figure size (width, height in inches)
    plt.figure(figsize=(12, 6))

    # 3. Plot the data
    # marker='o' adds dots to each data point so you can see exactly where the months align
    plt.plot(result['date'], result['total_value'], marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Portfolio Total Value')

    # 4. Add titles and labels
    plt.title('Backtest Equity Curve: Total Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Value (Million VND)', fontsize=12)

    # 5. Format the chart for readability
    plt.xticks(rotation=45)                  # Angle the dates so they don't overlap
    plt.grid(True, linestyle='--', alpha=0.6) # Add a subtle background grid
    plt.legend(loc='upper left')             # Add a legend

    # 6. Adjust layout and display
    plt.tight_layout() # Ensures date labels at the bottom aren't cut off
    plt.show()


def run_xgboost_backtest(
    df,
    model,
    features,
    initial_capital=10000,
    buy_fraction=0.05,
    hold_fraction=0.15,
    trailing_stop=-0.10,
    take_profit=0.50,
    time_of_rebalance='M',
    trend_filter_col='dist_SMA_50',
    settlement_delay=3,
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
    import numpy as np

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year_time'] = df['date'].dt.to_period(time_of_rebalance)
    rebalance_dates = sorted(df.groupby('year_time')['date'].min().unique())

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

                n = len(targets)
                if n > 0 and cash > 0:
                    cash_per_stock = cash / n

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

        n_buy  = max(1, int(len(day_df) * buy_fraction))
        n_hold = max(1, int(len(day_df) * hold_fraction))

        target_buy_stocks  = day_df.head(n_buy)['Symbol'].tolist()
        target_hold_stocks = day_df.head(n_hold)['Symbol'].tolist()

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

            if sym not in target_hold_stocks:
                symbols_to_sell.append(sym)
            elif drawdown_from_peak <= trailing_stop:
                symbols_to_sell.append(sym)
            elif return_since_buy >= take_profit:
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

def plot_model_comparison(res_base, res_lstm):
    plt.figure(figsize=(12, 6))
    
    # Chuẩn hóa về tỷ lệ % lợi nhuận để dễ so sánh
    base_return = (res_base['total_value'] / res_base['total_value'].iloc[0] - 1) * 100
    lstm_return = (res_lstm['total_value'] / res_lstm['total_value'].iloc[0] - 1) * 100
    
    plt.plot(res_base['date'], base_return, label='Baseline XGBoost (Stable)', color='#1f77b4', linewidth=2)
    plt.plot(res_lstm['date'], lstm_return, label='Advanced LSTM (Unstable)', color='#d62728', linestyle='--', linewidth=2)
    
    plt.title('Equity Curve Comparison: Baseline vs. LSTM', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
