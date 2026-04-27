
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


def run_xgboost_backtest(df, model, features, initial_capital=10000, buy_fraction=0.05, hold_fraction=0.15, trailing_stop=-0.10, take_profit=0.50, time_of_rebalance='M', trend_filter_col='dist_SMA_50'):
    """
    Simulates a portfolio with Hysteresis, a TRAILING Stop-Loss, and a Dynamic Trend Filter.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year_time'] = df['date'].dt.to_period(time_of_rebalance)
    rebalance_dates = sorted(df.groupby('year_time')['date'].min().unique())
    
    cash = initial_capital
    portfolio = {}  
    history = []    
    
    for date in rebalance_dates:
        day_df = df[df['date'] == date].copy()
        if day_df.empty:
            continue
            
        prices = day_df.set_index('Symbol')['close'].to_dict()
        
        # Predict ranking scores
        #X_day = day_df[features]
        #day_df['pred_score'] = model.predict(X_day)
        day_df = day_df.sort_values(by='pred_score', ascending=False)
        
        # Calculate thresholds
        n_buy = max(1, int(len(day_df) * buy_fraction))
        n_hold = max(1, int(len(day_df) * hold_fraction))
        
        target_buy_stocks = day_df.head(n_buy)['Symbol'].tolist()
        target_hold_stocks = day_df.head(n_hold)['Symbol'].tolist()
        
        # --- UPDATE HIGHEST PRICES (TRAILING STOP PREP) ---
        for sym in portfolio:
            if sym in prices:
                if prices[sym] > portfolio[sym]['highest_price']:
                    portfolio[sym]['highest_price'] = prices[sym]
        
        # --- PHASE 1: SELL ---
        symbols_to_sell = []
        
        for sym, pos_data in portfolio.items():
            if sym in prices:
                current_price = prices[sym]
                buy_price = pos_data['buy_price']
                highest_price = pos_data['highest_price']
                
                return_since_buy = (current_price - buy_price) / buy_price
                drawdown_from_peak = (current_price - highest_price) / highest_price
                
                if sym not in target_hold_stocks:
                    symbols_to_sell.append(sym)
                elif drawdown_from_peak <= trailing_stop:  
                    symbols_to_sell.append(sym)
                elif return_since_buy >= take_profit:
                    symbols_to_sell.append(sym)
        
        # Execute Sells
        for sym in symbols_to_sell:
            shares = portfolio[sym]['shares']
            price = prices[sym]
            
            sell_value = shares * price
            fee = sell_value * 0.001          # 0.1% transaction fee
            tax = sell_value * 0.001          # 0.1% tax
            per_share_fee = shares * 0.3      # 0.3 VND per share fee
            
            cash += (sell_value - fee - tax - per_share_fee)
            del portfolio[sym]

        # --- PHASE 2: BUY (WITH DYNAMIC TREND FILTER) ---
        new_stocks = []
        for sym in target_buy_stocks:
            if sym not in portfolio:
                # If a trend filter is provided, check it. Otherwise, approve the buy.
                if trend_filter_col is not None and trend_filter_col in day_df.columns:
                    stock_data = day_df[day_df['Symbol'] == sym]
                    if not stock_data.empty:
                        trend_val = stock_data[trend_filter_col].values[0]
                        if trend_val > 1.0:
                            new_stocks.append(sym)
                else:
                    new_stocks.append(sym) # Approve buy if no filter is active
        
        # Distribute Cash
        if new_stocks and cash > 0:
            cash_per_stock = cash / len(new_stocks)
            
            for sym in new_stocks:
                if sym in prices:
                    price = prices[sym]
                    max_shares = cash_per_stock / (price * 1.001)
                    shares_to_buy = int(max_shares // 100) * 100
                    
                    if shares_to_buy > 0:
                        buy_value = shares_to_buy * price
                        fee = buy_value * 0.001
                        total_cost = buy_value + fee
                        
                        portfolio[sym] = {
                            'shares': shares_to_buy,
                            'buy_price': price,
                            'highest_price': price 
                        }
                        cash -= total_cost
        
        # --- PHASE 3: RECORD NAV ---
        portfolio_value = cash
        for sym, pos_data in portfolio.items():
            if sym in prices:
                portfolio_value += pos_data['shares'] * prices[sym]
                
        history.append({
            'date': date,
            'total_value': portfolio_value,
            'cash': cash,
            'number_of_holdings': len(portfolio)
        })
        
    return pd.DataFrame(history)

# Run it! (This might take a few minutes depending on your computer speed)
# best_params = optimize_xgboost_ranker(df, features, n_trials=50)


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
