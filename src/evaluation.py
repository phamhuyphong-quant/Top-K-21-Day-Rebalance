
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
import optuna
import numpy as np

def feature_influence(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', title='What drives the ranking?')
    plt.show()


def evaluate_ranking_performance(test_df, X_test, ranker, top_quantile=0.2, ret_col='next_1m_ret'):
    """
    Evaluates the ranker by comparing the win rate of top-ranked picks 
    against the overall market baseline.
    """
    # 1. Generate scores and add to a copy to avoid SettingWithCopy warnings
    eval_df = test_df.copy()
    eval_df['score'] = ranker.predict(X_test)
    
    # 2. Identify top picks per date using the predicted score
    # We use transform to keep the index aligned with the original dataframe
    score_rank_pct = eval_df.groupby('date')['score'].rank(ascending=False, pct=True)
    top_picks = eval_df[score_rank_pct <= top_quantile]
    
    # 3. Calculate Win Rates
    # Win rate = percentage of picks where future returns were positive
    top_win_rate = (top_picks[ret_col] > 0).mean()
    market_win_rate = (eval_df[ret_col] > 0).mean()
    
    # 4. Calculate "Lift" (How much better are we than random?)
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

def predicted_quintile_chart(test_df, X_test, ranker):
    df_plot = test_df.copy()
    # Generate the missing column
    df_plot['score'] = ranker.predict(X_test)
    
    df_plot['pred_quintile'] = df_plot.groupby('date')['score'].transform(
        lambda x: pd.qcut(x, 5, labels=False)
    )

    performance = df_plot.groupby('pred_quintile')['next_1m_ret'].mean()
    performance.plot(kind='bar', title='Future Return by Predicted Quintile')
    plt.ylabel('Average 1-Month Forward Return')
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
def walk_forward_cv(df, features, model_params=None, initial_train_months=12, test_months=6, gap_days=21, callback=None, pretrained_model=None):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'Symbol']).copy()
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    total_folds = 0
    temp_date = min_date + pd.DateOffset(months=initial_train_months)
    while temp_date < max_date:
        total_folds += 1
        temp_date += pd.DateOffset(months=test_months)
    
    current_train_end = min_date + pd.DateOffset(months=initial_train_months)
    oos_predictions = []
    fold = 1
    
    while current_train_end < max_date:
        train_cutoff = current_train_end - pd.Timedelta(days=gap_days)
        test_start = current_train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        train_df = df[df['date'] <= train_cutoff].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()
        
        if test_df.empty:
            break
            
        fold_msg = (f"--- Fold {fold} ---\n"
                    f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)\n"
                    f"Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")
        
        print(fold_msg) 
        if callback:
            callback(fold, total_folds, fold_msg)
        
        X_test = test_df[features]
        
        # --- NEW LOGIC: Use pretrained model if provided ---
        if pretrained_model is not None:
            test_df['pred_score'] = pretrained_model.predict(X_test)
        else:
            # Otherwise, train a new model per fold
            X_train, y_train, qids_train = train_df[features], train_df['target_quintile'], train_df['qid']
            y_test, qids_test = test_df['target_quintile'], test_df['qid']
            
            if model_params is None:
                model_params = {
                    'tree_method': 'hist', 'objective': 'rank:ndcg', 
                    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4,
                    'colsample_bytree': 0.7, 'subsample': 0.8, 'random_state': 42
                }

            ranker = xgb.XGBRanker(**model_params)
            ranker.fit(X_train, y_train, qid=qids_train, eval_set=[(X_test, y_test)], eval_qid=[qids_test], verbose=False)
            test_df['pred_score'] = ranker.predict(X_test)
            
        oos_predictions.append(test_df)
        current_train_end = test_end
        fold += 1
        
    final_oos_df = pd.concat(oos_predictions)
    print("\nWalk-Forward CV Complete.")
    
    return final_oos_df

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

# Run it! (This might take a few minutes depending on your computer speed)
# best_params = optimize_xgboost_ranker(df, features, n_trials=50)