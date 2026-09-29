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

    print(f"IC Mean : {ic_mean:.4f}   (target: > 0.05 | good: 0.05–0.08)")
    print(f"IC Std  : {ic_std:.4f}   (target: < 0.12)")
    print(f"IC IR   : {ic_ir:.4f}   (target: > 0.3)")
    return {'ic_mean': ic_mean, 'ic_std': ic_std, 'ic_ir': ic_ir, 'daily_ic': daily_ic}

def compute_top_quantile_win_rate(test_df, X_test=None, ranker=None, top_quantile=0.2, ret_col='next_1m_ret'):
    eval_df = test_df.copy()
    if ranker is not None and X_test is not None:
        eval_df['score'] = ranker.predict(X_test)
    elif 'pred_score' in eval_df.columns:
        eval_df['score'] = eval_df['pred_score']
    elif 'score' in eval_df.columns:
        pass
    else:
        raise ValueError("No score column found.")
    
    score_rank_pct = eval_df.groupby('date')['score'].rank(ascending=False, pct=True)
    top_picks = eval_df[score_rank_pct <= top_quantile]
    top_win_rate = (top_picks[ret_col] > 0).mean()
    market_win_rate = (eval_df[ret_col] > 0).mean()
    lift = top_win_rate - market_win_rate
    
    print(f"--- Backtest Results (Top {top_quantile*100:.0f}% Picks) ---")
    print(f"Top Picks Win Rate:   {top_win_rate:.2%}")
    print(f"Market Baseline:      {market_win_rate:.2%}")
    print(f"Excess Win Rate:      {lift:+.2%}")
    return {'top_win_rate': top_win_rate, 'market_win_rate': market_win_rate, 'lift': lift, 'top_picks_df': top_picks}

def plot_feature_ic(test_df, features, target_col='next_1m_ret'):
    ic_dict = {feat: test_df[feat].corr(test_df[target_col], method='spearman') for feat in features}
    ic_series = pd.Series(ic_dict).sort_values()
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in ic_series]
    ic_series.plot(kind='barh', color=colors, edgecolor='black')
    plt.title('Feature Influence via Information Coefficient (OOS Data)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_feature_ir(test_df, features, target_col='next_1m_ret'):
    ic_ir_results = {}
    for feat in features:
        daily_ic = test_df.groupby('date').apply(lambda x: x[feat].corr(x[target_col], method='spearman'), include_groups=False)
        ic_mean, ic_std = daily_ic.mean(), daily_ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_ir_results[feat] = {'IC Mean': ic_mean, 'IC Std': ic_std, 'IC IR': ic_ir}
    ir_df = pd.DataFrame(ic_ir_results).T.sort_values(by='IC IR', ascending=True)
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in ir_df['IC IR']]
    ir_df['IC IR'].plot(kind='barh', color=colors, edgecolor='black')
    plt.title('Feature Consistency (IC IR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return ir_df

def plot_feature_rolling_ir(test_df, feature, target_col='next_1m_ret', window=6):
    monthly_ic = test_df.groupby(test_df['date'].dt.to_period('M')).apply(lambda x: x[feature].corr(x[target_col], method='spearman'), include_groups=False)
    rolling_ir = monthly_ic.rolling(window=window).mean() / monthly_ic.rolling(window=window).std()
    rolling_ir.plot(title=f'Rolling {window}-Month IC IR for {feature}')
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

def plot_return_by_predicted_quintile(test_df, X_test=None, ranker=None):
    df_plot = test_df.copy()
    if ranker is not None and X_test is not None:
        df_plot['score'] = ranker.predict(X_test)
    elif 'pred_score' in df_plot.columns:
        df_plot['score'] = df_plot['pred_score']
    else:
        raise ValueError("No score column found.")
    df_plot['pred_quintile'] = df_plot.groupby('date')['score'].transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=[1, 2, 3, 4, 5]))
    performance = df_plot.groupby('pred_quintile')['next_1m_ret'].mean()
    plt.figure(figsize=(8, 5))
    performance.plot(kind='bar', title='Future Return by Predicted Quintile', color='#1f77b4', edgecolor='black')
    plt.show()

def plot_equity_curves(*results, labels=None, normalize=False, regime_colors=None, show_regime = False):
    plt.figure(figsize=(14, 7))
    
    # 1. Bảng màu mặc định cho 4 Regimes (bạn có thể tùy chỉnh)
    if regime_colors is None:
        regime_colors = {
            'Q1': '#2ca02c',  # Xanh lá (vd: Tăng trưởng tốt)
            'Q2': '#1f77b4',  # Xanh dương (vd: Ổn định)
            'Q3': '#ff7f0e',  # Cam (vd: Cảnh báo / Biến động)
            'Q4': '#d62728'   # Đỏ (vd: Giảm mạnh / Suy thoái)
        }

    # 2. Xử lý tô màu nền Regime (Lấy theo DataFrame đầu tiên)
    if show_regime and len(results) > 0 and 'regime_bucket_monthly' in results[0].columns:
        df_regime = results[0].copy()
        df_regime['date'] = pd.to_datetime(df_regime['date'])
        df_regime = df_regime.sort_values('date').reset_index(drop=True)

        # Tìm các khoảng thời gian liên tục của từng regime
        # (Xác định điểm bắt đầu/thay đổi trạng thái)
        df_regime['regime_change'] = df_regime['regime_bucket_monthly'] != df_regime['regime_bucket_monthly'].shift(1)
        df_regime['group_id'] = df_regime['regime_change'].cumsum()

        added_regime_labels = set()

        for _, group in df_regime.groupby('group_id'):
            regime = group['regime_bucket_monthly'].iloc[0]
            if regime == 'none':
                continue

            start_date = group['date'].iloc[0]
            end_date = group['date'].iloc[-1]
            
            color = regime_colors.get(regime, '#cccccc') # Mặc định xám nếu không có trong dict
            
            # Chỉ thêm label vào legend 1 lần cho mỗi loại Regime
            reg_label = f'Regime {regime}' if regime not in added_regime_labels else None
            if reg_label:
                added_regime_labels.add(regime)

            # Tô màu nền cho khoảng thời gian này
            plt.axvspan(start_date, end_date, color=color, alpha=0.18, label=reg_label)

    # 3. Vẽ đường Equity Curve (Giữ nguyên logic của bạn)
    for i, result in enumerate(results):
        result = result.copy()
        result['date'] = pd.to_datetime(result['date'])
        label = labels[i] if labels and i < len(labels) else f'Strategy {i+1}'
        y = result['total_value']
        if normalize: 
            y = (y / y.iloc[0] - 1) * 100
            
        plt.plot(result['date'], y, linewidth=2, label=label, color='black' if len(results)==1 else None)

    # 4. Trang trí biểu đồ
    plt.title('Equity Curve with Regime Buckets', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Return (%)' if normalize else 'Total Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()



def print_performance_report(result, initial_capital=None, rf_annual=0.045, trading_days_per_year=252, verbose=True):
    nav = result['total_value'].copy()
    if initial_capital is None: 
        initial_capital = nav.iloc[0]    
    
    # 1. Renamed to daily_returns to reflect the daily precision from your OOP engine
    daily_returns = nav.pct_change().dropna()
    
    nav_df = result.sort_values('date')
    days_elapsed = (nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days
    n_years = max(days_elapsed / 365.25, 1/12)
    
    total_return = (nav.iloc[-1] / initial_capital) - 1
    cagr = (nav.iloc[-1] / initial_capital) ** (1 / n_years) - 1
    
    # 2. Scale the annual risk-free hurdle rate down to a daily rate
    daily_rf = rf_annual / trading_days_per_year
    excess_ret = daily_returns - daily_rf
    
    # 3. Annualize the Sharpe Ratio using the square root of daily trading days
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(trading_days_per_year) if excess_ret.std() > 0 else 0
    
    # 4. Calculate Sortino using daily downside risk and scale by sqrt of daily trading days
    downside_ret = excess_ret.clip(upper=0)
    downside_std = downside_ret.std()
    sortino = (excess_ret.mean() / downside_std) * np.sqrt(trading_days_per_year) if downside_std > 0 else 0
    
    max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    if verbose:
        print("=" * 40)
        print(f"  Total Return    : {total_return:>+.2%}")
        print(f"  CAGR            : {cagr:>+.2%}")
        print(f"  Sharpe Ratio    : {sharpe:>6.2f}")
        print(f"  Sortino Ratio   : {sortino:>6.2f}")
        print(f"  Max Drawdown    : {max_dd:>+.2%}")
        print(f"  Calmar Ratio    : {calmar:>6.2f}")
        print("=" * 40)
    
    return {
        'total_return': total_return, 
        'cagr': cagr, 
        'sharpe': sharpe, 
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar
    }
