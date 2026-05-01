import numpy as np
import optuna
import pandas as pd
import os
from gplearn.genetic import SymbolicTransformer
import xgboost as xgb
from sklearn.metrics import ndcg_score
import torch
import gc
import torch.optim as optim

from src.alpha_mining import WorldQuantAlphas
from src.deep_combiner import DynamicAlphaCombiner
from src.features import build_features

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

def base_model():
    return {

        'tree_method': 'hist',
        #'device': 'cuda',
        'objective': 'rank:ndcg',

        'n_estimators': 100,

        'learning_rate': 0.1,

        'max_depth': 4,

        'colsample_bytree': 0.7,

        'subsample': 0.8,

        'random_state': 42,

        #'lambdarank_pair_method': 'topk',

        #'nthread' : 1,

        #'n_jobs' : 1

        #'lambdarank_num_pair_per_sample':10



    }
def alpha_model():
    return {
        'tree_method': 'hist', # Sử dụng GPU để huấn luyện
        'device':'cuda',
        'predictor': 'gpu_predictor',
        'objective': 'rank:ndcg', 
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'colsample_bytree': 0.5,    
        'subsample': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': 42
    }
def train_mega_combiner(train_df, alpha_cols, epochs=5):
    """
    Huấn luyện mạng LSTM-Attention để tạo ra trọng số tổ hợp Alpha động.
    """
    num_alphas = len(alpha_cols)
    model = DynamicAlphaCombiner(num_alphas=num_alphas)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Chuyển đổi DataFrame sang Tensor (Batch, 1, Features) - Đơn giản hóa cho 1 bước thời gian
    X_train = torch.tensor(train_df[alpha_cols].values, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_df['risk_adj_ret'].values, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    return model
def walk_forward_cv(df, features, model_params=None, initial_train_months=12, 
                    test_months=6, gap_days=21, callback=None, use_mega=False, use_gp=False):
    """
    Hàm Walk-forward CV hoàn chỉnh. 
    Sử dụng eval_set để tối ưu hóa quá trình học của XGBoost Ranker.
    """
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'Symbol']).copy()
    total_months = (df['date'].max().year - df['date'].min().year) * 12 + \
                   (df['date'].max().month - df['date'].min().month)
    total_folds = max(1, (total_months - initial_train_months) // test_months)
    alpha_pool = list(features)
    
    # 1. Chuẩn bị Alpha Pool nếu dùng Mega Alpha
    if use_mega:
        from src.alpha_mining import WorldQuantAlphas
        wq = WorldQuantAlphas(df)
        wq_df = wq.generate_all()
        new_cols = [c for c in wq_df.columns if c not in df.columns]
        df = pd.concat([df, wq_df[new_cols]], axis=1)
        wq_cols = [c for c in wq_df.columns if 'WQ_Alpha' in c][:10]
        alpha_pool = features + wq_cols

    current_train_end = df['date'].min() + pd.DateOffset(months=initial_train_months)
    max_date = df['date'].max()
    oos_predictions = []
    fold = 1
    
    while current_train_end < max_date:
        train_cutoff = current_train_end - pd.Timedelta(days=gap_days)
        test_start = current_train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        train_df = df.loc[df['date'] <= train_cutoff].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()
        
        if test_df.empty: break
            
        current_features = list(features)

        # --- GP MINING (Nếu bật) ---
        if use_gp:
            from gplearn.genetic import SymbolicTransformer
            gp_model = SymbolicTransformer(
                generations=20, population_size=1000, n_components=10,
                metric='pearson', n_jobs=-1, random_state=42
            )
            X_train_gp = np.nan_to_num(train_df[current_features].values)
            gp_model.fit(X_train_gp, np.nan_to_num(train_df['risk_adj_ret'].values))
            
            gp_train = gp_model.transform(X_train_gp)
            gp_test = gp_model.transform(np.nan_to_num(test_df[current_features].values))
            
            for i in range(gp_train.shape[1]):
                col_name = f'GP_Alpha_{i}'
                train_df[col_name], test_df[col_name] = gp_train[:, i], gp_test[:, i]
                current_features.append(col_name)

        # --- LSTM MEGA ALPHA (Nếu bật) ---
        if use_mega:
            from src.models import train_mega_combiner
            combiner = train_mega_combiner(train_df, alpha_pool, epochs=100)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            combiner.to(device).eval()
            
            with torch.no_grad():
                train_tensor = torch.tensor(train_df[alpha_pool].values, dtype=torch.float32).unsqueeze(1).to(device)
                test_tensor = torch.tensor(test_df[alpha_pool].values, dtype=torch.float32).unsqueeze(1).to(device)
                m_alpha_train, _ = combiner(train_tensor)
                m_alpha_test, _ = combiner(test_tensor)
                train_df['Mega_Alpha'], test_df['Mega_Alpha'] = m_alpha_train.cpu().numpy(), m_alpha_test.cpu().numpy()
            
            current_features.append('Mega_Alpha')

        # --- XGBOOST RANKER (Cập nhật quan trọng nhất) ---
        # 1. Chuẩn bị tập Train
        X_train = train_df[current_features]
        y_train = train_df['target_quintile']
        qids_train = train_df['qid']
        
        # 2. Chuẩn bị tập Validation (chính là tập Test OOS)
        X_test = test_df[current_features]
        y_test = test_df['target_quintile']
        qids_test = test_df['qid']
        
        # 3. Khởi tạo Params
        params = model_params if model_params else base_model()
        ranker = xgb.XGBRanker(**params)
        
        # 4. Huấn luyện với eval_set để mô hình hội tụ tốt nhất
        ranker.fit(
            X_train, y_train, qid=qids_train, 
            eval_set=[(X_test, y_test)], 
            eval_qid=[qids_test], 
            verbose=False
        )
        
        # 5. Dự báo điểm số
        test_df['pred_score'] = ranker.predict(X_test)
        
        # --- ĐÁNH GIÁ NDCG THỰC TẾ THEO NGÀY ---
        daily_ndcg = []
        for d, grp in test_df.groupby('date'):
            if len(grp) > 1:
                score = ndcg_score([grp['target_quintile'].values], [grp['pred_score'].values])
                daily_ndcg.append(score)
        
        current_ndcg = np.mean(daily_ndcg) if daily_ndcg else 0
        msg = f"Fold {fold}/{total_folds} ({test_start.strftime('%Y-%m')}): NDCG = {current_ndcg:.4f}"
        print(msg)

        oos_predictions.append(test_df)
        if callback:
            callback(fold, total_folds, msg)
        # Giải phóng bộ nhớ
        del train_df, X_train, y_train, X_test, y_test
        gc.collect()
        
        current_train_end = test_end
        fold += 1
        
    print("\n✅ Hoàn thành Walk-forward CV.")
    return pd.concat(oos_predictions)
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