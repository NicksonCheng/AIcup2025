import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import psutil
import os
import pickle

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST_voting/selected_training8_ST.csv"
output_dir = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST_voting"

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# 1. 讀取數據
print("讀取訓練數據...")
df = pd.read_csv(input_file)
df.fillna(0, inplace=True)
print(f"訓練數據形狀: {df.shape}")
print_memory_usage()

X = df.drop(columns=['ID', '飆股']).values
y = df['飆股'].values
input_dim = X.shape[1]
print(f"特徵數量: {input_dim}")

# 標準化
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
print(f"正規化後輸入形狀: {X_scaled.shape}")

# 儲存標準化器
os.makedirs(output_dir, exist_ok=True)
scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"標準化器已儲存至: {scaler_path}")

# 2. 定義貝葉斯優化目標函數（5 折 CV 的 F1-score）
def lgb_cv(learning_rate, max_depth, num_leaves, feature_fraction, bagging_fraction, scale_pos_weight, min_child_weight):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_child_weight': min_child_weight,
        'n_jobs': 24,
        'verbose': -1
    }
    
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X_scaled):
        X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds_proba = model.predict(X_fold_val)
        val_preds = (val_preds_proba > 0.2).astype(int)  # 固定閾值 0.2
        f1 = f1_score(y_fold_val, val_preds)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

# 3. 定義參數範圍
pbounds = {
    'learning_rate': (0.005, 0.1),
    'max_depth': (3, 15),
    'num_leaves': (15, 63),
    'feature_fraction': (0.6, 1.0),
    'bagging_fraction': (0.6, 1.0),
    'scale_pos_weight': (5, 50),
    'min_child_weight': (0.001, 10)
}

# 4. 執行貝葉斯優化
optimizer = BayesianOptimization(
    f=lgb_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("開始貝葉斯優化...")
optimizer.maximize(
    init_points=5,  # 隨機探索次數
    n_iter=25       # 優化迭代次數
)

# 5. 提取最佳參數並儲存
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_leaves'] = int(best_params['num_leaves'])
print(f"最佳參數: {best_params}")
print(f"最佳 F1-score: {optimizer.max['target']:.4f}")

# 儲存最佳參數
params_path = os.path.join(output_dir, "best_params.pkl")
with open(params_path, 'wb') as f:
    pickle.dump(best_params, f)
print(f"最佳參數已儲存至: {params_path}")

# 6. 使用最佳參數訓練最終模型
final_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': best_params['learning_rate'],
    'max_depth': best_params['max_depth'],
    'num_leaves': best_params['num_leaves'],
    'feature_fraction': best_params['feature_fraction'],
    'bagging_fraction': best_params['bagging_fraction'],
    'bagging_freq': 5,
    'scale_pos_weight': best_params['scale_pos_weight'],
    'min_child_weight': best_params['min_child_weight'],
    'n_jobs': 24,
    'verbose': -1
}

print("使用最佳參數訓練最終模型...")
train_data = lgb.Dataset(X_scaled, label=y)
final_model = lgb.train(
    final_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# 儲存最終模型
model_path = os.path.join(output_dir, "lightgbm_final.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"最終模型已儲存至: {model_path}")

print_memory_usage()
print("模型訓練完成！")