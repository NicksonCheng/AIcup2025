import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import optuna
import psutil
import os
import pickle
import pywt

from catboost.utils import get_gpu_device_count
print(f"可用 GPU 數量: {get_gpu_device_count()}")

# 檔案路徑 # params 視需求修改
input_file = "../CatBoost/Dataset/selected_training.csv"
output_dir = "../CatBoost"
n_jobs = 1 # 使用 GPU 須注意數量，太多 Process 同時請求使用，會 Crash

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# 小波轉換函數
def wavelet_transform(X, start_idx, length=20):
    series = X[:, start_idx:start_idx+length]
    cA, cD = pywt.dwt(series, 'db1', axis=1)
    return np.hstack((cA, cD))

# 1. 讀取數據
print("讀取訓練數據...")
df = pd.read_csv(input_file)
df.fillna(0, inplace=True)
print(f"訓練數據形狀: {df.shape}")
print_memory_usage()

X = df.drop(columns=['ID', '飆股']).values
y = df['飆股'].values
input_dim = X.shape[1]
print(f"原始特徵數量: {input_dim}")

# 應用小波轉換
wavelet_price = wavelet_transform(X, 51, 20)  # 收盤價 (索引 51-70)
wavelet_volume = wavelet_transform(X, 63, 20)  # 成交量 (索引 63-82)
X_transformed = np.hstack((X, wavelet_price, wavelet_volume))
print(f"轉換後特徵數量: {X_transformed.shape[1]}")

# 標準化
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_transformed)
print(f"正規化後輸入形狀: {X_scaled.shape}")

# 儲存標準化器
os.makedirs(output_dir, exist_ok=True)
scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"標準化器已儲存至: {scaler_path}")

# 2. 定義 Optuna 目標函數
def objective(trial):
    ## 註解是 CPU 才能找的參數
    # GPU 使用參數 task_type, devices, gpu_ram_part
    params =  {
        'loss_function': 'Logloss', 
        'eval_metric': 'F1',
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10), 
        # 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        # 'bootstrap_type': 'Bernoulli',
        'task_type': 'GPU',
        'devices' : '0',
        'gpu_ram_part': 0.7, # VRAM 使用比例，默認 0.95
        'verbose': 0
    }
    
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X_scaled):
        X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        train_pool = Pool(X_fold_train, label=y_fold_train)
        val_pool = Pool(X_fold_val, label=y_fold_val)
        
        model = CatBoostClassifier(**params).fit(
                train_pool, 
                eval_set=val_pool,
                early_stopping_rounds=50, 
                verbose=False,
            )
        
        
        val_preds_proba = model.predict_proba(val_pool)[:, 1]
        thresholds = np.linspace(0.1, 0.5, 41)
        best_f1 = 0
        for thresh in thresholds:
            val_preds = (val_preds_proba > thresh).astype(int)
            f1 = f1_score(y_fold_val, val_preds)
            if f1 > best_f1:
                best_f1 = f1
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores)

# 3. 執行 Optuna 優化
study = optuna.create_study(direction='maximize', study_name='catboost_optuna')
print("開始 Optuna 優化...")
study.optimize(objective, n_trials=100, n_jobs=n_jobs)

# 4. 提取最佳參數並儲存
best_params = study.best_params
best_params['depth'] = int(best_params['depth'])
print(f"最佳參數: {best_params}")
print(f"最佳 F1-score: {study.best_value:.4f}")

# 儲存最佳參數
params_path = os.path.join(output_dir, "best_params.pkl")
with open(params_path, 'wb') as f:
    pickle.dump(best_params, f)
print(f"最佳參數已儲存至: {params_path}")

# 5. 使用最佳參數訓練最終模型
final_params = best_params
final_params.update({
    'loss_function': 'Logloss', 
    'eval_metric': 'F1',
    'iterations': 1000,
    'learning_rate': best_params['learning_rate'],
    'depth': best_params['depth'],
    'l2_leaf_reg': best_params['l2_leaf_reg'],
    # 'subsample': best_params['subsample'],
    # 'colsample_bylevel': best_params['colsample_bylevel'],
    # 'bootstrap_type': 'Bernoulli', 
    'task_type': 'GPU',
    'devices' : '0',
    'gpu_ram_part': 0.7,
    'verbose': 0
})

print("使用最佳參數訓練最終模型...")
train_pool = Pool(X_scaled, label=y)
final_model = CatBoostClassifier(**final_params).fit(
                train_pool, 
                eval_set=train_pool,
                early_stopping_rounds=50, 
                verbose=False
            )

# 儲存最終模型
model_path = os.path.join(output_dir, "catboost_final.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"最終模型已儲存至: {model_path}")

print_memory_usage()
print("模型訓練完成！")
