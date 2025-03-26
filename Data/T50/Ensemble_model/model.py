import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import psutil
import os
import pickle

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/Data/T50/selected_training.csv"
output_dir = "/home/r1419r1419/Stock_Competition/Data/T50/Ensemble_model"

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# LightGBM 參數
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 5.0,
    'n_jobs': 24,
    'verbose': -1
}

# XGBoost 參數
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 7,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5.0,
    'nthread': 24,
    'verbosity': 0
}
EPOCHS = 500

# 1. 讀取數據
print("讀取 selected_training.csv...")
df = pd.read_csv(input_file)
print(f"數據形狀: {df.shape}")
df.fillna(0, inplace=True)
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

# 分割訓練與驗證集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}")

# 2. 建立數據集
lgb_train_data = lgb.Dataset(X_train, label=y_train)
lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_train_data)
xgb_train_data = xgb.DMatrix(X_train, label=y_train)
xgb_val_data = xgb.DMatrix(X_val, label=y_val)

# 3. 訓練模型
print("訓練 LightGBM 模型（CPU 版本）...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=EPOCHS,
    valid_sets=[lgb_val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)]
)

print("訓練 XGBoost 模型（CPU 版本）...")
xgb_model = xgb.train(
    xgb_params,
    xgb_train_data,
    num_boost_round=EPOCHS,
    evals=[(xgb_val_data, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=10
)

# 儲存模型
lgb_model_path = os.path.join(output_dir, "lightgbm_model.pkl")
xgb_model_path = os.path.join(output_dir, "xgboost_model.pkl")
with open(lgb_model_path, 'wb') as f:
    pickle.dump(lgb_model, f)
with open(xgb_model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"LightGBM 模型已儲存至: {lgb_model_path}")
print(f"XGBoost 模型已儲存至: {xgb_model_path}")

# 4. 預測與 Ensemble
print("進行預測...")
lgb_preds_proba = lgb_model.predict(X_val)
xgb_preds_proba = xgb_model.predict(xgb_val_data)
ensemble_preds_proba = 0.5 * lgb_preds_proba + 0.5 * xgb_preds_proba

# 動態閾值選擇（專注 F1-score）
thresholds = np.arange(0.1, 1.0, 0.05)
best_f1, best_threshold = 0, 0
for thresh in thresholds:
    ensemble_preds = (ensemble_preds_proba > thresh).astype(int)
    f1 = f1_score(y_val, ensemble_preds)
    n_positive = np.sum(ensemble_preds)
    print(f"閾值 {thresh:.2f}, F1: {f1:.4f}, 正樣本數: {n_positive}")
    if f1 > best_f1:
        best_f1, best_threshold = f1, thresh
print(f"最佳閾值: {best_threshold:.2f}, 最佳 F1: {best_f1:.4f}")

# 儲存最佳閾值
threshold_path = os.path.join(output_dir, "best_threshold.txt")
with open(threshold_path, 'w') as f:
    f.write(str(best_threshold))
print(f"最佳閾值已儲存至: {threshold_path}")

# 使用最佳閾值生成最終預測
ensemble_preds = (ensemble_preds_proba > best_threshold).astype(int)

# 儲存驗證集預測結果
pred_df = pd.DataFrame({
    'ID': df['ID'].iloc[-X_val.shape[0]:],
    '飆股預測': ensemble_preds,
    '飆股概率': ensemble_preds_proba
})
pred_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
print(f"驗證集預測結果已儲存至: {os.path.join(output_dir, 'validation_predictions.csv')}")

print_memory_usage()
print("訓練完成！")