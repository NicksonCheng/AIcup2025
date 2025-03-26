import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 檔案路徑
test_file = "/home/r1419r1419/Stock_Competition/Data/T50/selected_public_x.csv"
model_dir = "/home/r1419r1419/Stock_Competition/Data/T50/CatBoost_model"
output_dir = "/home/r1419r1419/Stock_Competition/Data/T50/CatBoost_model"
scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")
model_path = os.path.join(model_dir, "catboost_model.pkl")
threshold_path = os.path.join(model_dir, "best_threshold.txt")

# 1. 讀取測試資料
print("讀取測試資料...")
df_test = pd.read_csv(test_file)
print(f"測試資料形狀: {df_test.shape}")
df_test.fillna(0, inplace=True)

X_test = df_test.drop(columns=['ID']).values
test_ids = df_test['ID'].values
input_dim = X_test.shape[1]
print(f"測試特徵數量: {input_dim}")

# 2. 載入標準化器並正規化測試資料
print("載入標準化器並正規化測試資料...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)
print(f"正規化後測試資料形狀: {X_test_scaled.shape}")

# 3. 載入 CatBoost 模型
print("載入 CatBoost 模型...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"載入模型: {model_path}")

# 4. 載入最佳閾值
print("載入最佳閾值...")
with open(threshold_path, 'r') as f:
    THRESHOLD = float(f.read())
print(f"使用閾值: {THRESHOLD}")

# 5. 進行預測
print("進行預測...")
test_preds_proba = model.predict_proba(X_test_scaled)[:, 1]  # 取正類概率
test_preds = (test_preds_proba > THRESHOLD).astype(int)

# 6. 生成預測結果
print("生成預測結果...")
output_df = pd.DataFrame({
    'ID': test_ids,
    '飆股': test_preds
})

output_file = os.path.join(output_dir, "submission.csv")
output_df.to_csv(output_file, index=False)
print(f"預測結果已儲存至: {output_file}")

# 7. 顯示範例
print("預測結果前五筆範例:")
print(output_df.head())