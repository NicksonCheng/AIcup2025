import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 檔案路徑
test_file = "Stock_Competition/38_Public_Test_Set_and_Submmision_Template/selected_public_x.csv"  # 請確認實際路徑
model_dir = "Stock_Competition/GRU6"
output_dir = "Stock_Competition/38_Public_Test_Set_and_Submmision_Template"
scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")

# 模型名稱與權重
MODEL_NAMES = ["gru_2.0_700", "gru_2.1_700", "gru_2.2_700", "gru_3.0_700", "gru_3.1_700", "gru_3.2_700"]
WEIGHTS = np.array([1.0] * len(MODEL_NAMES)) / len(MODEL_NAMES)
BATCH_SIZE = 64

# 1. 讀取測試資料
print("讀取測試資料...")
df_test = pd.read_csv(test_file)
print(f"測試資料形狀: {df_test.shape}")
df_test.fillna(0, inplace=True)  # 缺失值補 0

# 分離 ID 和特徵
X_test = df_test.drop(columns=['ID']).values
test_ids = df_test['ID'].values
input_dim = X_test.shape[1]
print(f"測試特徵數量: {input_dim}")

# 2. 載入標準化器並正規化
print("載入標準化器並正規化測試資料...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

X_test_2d = X_test.reshape(-1, input_dim)  # 轉為 2D
X_test_scaled_2d = scaler.transform(X_test_2d)  # 使用訓練時的 scaler
X_test = X_test_scaled_2d.reshape(-1, 1, input_dim)  # 轉回 3D
print(f"正規化後測試資料形狀: {X_test.shape}")

# 3. 載入模型並預測
print("載入模型並進行預測...")
pipelines = {}
for model_name in MODEL_NAMES:
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    pipelines[model_name] = tf.keras.models.load_model(model_path)
    print(f"載入模型: {model_path}")

# Ensemble 預測
test_preds = np.zeros((X_test.shape[0], len(MODEL_NAMES)))
for i, model_name in enumerate(MODEL_NAMES):
    test_preds[:, i] = pipelines[model_name].predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()

# 加權平均
ensemble_pred_proba = np.average(test_preds, axis=1, weights=WEIGHTS)
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

# 4. 準備輸出
print("生成預測結果...")
output_df = pd.DataFrame({
    'ID': test_ids,
    '飆股': ensemble_pred
})

# 儲存結果
output_file = os.path.join(output_dir, "submission.csv")
output_df.to_csv(output_file, index=False)
print(f"預測結果已儲存至: {output_file}")

# 顯示前五筆結果
print("預測結果前五筆範例:")
print(output_df.head())