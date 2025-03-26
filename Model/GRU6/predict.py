import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 檔案路徑
test_file = "/home/r1419r1419/Stock_Competition/38_Public_Test_Set_and_Submmision_Template/selected_public_x.csv"  # 請確認實際路徑
model_dir = "/home/r1419r1419/Stock_Competition/GRU6"
output_dir = "/home/r1419r1419/Stock_Competition/38_Public_Test_Set_and_Submmision_Template"
scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")

MODEL_NAMES = ["gru_2.0_700", "gru_2.1_700", "gru_2.2_700", "gru_3.0_700", "gru_3.1_700", "gru_3.2_700"]
WEIGHTS = np.array([1.0] * len(MODEL_NAMES)) / len(MODEL_NAMES)
BATCH_SIZE = 64
THRESHOLD = 0.9

print("讀取測試資料...")
df_test = pd.read_csv(test_file)
print(f"測試資料形狀: {df_test.shape}")
df_test.fillna(0, inplace=True)

X_test = df_test.drop(columns=['ID']).values
test_ids = df_test['ID'].values
input_dim = X_test.shape[1]
print(f"測試特徵數量: {input_dim}")

print("載入標準化器並正規化測試資料...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

X_test_2d = X_test.reshape(-1, input_dim)
X_test_scaled_2d = scaler.transform(X_test_2d)
X_test = X_test_scaled_2d.reshape(-1, 1, input_dim)
print(f"正規化後測試資料形狀: {X_test.shape}")

print("載入模型並進行預測...")
pipelines = {}
for model_name in MODEL_NAMES:
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    # 使用 custom_objects 忽略不支援的參數
    pipelines[model_name] = tf.keras.models.load_model(
        model_path,
        custom_objects={"GRU": tf.keras.layers.GRU}  # 明確指定 GRU 層
    )
    print(f"載入模型: {model_path}")

test_preds = np.zeros((X_test.shape[0], len(MODEL_NAMES)))
for i, model_name in enumerate(MODEL_NAMES):
    test_preds[:, i] = pipelines[model_name].predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()

ensemble_pred_proba = np.average(test_preds, axis=1, weights=WEIGHTS)
ensemble_pred = (ensemble_pred_proba > THRESHOLD).astype(int)

print("生成預測結果...")
output_df = pd.DataFrame({
    'ID': test_ids,
    '飆股': ensemble_pred
})

output_file = os.path.join(output_dir, "submission.csv")
output_df.to_csv(output_file, index=False)
print(f"預測結果已儲存至: {output_file}")

print("預測結果前五筆範例:")
print(output_df.head())