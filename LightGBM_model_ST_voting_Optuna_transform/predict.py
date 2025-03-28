# 檔案路徑
test_file = "/home/r1419r1419/Stock_Competition/LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_public_x8.csv"
model_dir = "/home/r1419r1419/Stock_Competition/LightGBM_model_ST_voting_Optuna_transform"
output_dir = "/home/r1419r1419/Stock_Competition/LightGBM_model_ST_voting_Optuna_transform"
scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")
model_path = os.path.join(model_dir, "lightgbm_final.pkl")

# 手動設置閾值
manual_threshold = 0.03  # 可自行調整，例如 0.05, 0.08 等
target_positive = 176    # 目標正樣本數

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

# 1. 讀取測試資料
print("讀取測試資料...")
df_test = pd.read_csv(test_file)
print(f"測試資料形狀: {df_test.shape}")
df_test.fillna(0, inplace=True)

X_test = df_test.drop(columns=['ID']).values
test_ids = df_test['ID'].values
input_dim = X_test.shape[1]
print(f"測試特徵數量: {input_dim}")

# 應用小波轉換
wavelet_price_test = wavelet_transform(X_test, 51, 20)
wavelet_volume_test = wavelet_transform(X_test, 63, 20)
X_test_transformed = np.hstack((X_test, wavelet_price_test, wavelet_volume_test))
print(f"轉換後測試特徵數量: {X_test_transformed.shape[1]}")

# 2. 載入標準化器並正規化測試資料
print("載入標準化器並正規化測試資料...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test_transformed)
print(f"正規化後測試資料形狀: {X_test_scaled.shape}")

# 3. 載入 LightGBM 模型
print("載入 LightGBM 模型...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"載入模型: {model_path}")

# 4. 進行預測並獲取概率
print("進行預測...")
test_preds_proba = model.predict(X_test_scaled)

# 5. 使用手動閾值進行預測
print(f"使用手動閾值: {manual_threshold}")
test_preds = (test_preds_proba > manual_threshold).astype(int)
positive_count = sum(test_preds)
print(f"預測正樣本數: {positive_count} (目標: {target_positive})")

# 若正樣本數不夠接近目標，提示調整建議
if positive_count < target_positive - 10 or positive_count > target_positive + 10:
    print(f"提示：正樣本數 {positive_count} 與目標 {target_positive} 差距較大，建議調整閾值。")
    if positive_count < target_positive:
        print(f"建議降低閾值至 {manual_threshold - 0.02:.2f} 或更低。")
    else:
        print(f"建議提高閾值至 {manual_threshold + 0.02:.2f} 或更高。")

# 6. 生成預測結果
print("生成預測結果...")
output_df = pd.DataFrame({
    'ID': test_ids,
    '飆股': test_preds
})

output_file = os.path.join(output_dir, "submission_manual_threshold.csv")
output_df.to_csv(output_file, index=False)
print(f"預測結果已儲存至: {output_file}")

# 7. 顯示範例
print("預測結果前五筆範例:")
print(output_df.head())

# 8. 記憶體使用
print_memory_usage()