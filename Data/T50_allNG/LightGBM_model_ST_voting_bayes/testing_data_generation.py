import pandas as pd
import numpy as np
import psutil

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/38_Public_Test_Set_and_Submmision_Template/public_x.csv"
selected_features_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_features8.csv"
output_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST_voting/selected_public_x8.csv"
chunksize = 10000

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# 1. 讀取特徵並使用投票法篩選
print("讀取特徵選擇結果並應用投票法...")
features_df = pd.read_csv(selected_features_file)
# 統計每個特徵出現次數
feature_counts = features_df['Feature'].value_counts()
# 投票法：保留出現 4 次以上的特徵
min_votes = 4
selected_features = feature_counts[feature_counts >= min_votes].index.tolist()
print(f"投票法篩選後特徵數: {len(selected_features)}")
print(f"篩選特徵: {selected_features}")

# 加入 ID（測試集無 "飆股"）
columns_to_keep = ['ID'] + selected_features
print(f"總保留欄位數: {len(columns_to_keep)}")

# 2. 分塊讀取並處理數據
print(f"開始處理 {input_file}...")
all_data = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    # 只保留指定欄位並補 0
    chunk_filtered = chunk[columns_to_keep].fillna(0)
    all_data.append(chunk_filtered)
    print(f"已處理 {len(chunk)} 行，當前記憶體使用: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

# 合併所有分塊數據
df = pd.concat(all_data, ignore_index=True)
print(f"合併後數據形狀: {df.shape}")

# 3. 儲存結果
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"處理完成，數據已儲存至 {output_file}")
print_memory_usage()

print("程式執行完成！")