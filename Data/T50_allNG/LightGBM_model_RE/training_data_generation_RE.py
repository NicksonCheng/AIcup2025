import pandas as pd
import numpy as np
import psutil

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/38_Training_Data_Set/training.csv"
selected_features_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_features8.csv"
output_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_training8_RE.csv"
chunksize = 10000

# 複製次數
replication_factor = 2  # 複製 2 次

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# 1. 讀取聯集特徵
print("讀取特徵選擇結果...")
features_df = pd.read_csv(selected_features_file)
unique_features = features_df['Feature'].unique().tolist()
print(f"聯集特徵數: {len(unique_features)}")
print(f"聯集特徵: {unique_features}")

# 加入 ID 和 飆股
columns_to_keep = ['ID', '飆股'] + unique_features
print(f"總保留欄位數: {len(columns_to_keep)}")

# 2. 分塊讀取並處理數據
print(f"開始處理 {input_file}...")
all_positive = []
all_negative = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    chunk_filtered = chunk[columns_to_keep].fillna(0)
    positive_chunk = chunk_filtered[chunk_filtered['飆股'] == 1]
    negative_chunk = chunk_filtered[chunk_filtered['飆股'] == 0]
    all_positive.append(positive_chunk)
    all_negative.append(negative_chunk)
    print(f"已處理 {len(chunk)} 行，當前記憶體使用: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

# 合併正負樣本
df_positive = pd.concat(all_positive, ignore_index=True)
df_negative = pd.concat(all_negative, ignore_index=True)
n_positive = len(df_positive)
n_negative = len(df_negative)
print(f"原始正樣本數: {n_positive}")
print(f"原始負樣本數: {n_negative}")

# 3. 複製正樣本
print(f"複製正樣本 {replication_factor} 次...")
df_positive_replicated = pd.concat([df_positive] * replication_factor, ignore_index=True)
print(f"複製後正樣本數: {len(df_positive_replicated)}")

# 4. 使用全部負樣本並合併
df_balanced = pd.concat([df_positive_replicated, df_negative], ignore_index=True)
print(f"合併後總數: {len(df_balanced)}, 正樣本比例: {len(df_positive_replicated)/len(df_balanced):.4f}")

# 5. 打亂數據
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"最終數據形狀: {df_balanced.shape}")

# 6. 儲存結果
df_balanced = df_balanced[columns_to_keep]  # 確保欄位順序一致
df_balanced.to_csv(output_file, index=False, encoding='utf-8')
print(f"處理完成，數據已儲存至 {output_file}")
print_memory_usage()

print("程式執行完成！")