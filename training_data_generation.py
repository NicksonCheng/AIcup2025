import pandas as pd
import numpy as np
import psutil

# 檔案路徑
selected_features_file = "Stock_Competition/selected_features7.csv"
input_file = "Stock_Competition/38_Training_Data_Set/training.csv"
output_file = "Stock_Competition/38_Training_Data_Set/selected_training.csv"

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

# 2. 分塊讀取並收集所有數據
print("開始處理 training.csv...")
chunksize = 10000
all_data = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    # 只保留指定欄位
    chunk_filtered = chunk[columns_to_keep]
    all_data.append(chunk_filtered)
    print(f"已處理 {chunk.index[-1] + 1} 行，當前記憶體使用: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

# 合併所有分塊數據
df = pd.concat(all_data, ignore_index=True)
print(f"合併後數據形狀: {df.shape}")

# 3. 分離正負樣本
df_positive = df[df['飆股'] == 1]  # 所有飆股=1 的樣本
df_negative = df[df['飆股'] == 0]  # 所有飆股=0 的樣本
n_positive = len(df_positive)
print(f"飆股=1 的樣本數: {n_positive}")
print(f"飆股=0 的樣本數: {len(df_negative)}")

# 4. 從負樣本中隨機抽樣
df_negative_sampled = df_negative.sample(n=n_positive, random_state=42)  # 隨機抽樣與正樣本數量相等
print(f"抽樣後飆股=0 的樣本數: {len(df_negative_sampled)}")

# 5. 合併正負樣本
df_balanced = pd.concat([df_positive, df_negative_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # 打亂順序
print(f"平衡後數據形狀: {df_balanced.shape}")

# 6. 儲存結果
df_balanced.to_csv(output_file, index=False)
print(f"處理完成，平衡數據已儲存至 {output_file}")
print_memory_usage()

print("程式執行完成！")