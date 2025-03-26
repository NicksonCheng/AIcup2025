import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import psutil

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/38_Training_Data_Set/training.csv"
selected_features_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_features8.csv"
output_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_training8.csv"
chunksize = 10000

# 目標樣本數
target_positive_samples = 10000
# 移除 target_negative_samples，因為要使用全部負樣本

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
    # 只保留指定欄位並補 0
    chunk_filtered = chunk[columns_to_keep].fillna(0)
    # 分離正負樣本
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

# 3. 準備 SMOTE 數據（正樣本 + 等量負樣本）
print("準備 SMOTE 數據...")
n_smote_negative = n_positive  # 初始負樣本數與正樣本相等
df_negative_smote = df_negative.sample(n=n_smote_negative, random_state=42)
df_smote_input = pd.concat([df_positive, df_negative_smote], ignore_index=True)
X_smote_input = df_smote_input.drop(columns=['ID', '飆股'])
y_smote_input = df_smote_input['飆股']

# 4. 使用 SMOTE 增強正樣本至 10,000
print("使用 SMOTE 生成正樣本至 10,000...")
smote = SMOTE(sampling_strategy={1: target_positive_samples}, random_state=42, k_neighbors=min(5, n_positive-1))
X_smote, y_smote = smote.fit_resample(X_smote_input, y_smote_input)
print(f"SMOTE 後正樣本數: {sum(y_smote == 1)}")
print(f"SMOTE 後負樣本數: {sum(y_smote == 0)}")

# 將 SMOTE 數據轉回 DataFrame
df_smote = pd.DataFrame(X_smote, columns=unique_features)
df_smote['飆股'] = y_smote
df_smote['ID'] = range(len(df_smote))  # 重新生成 ID

# 5. 提取 SMOTE 正樣本並使用全部原始負樣本
df_smote_positive = df_smote[df_smote['飆股'] == 1]  # 10,000 正樣本
df_negative_all = df_negative  # 使用全部負樣本（199,394）
print(f"使用全部負樣本數: {len(df_negative_all)}")

# 6. 合併並打亂數據
df_balanced = pd.concat([df_smote_positive, df_negative_all], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # 打亂排序
print(f"最終數據形狀: {df_balanced.shape}")

# 7. 儲存結果
df_balanced = df_balanced[columns_to_keep]  # 確保欄位順序一致
df_balanced.to_csv(output_file, index=False, encoding='utf-8')
print(f"處理完成，數據已儲存至 {output_file}")
print_memory_usage()

print("程式執行完成！")