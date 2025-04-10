import pandas as pd
import numpy as np
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
import psutil

# 檔案路徑
selected_features_file = "Stock_Competition/selected_features8.csv"
input_file = "Stock_Competition/38_Training_Data_Set_V2/training.csv"
output_file = "./selected_training7.csv"
chunksize = 10000


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

# 3. 準備 KMeansSMOTE + Tomek Links
print("準備 SMOTE + Tomek Links 數據...")
X_all = pd.concat([df_positive, df_negative], ignore_index=True).drop(columns=['ID', '飆股'])
y_all = pd.concat([df_positive, df_negative], ignore_index=True)['飆股']

# 4. 使用 Pipeline 執行 KMeansSMOTE + Tomek Links
print(f"使用 KMeansSMOTE 生成正樣本 並應用 Tomek Links...")
kmeans_smote  = KMeansSMOTE(sampling_strategy=1, random_state=42)
tomek = TomekLinks(sampling_strategy='majority')
pipeline = Pipeline([('KMeansSMOTE', kmeans_smote ), ('tomek', tomek)])
X_resampled, y_resampled = pipeline.fit_resample(X_all, y_all)
print(f"SMOTE + Tomek 後正樣本數: {sum(y_resampled == 1)}")
print(f"SMOTE + Tomek 後負樣本數: {sum(y_resampled == 0)}")

# 5. 轉回 DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=unique_features)
df_balanced['飆股'] = y_resampled
df_balanced['ID'] = range(len(df_balanced))  # 重新生成 ID

# 6. 打亂數據
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"最終數據形狀: {df_balanced.shape}")

# 7. 儲存結果
df_balanced = df_balanced[columns_to_keep]  # 確保欄位順序一致
df_balanced.to_csv(output_file, index=False, encoding='utf-8')
print(f"處理完成，數據已儲存至 {output_file}")
print_memory_usage()

print("程式執行完成！")