import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import psutil

# 檔案路徑
input_file = "Stock_Competition/38_Training_Data_Set/training.csv"  # 替換為實際路徑
output_sample_file = "Stock_Competition/sampled_data7.csv"
output_selected_features_file = "Stock_Competition/selected_features7.csv"

# 目標總樣本數（正負各半）
target_sample_size_per_class = 16739  # 總計 33,478，若正樣本不足可調整

# 檢查記憶體使用函數
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

# 1. 統計正負樣本數量
print("統計正負樣本數量...")
chunksize = 10000
positive_indices = []
negative_indices = []
current_idx = 0

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    positive_mask = chunk['飆股'] == 1
    positive_local_indices = np.where(positive_mask)[0]
    negative_local_indices = np.where(~positive_mask)[0]
    positive_global_indices = [current_idx + i for i in positive_local_indices]
    negative_global_indices = [current_idx + i for i in negative_local_indices]
    positive_indices.extend(positive_global_indices)
    negative_indices.extend(negative_global_indices)
    current_idx += len(chunk)
    print(f"已處理 {current_idx} 行，當前記憶體使用: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

total_positive = len(positive_indices)
total_negative = len(negative_indices)
total_rows = total_positive + total_negative
print(f"總行數: {total_rows}")
print(f"正樣本數: {total_positive}，負樣本數: {total_negative}")

# 2. 確定實際抽樣數量
if total_positive < target_sample_size_per_class:
    print(f"警告: 正樣本數 ({total_positive}) 小於目標 ({target_sample_size_per_class})，將使用所有正樣本")
    sample_size_per_class = total_positive
else:
    sample_size_per_class = target_sample_size_per_class
print(f"每類抽樣數量: {sample_size_per_class}")

# 3. 分層隨機抽樣
np.random.seed(42)
positive_sample_indices = np.random.choice(positive_indices, sample_size_per_class, replace=False)
negative_sample_indices = np.random.choice(negative_indices, sample_size_per_class, replace=False)
sample_indices = np.concatenate([positive_sample_indices, negative_sample_indices])
sample_indices.sort()

# 4. 分塊抽樣並寫入檔案
print("開始抽樣...")
current_idx = 0
target_idx_pos = 0
header_written = False

with open(output_sample_file, 'w') as f:
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_indices = range(current_idx, current_idx + len(chunk))
        sampled_chunk = []
        while (target_idx_pos < len(sample_indices) and 
               sample_indices[target_idx_pos] < current_idx + len(chunk)):
            rel_idx = sample_indices[target_idx_pos] - current_idx
            sampled_chunk.append(chunk.iloc[rel_idx])
            target_idx_pos += 1
        
        if sampled_chunk:
            chunk_df = pd.concat(sampled_chunk, axis=1).T.reset_index(drop=True)
            if not header_written:
                chunk_df.to_csv(f, index=False)
                header_written = True
            else:
                chunk_df.to_csv(f, mode='a', header=False, index=False)
        
        current_idx += len(chunk)
        if target_idx_pos >= len(sample_indices):
            break

print(f"抽樣完成，樣本已儲存至 {output_sample_file}")
print_memory_usage()

# 5. 讀取抽樣數據並處理 NaN
print("讀取抽樣數據並處理 NaN...")
sampled_df = pd.read_csv(output_sample_file)
sampled_df.fillna(0, inplace=True)
sampled_df.to_csv(output_sample_file, index=False)
print(f"NaN 處理完成，數據更新至 {output_sample_file}")

# 6. 特徵選擇
print("讀取數據進行特徵選擇...")
sampled_df = pd.read_csv(output_sample_file)

# 移除非數值欄位
numeric_cols = sampled_df.select_dtypes(include=['int', 'float', 'bool']).columns
non_numeric_cols = set(sampled_df.columns) - set(numeric_cols)
if non_numeric_cols:
    print(f"發現非數值欄位: {non_numeric_cols}，將移除...")
    sampled_df = sampled_df.drop(columns=list(non_numeric_cols - {'飆股'}))

X = sampled_df.drop(columns=['飆股'])
y = sampled_df['飆股']

# 多種特徵選擇方法
results = []

# 方法 1: LightGBM 特徵重要性
print("使用 LightGBM 選擇特徵...")
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=8)
lgb_model.fit(X, y)
lgb_importances = pd.Series(lgb_model.feature_importances_, index=X.columns)
lgb_top = lgb_importances.sort_values(ascending=False).head(20)
print("LightGBM 前 20 個特徵:")
print(lgb_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'LightGBM',
    'Feature': lgb_top.index,
    'Score': lgb_top.values,
    'Rank': range(1, 21)
}))

# 方法 2: 互信息
print("使用互信息選擇特徵...")
mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=8)
mi_importances = pd.Series(mi_scores, index=X.columns)
mi_top = mi_importances.sort_values(ascending=False).head(20)
print("互信息 前 20 個特徵:")
print(mi_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'MutualInfo',
    'Feature': mi_top.index,
    'Score': mi_top.values,
    'Rank': range(1, 21)
}))

# 方法 3: Spearman 相關性
print("使用 Spearman 相關性選擇特徵...")
spearman_corr = X.corrwith(y, method='spearman').abs()
spearman_top = spearman_corr.sort_values(ascending=False).head(20)
print("Spearman 前 20 個特徵:")
print(spearman_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'Spearman',
    'Feature': spearman_top.index,
    'Score': spearman_top.values,
    'Rank': range(1, 21)
}))

# 方法 4: 隨機森林特徵重要性
print("使用隨機森林選擇特徵...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_top = rf_importances.sort_values(ascending=False).head(20)
print("隨機森林 前 20 個特徵:")
print(rf_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'RandomForest',
    'Feature': rf_top.index,
    'Score': rf_top.values,
    'Rank': range(1, 21)
}))

# 方法 5: 皮爾森相關係數
print("使用皮爾森相關係數選擇特徵...")
pearson_corr = X.corrwith(y, method='pearson').abs()
pearson_top = pearson_corr.sort_values(ascending=False).head(20)
print("皮爾森 前 20 個特徵:")
print(pearson_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'Pearson',
    'Feature': pearson_top.index,
    'Score': pearson_top.values,
    'Rank': range(1, 21)
}))

# 方法 6: 卡方檢定
print("使用卡方檢定選擇特徵...")
X_chi = X.copy()
X_chi[X_chi < 0] = 0  # 轉為非負
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_chi_discretized = discretizer.fit_transform(X_chi)
chi2_scores, _ = chi2(X_chi_discretized, y)
chi2_importances = pd.Series(chi2_scores, index=X.columns)
chi2_top = chi2_importances.sort_values(ascending=False).head(20)
print("卡方 前 20 個特徵:")
print(chi2_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'ChiSquare',
    'Feature': chi2_top.index,
    'Score': chi2_top.values,
    'Rank': range(1, 21)
}))

# 方法 7: Lasso 回歸
print("使用 Lasso 回歸選擇特徵...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y)
lasso_coef = pd.Series(np.abs(lasso.coef_), index=X.columns)
lasso_top = lasso_coef.sort_values(ascending=False).head(20)
print("Lasso 前 20 個特徵:")
print(lasso_top.index.tolist())
results.append(pd.DataFrame({
    'Method': 'Lasso',
    'Feature': lasso_top.index,
    'Score': lasso_top.values,
    'Rank': range(1, 21)
}))

# 合併並儲存結果
final_results = pd.concat(results, ignore_index=True)
final_results.to_csv(output_selected_features_file, index=False)
print(f"所有特徵選擇結果已儲存至 {output_selected_features_file}")
print(f"總計選出 {len(final_results)} 條記錄（7 方法各 20 個特徵）")

# 清理記憶體
del sampled_df, X, y, lgb_model, rf_model, lasso
print_memory_usage()

print("程式執行完成！")