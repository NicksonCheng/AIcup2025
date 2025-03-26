import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import psutil

# 檔案路徑
input_file = "/home/r1419r1419/Stock_Competition/38_Training_Data_Set/training.csv"
output_selected_features_file = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/selected_features7.csv"
chunksize = 15000

# 在這裡直接設定要選擇的前 X 個特徵數量
top_n = 50  # 修改此數字即可，例如 20、50、100

# 檢查記憶體使用
def print_memory_usage():
    process = psutil.Process()
    print(f"當前記憶體使用: {process.memory_info().rss / 1024**3:.2f} GB")

# 初始化特徵重要性字典
feature_scores = {
    'LightGBM': {}, 'MutualInfo': {}, 'Spearman': {}, 'RandomForest': {},
    'Pearson': {}, 'ChiSquare': {}, 'Lasso': {}
}

# 分塊讀取並計算特徵重要性
print("分塊讀取 training.csv 並進行特徵選擇...")
for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    chunk.fillna(0, inplace=True)
    X = chunk.drop(columns=['ID', '飆股'])
    y = chunk['飆股']
    print(f"處理分塊 {chunk_idx + 1}，行數: {len(chunk)}")

    # LightGBM (CPU 版本)
    lgb_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, n_jobs=24)
    lgb_model.fit(X, y)
    for f, s in zip(X.columns, lgb_model.feature_importances_):
        feature_scores['LightGBM'][f] = feature_scores['LightGBM'].get(f, 0) + s

    # 互信息 (24 核心並行)
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=24)
    for f, s in zip(X.columns, mi_scores):
        feature_scores['MutualInfo'][f] = feature_scores['MutualInfo'].get(f, 0) + s

    # Spearman
    spearman_corr = X.corrwith(y, method='spearman').abs()
    for f, s in spearman_corr.items():
        feature_scores['Spearman'][f] = feature_scores['Spearman'].get(f, 0) + s

    # 隨機森林 (CPU 版本)
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=24)
    rf_model.fit(X, y)
    for f, s in zip(X.columns, rf_model.feature_importances_):
        feature_scores['RandomForest'][f] = feature_scores['RandomForest'].get(f, 0) + s

    # 皮爾森
    pearson_corr = X.corrwith(y, method='pearson').abs()
    for f, s in pearson_corr.items():
        feature_scores['Pearson'][f] = feature_scores['Pearson'].get(f, 0) + s

    # 卡方
    X_chi = X.copy()
    X_chi[X_chi < 0] = 0
    X_chi_discretized = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(X_chi)
    chi2_scores, _ = chi2(X_chi_discretized, y)
    for f, s in zip(X.columns, chi2_scores):
        feature_scores['ChiSquare'][f] = feature_scores['ChiSquare'].get(f, 0) + s

    # Lasso
    X_scaled = StandardScaler().fit_transform(X)
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    for f, s in zip(X.columns, np.abs(lasso.coef_)):
        feature_scores['Lasso'][f] = feature_scores['Lasso'].get(f, 0) + s

    print_memory_usage()

# 選出每種方法的前 X 個特徵並生成結果
results = []
for method, scores in feature_scores.items():
    top_features = pd.Series(scores).sort_values(ascending=False).head(top_n)
    print(f"{method} 前 {top_n} 個特徵: {top_features.index.tolist()}")
    results.append(pd.DataFrame({
        'Method': method,
        'Feature': top_features.index,
        'Score': top_features.values,
        'Rank': range(1, top_n + 1)
    }))

# 儲存特徵選擇結果
final_results = pd.concat(results, ignore_index=True)
final_results.to_csv(output_selected_features_file, index=False, encoding='utf-8')
print(f"特徵選擇結果儲存至: {output_selected_features_file}")
print(f"總計選出 {len(final_results)} 條記錄（7 方法各 {top_n} 個特徵）")

# 計算聯集
all_selected_features = set(final_results['Feature'])
print(f"特徵聯集總數: {len(all_selected_features)}")
print(f"特徵聯集: {list(all_selected_features)}")
print_memory_usage()

print("程式執行完成！")