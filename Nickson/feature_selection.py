import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import psutil
import os

# File paths
input_file = "../Stock_Competition/38_Training_Data_Set/training.csv"
output_dir = "output"
selected_features_file = os.path.join(output_dir, "selected_features_improved.csv")
chunksize = 15000

# Domain knowledge features
domain_features = [
    "外資券商_分點進出",
    "主力券商_分點進出",
    "個股主力買賣超統計_近1日主力買賣超",
    "個股主力買賣超統計_近5日主力買賣超",
    "技術指標_週RSI(5)",
    "技術指標_月MACD",
    "技術指標_保力加通道–頂部(20)",
    "技術指標_保力加通道–均線(20)",
    "技術指標_保力加通道–底部(20)",
    "月營收_單月合併營收年成長(%)",
    "月營收_累計合併營收成長(%)",
    "季IFRS財報_營業利益率(%)",
    "季IFRS財報_稅後純益率(%)",
    "季IFRS財報_淨值週轉率(次)",
    "日外資_外資買賣超",
    "日自營_自營商買賣超",
    "日投信_投信買賣超",
    "買超第1名分點張增減",
    "賣超第1名分點張增減",
]


# Memory usage function
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024**3:.2f} GB")


# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize feature scores
feature_scores = {"LightGBM": {}, "MutualInfo": {}, "RandomForest": {}, "Lasso": {}}
weights = {"LightGBM": 0.4, "MutualInfo": 0.3, "RandomForest": 0.2, "Lasso": 0.1}

# Process data in chunks
print("Starting feature selection...")
for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    chunk[numeric_cols] = chunk[numeric_cols].fillna(chunk[numeric_cols].median())
    X = chunk.drop(columns=["ID", "飆股"])
    y = chunk["飆股"]
    print(f"Processing chunk {chunk_idx + 1}, rows: {len(chunk)}")

    # Pre-filter low-variance features, excluding domain features
    print("Pre-filtering low-variance features...")
    available_domain_features = [f for f in domain_features if f in X.columns]
    non_domain_columns = [
        col for col in X.columns if col not in available_domain_features
    ]
    X_non_domain = X[non_domain_columns]

    selector = VarianceThreshold(threshold=0.01)
    X_non_domain_transformed = selector.fit_transform(X_non_domain)
    selected_columns = X_non_domain.columns[selector.get_support()].tolist()

    X = X[selected_columns + available_domain_features]
    print(f"Reduced to {X.shape[1]} features after variance filtering")

    # Sample-based correlation, excluding domain features
    print("Computing correlations on sample...")
    non_domain_columns = [
        col for col in X.columns if col not in available_domain_features
    ]
    X_non_domain = X[non_domain_columns]
    sample_size = min(1000, len(X_non_domain))
    X_sample = X_non_domain.sample(n=sample_size, random_state=42)
    corr_matrix = X_sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    X = X[
        [
            col
            for col in X.columns
            if col not in to_drop or col in available_domain_features
        ]
    ]
    print(f"Reduced to {X.shape[1]} features after correlation filtering")

    # Ensure domain features are retained (only those available)
    final_columns = list(
        set(X.columns).union([f for f in domain_features if f in chunk.columns])
    )
    X = X[[col for col in final_columns if col in X.columns]]
    print(f"Final feature count including domain features: {X.shape[1]}")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, n_jobs=8)
    lgb_model.fit(X, y)
    for f, s in zip(X.columns, lgb_model.feature_importances_):
        feature_scores["LightGBM"][f] = feature_scores["LightGBM"].get(f, 0) + s

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=8)
    for f, s in zip(X.columns, mi_scores):
        feature_scores["MutualInfo"][f] = feature_scores["MutualInfo"].get(f, 0) + s

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=8)
    rf_model.fit(X, y)
    for f, s in zip(X.columns, rf_model.feature_importances_):
        feature_scores["RandomForest"][f] = feature_scores["RandomForest"].get(f, 0) + s

    # Lasso
    X_scaled = StandardScaler().fit_transform(X)
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    for f, s in zip(X.columns, np.abs(lasso.coef_)):
        feature_scores["Lasso"][f] = feature_scores["Lasso"].get(f, 0) + s

    print_memory_usage()

# Combine scores with weights
combined_scores = {}
for feature in set().union(*[set(scores.keys()) for scores in feature_scores.values()]):
    score = sum(
        weights[method] * feature_scores[method].get(feature, 0)
        for method in feature_scores
    )
    combined_scores[feature] = score

# Select top 50 features, ensuring domain features
top_features = pd.Series(combined_scores).nlargest(50).index
selected_features = list(
    set(top_features).union([f for f in domain_features if f in chunk.columns])
)
print(f"Selected {len(selected_features)} features: {selected_features}")

# Save selected features
pd.DataFrame({"Feature": selected_features}).to_csv(selected_features_file, index=False)
print(f"Features saved to {selected_features_file}")
print_memory_usage()
print("Feature selection completed!")
