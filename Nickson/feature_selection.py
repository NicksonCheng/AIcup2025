# feature_selection.py
import lightgbm as lgb
import numpy as np
from tqdm import tqdm
import os

def lightgbm_feature_selection(train_data_generator, feature_names, n_features=50, chunk_size=5000, save_path="selected_features.npy"):
    # 檢查是否已有保存的特徵選擇結果
    if os.path.exists(save_path):
        print(f"Loading saved feature selection results from {save_path}...", flush=True)
        top_features_idx = np.load(save_path)
        top_features = feature_names[top_features_idx]
        return top_features_idx, top_features

    # 若無保存結果，進行特徵選擇
    params = {
        'objective': 'binary',
        'metric': 'f1',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1
    }
    importance_sum = np.zeros(len(feature_names))
    total_samples = 0

    for chunk in tqdm(train_data_generator, desc="Feature selection", leave=True):
        X_chunk, y_chunk, _ = chunk
        if np.any(np.isnan(X_chunk)) or np.any(np.isinf(X_chunk)):
            print("Warning: NaN or Inf found in chunk, replacing with 0", flush=True)
            X_chunk = np.nan_to_num(X_chunk, 0)
        
        if len(np.unique(y_chunk)) < 2:
            print("Skipping chunk with single class", flush=True)
            continue
        
        weight = sum(y_chunk == 0) / sum(y_chunk == 1) if sum(y_chunk == 1) > 0 else 1
        params['scale_pos_weight'] = weight
        train_data = lgb.Dataset(X_chunk, label=y_chunk)
        model = lgb.train(params, train_data, num_boost_round=50)
        importance_sum += model.feature_importance(importance_type='gain') * len(y_chunk)
        total_samples += len(y_chunk)

    if total_samples == 0:
        raise ValueError("No valid samples processed for feature selection")
    
    importance_avg = importance_sum / total_samples
    sorted_idx = np.argsort(importance_avg)[::-1]
    top_features_idx = sorted_idx[:n_features]
    top_features = feature_names[top_features_idx]
    
    # 保存特徵選擇結果
    np.save(save_path, top_features_idx)
    print(f"Feature selection results saved to {save_path}", flush=True)
    print("Top selected features:", top_features, flush=True)
    return top_features_idx, top_features