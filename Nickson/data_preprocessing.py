# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

def load_data_chunked(train_path, test_path, chunk_size=5000):
    scaler = StandardScaler()
    feature_names = None
    
    print("Determining feature names from training data...", flush=True)
    first_chunk = True
    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        if first_chunk:
            feature_names = chunk.drop(columns=['ID', '飆股']).columns
            first_chunk = False
        break
    
    print("Checking test data feature consistency...", flush=True)
    test_chunk = next(pd.read_csv(test_path, chunksize=chunk_size))
    test_features = test_chunk.drop(columns=['ID']).columns
    missing_in_test = set(feature_names) - set(test_features)
    extra_in_test = set(test_features) - set(feature_names)
    
    if missing_in_test:
        print(f"Warning: Features missing in test data: {missing_in_test}", flush=True)
    if extra_in_test:
        print(f"Warning: Extra features in test data: {extra_in_test}", flush=True)
    
    common_features = [f for f in feature_names if f in test_features]

    def process_chunk(chunk, is_train=True):
        X_chunk = chunk.drop(columns=['ID', '飆股' if is_train else 'ID'])
        X_chunk = X_chunk[common_features]
        X_chunk = X_chunk.fillna(0)
        X_chunk = X_chunk.replace([np.inf, -np.inf], 0) # replace NaN to 0
        X_scaled = scaler.partial_fit(X_chunk).transform(X_chunk)
        
        if is_train:
            return X_scaled, chunk['飆股'].values, chunk['ID'].values
        return X_scaled, chunk['ID'].values

    def train_data_generator():
        for chunk in tqdm(pd.read_csv(train_path, chunksize=chunk_size), desc="Loading training data", leave=True):
            X_scaled, y, ids = process_chunk(chunk, is_train=True)
            yield X_scaled, y, ids

    def test_data_generator():
        for chunk in tqdm(pd.read_csv(test_path, chunksize=chunk_size), desc="Loading testing data", leave=True):
            X_scaled, ids = process_chunk(chunk, is_train=False)
            yield X_scaled, ids

    return train_data_generator(), test_data_generator(), np.array(common_features)

def prepare_submission(predictions, test_ids, output_path):
    submission = pd.DataFrame({'ID': test_ids, '飆股': predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}", flush=True)