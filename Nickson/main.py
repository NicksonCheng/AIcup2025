# main.py
import argparse
import itertools
from data_preprocessing import load_data_chunked, prepare_submission
from feature_selection import lightgbm_feature_selection
from model import TransformerStockPredictor
from utils import StockDataset, batch_smote, train_model, predict, load_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 檔案路徑
TRAIN_PATH = "../Stock_competition/38_Training_Data_Set/training.csv"
TEST_PATH = "../Stock_competition/38_Public_Test_Set_and_Submmision_Template/public_x.csv"
SUBMISSION_PATH = "submission.csv"
FEATURES_PATH = "selected_features.npy"
MODEL_PATH = "trained_model.pth"

def main(args):
    # 根據模式調整參數
    CHUNK_SIZE = 20000 if args.mode == "train" else 5000
    BATCH_SIZE = 512 if args.mode == "train" else 32
    N_CHUNKS = None if args.mode == "train" else 3
    EPOCHS = 50 if args.mode == "train" else 5
    
    # 1. 分塊加載數據
    print(f"Step 1: Loading data in chunks ({'all' if N_CHUNKS is None else f'first {N_CHUNKS}'} chunks)...", flush=True)
    train_data_gen_full, test_data_gen_full, feature_names = load_data_chunked(TRAIN_PATH, TEST_PATH, chunk_size=CHUNK_SIZE)
    
    # 2. 特徵選擇或載入已有結果
    print("Step 2: Performing or loading feature selection...", flush=True)
    train_data_gen = train_data_gen_full if N_CHUNKS is None else list(itertools.islice(train_data_gen_full, N_CHUNKS))
    top_features_idx, top_features = lightgbm_feature_selection(iter(train_data_gen), feature_names, n_features=50, save_path=FEATURES_PATH)
    
    # 3. 加載數據並篩選特徵
    print("Step 3: Reloading data with selected features...", flush=True)
    train_data_gen_full, test_data_gen_full, _ = load_data_chunked(TRAIN_PATH, TEST_PATH, chunk_size=CHUNK_SIZE)
    X_train_list, y_train_list, train_ids_list = [], [], []
    X_test_list, test_ids_list = [], []
    
    for i, (X_chunk, y_chunk, ids_chunk) in enumerate(train_data_gen_full):
        if N_CHUNKS is None or i < N_CHUNKS:
            X_train_list.append(X_chunk[:, top_features_idx])
            y_train_list.append(y_chunk)
            train_ids_list.append(ids_chunk)
        else:
            break
    
    for X_chunk, ids_chunk in test_data_gen_full:
        X_test_list.append(X_chunk[:, top_features_idx])
        test_ids_list.append(ids_chunk)
    
    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    train_ids = np.hstack(train_ids_list)
    X_test = np.vstack(X_test_list)
    test_ids = np.hstack(test_ids_list)
    
    # Step 4: Predict mode
    if args.mode == "predict":
        print("Step 4: Loading model and predicting...", flush=True)
        model = load_model(TransformerStockPredictor, input_dim=50, model_path=MODEL_PATH)
        predictions = predict(model, X_test, seq_len=20, batch_size=BATCH_SIZE)
        print(f"Length of predictions: {len(predictions)}, Length of test_ids: {len(test_ids)}")
        prepare_submission(predictions, test_ids, SUBMISSION_PATH)
        return
    
    # 5. 分批SMOTE與數據準備
    print("Step 5: Balancing data with SMOTE...", flush=True)
    X_balanced_list, y_balanced_list = [], []
    for X_bal, y_bal in batch_smote(X_train, y_train, batch_size=CHUNK_SIZE):
        X_balanced_list.append(X_bal)
        y_balanced_list.append(y_bal)
    
    X_balanced = np.vstack(X_balanced_list)
    y_balanced = np.hstack(y_balanced_list)
    
    # 6. 分割訓練與驗證集
    print("Step 6: Splitting data into train and validation sets...", flush=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # 7. 準備數據集與加載器
    print("Step 7: Preparing datasets and dataloaders...", flush=True)
    train_dataset = StockDataset(X_tr, y_tr, seq_len=20)
    val_dataset = StockDataset(X_val, y_val, seq_len=20)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Step 8: Training
    print("Step 8: Training Transformer Stock Predictor model...", flush=True)
    model = TransformerStockPredictor(input_dim=50, d_model=512, n_heads=8, n_layers=6, dim_feedforward=2048)
    trained_model = train_model(model, train_loader, val_loader, epochs=EPOCHS, save_path=MODEL_PATH)
    
    # 9. 預測
    print("Step 9: Predicting on test set...", flush=True)
    predictions = predict(trained_model, X_test, seq_len=20, batch_size=BATCH_SIZE)
    print(f"Length of predictions: {len(predictions)}, Length of test_ids: {len(test_ids)}")
    
    # 10. 生成提交文件
    print("Step 10: Generating submission file...", flush=True)
    prepare_submission(predictions, test_ids, SUBMISSION_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction Script")
    parser.add_argument("--mode", choices=["test", "train", "predict"], default="test",
                        help="Mode: 'test' for quick test, 'train' for full training, 'predict' for prediction only")
    args = parser.parse_args()
    main(args)