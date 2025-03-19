import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import psutil
import os
import pickle

# 檔案路徑
input_file = "Stock_Competition/38_Training_Data_Set/selected_training.csv"
output_dir = "Stock_Competition/GRU6"

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"當前記憶體使用: {mem_info.rss / 1024**3:.2f} GB")

gpus = tf.config.list_physical_devices('GPU')
print(f"檢測到的 GPU: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

try:
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    print(f"使用策略: {strategy}")
except:
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0"])
    print("雙 GPU 初始化失敗，改用單 GPU: GPU:0")

# 模型參數
MODEL_NAMES = ["gru_2.0_700", "gru_2.1_700", "gru_2.2_700", "gru_3.0_700", "gru_3.1_700", "gru_3.2_700"]
MODEL_CONFIGS = {
    "gru_2.0_700": {'hidden_sizes': [500], 'dropout_rates': [0.3, 0.0, 0.0], 'hidden_sizes_linear': [500, 300], 'dropout_rates_linear': [0.2, 0.1], 'lr': 0.0005, 'random_seed': 0},
    "gru_2.1_700": {'hidden_sizes': [500], 'dropout_rates': [0.3, 0.0, 0.0], 'hidden_sizes_linear': [500, 300], 'dropout_rates_linear': [0.2, 0.1], 'lr': 0.0005, 'random_seed': 1},
    "gru_2.2_700": {'hidden_sizes': [500], 'dropout_rates': [0.3, 0.0, 0.0], 'hidden_sizes_linear': [500, 300], 'dropout_rates_linear': [0.2, 0.1], 'lr': 0.0005, 'random_seed': 2},
    "gru_3.0_700": {'hidden_sizes': [250, 150, 150], 'dropout_rates': [0.0, 0.0, 0.0], 'hidden_sizes_linear': [], 'dropout_rates_linear': [], 'lr': 0.0005, 'random_seed': 0},
    "gru_3.1_700": {'hidden_sizes': [250, 150, 150], 'dropout_rates': [0.0, 0.0, 0.0], 'hidden_sizes_linear': [], 'dropout_rates_linear': [], 'lr': 0.0005, 'random_seed': 1},
    "gru_3.2_700": {'hidden_sizes': [250, 150, 150], 'dropout_rates': [0.0, 0.0, 0.0], 'hidden_sizes_linear': [], 'dropout_rates_linear': [], 'lr': 0.0005, 'random_seed': 2}
}
WEIGHTS = np.array([1.0] * len(MODEL_NAMES)) / len(MODEL_NAMES)
BATCH_SIZE = 64
EPOCHS = 200  # 減少 epoch 避免過擬合

def build_gru_model(config, input_dim):
    inputs = tf.keras.Input(shape=(1, input_dim))
    x = inputs
    for i, (hs, dr) in enumerate(zip(config['hidden_sizes'], config['dropout_rates'])):
        return_sequences = (i < len(config['hidden_sizes']) - 1)
        x = tf.keras.layers.GRU(hs, return_sequences=return_sequences, dropout=dr,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # 加入 L2 正則化
    for hs, dr in zip(config['hidden_sizes_linear'], config['dropout_rates_linear']):
        x = tf.keras.layers.Dense(hs, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(dr)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 1. 讀取平衡數據
print("讀取 selected_training.csv...")
df = pd.read_csv(input_file)
print(f"數據形狀: {df.shape}")
df.fillna(0, inplace=True)
print_memory_usage()

X = df.drop(columns=['ID', '飆股']).values
y = df['飆股'].values
input_dim = X.shape[1]
print(f"特徵數量: {input_dim}")

scaler = MinMaxScaler(feature_range=(0, 1))
X_2d = X.reshape(-1, input_dim)
X_scaled_2d = scaler.fit_transform(X_2d)
X = X_scaled_2d.reshape(-1, 1, input_dim)
print(f"正規化後輸入形狀: {X.shape}")

scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"標準化器已儲存至: {scaler_path}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}")

os.makedirs(output_dir, exist_ok=True)

# 早停設置
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

pipelines = {}
with strategy.scope():
    for model_name in MODEL_NAMES:
        print(f"訓練模型: {model_name}")
        tf.random.set_seed(MODEL_CONFIGS[model_name]['random_seed'])
        model = build_gru_model(MODEL_CONFIGS[model_name], input_dim)
        
        history = model.fit(X_train, y_train, 
                           batch_size=BATCH_SIZE, 
                           epochs=EPOCHS, 
                           validation_data=(X_val, y_val),
                           callbacks=[early_stopping],  # 加入早停
                           verbose=1)
        
        model_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"模型已儲存至: {model_path}")
        
        pipelines[model_name] = model

print("進行 Ensemble 預測...")
val_preds = np.zeros((X_val.shape[0], len(MODEL_NAMES)))
for i, model_name in enumerate(MODEL_NAMES):
    val_preds[:, i] = pipelines[model_name].predict(X_val, batch_size=BATCH_SIZE, verbose=0).flatten()

ensemble_pred_proba = np.average(val_preds, axis=1, weights=WEIGHTS)
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

f1 = f1_score(y_val, ensemble_pred)
print(f"驗證集 F1 分數: {f1:.4f}")

pred_df = pd.DataFrame({
    'ID': df['ID'].iloc[-X_val.shape[0]:],
    '飆股預測': ensemble_pred,
    '飆股概率': ensemble_pred_proba
})
pred_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
print(f"驗證集預測結果已儲存至: {os.path.join(output_dir, 'validation_predictions.csv')}")

print_memory_usage()
print("訓練完成！")