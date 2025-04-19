import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import optuna
import joblib
import psutil
import pywt
import os

# File paths
input_file = "output/selected_training_improved.csv"
output_dir = "output"
scaler_file = os.path.join(output_dir, "robust_scaler.joblib")
model_file = os.path.join(output_dir, "lightgbm_final.joblib")


# Memory usage function
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024**3:.2f} GB")


# Wavelet transform function
def wavelet_transform(X, start_idx, length=20):
    series = X[:, start_idx : start_idx + length]
    cA, cD = pywt.dwt(series, "db1", axis=1)
    return np.hstack((cA, cD))


# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading training data...")
df = pd.read_csv(input_file)
df.fillna(0, inplace=True)
X = df.drop(columns=["ID", "飆股"]).values
y = df["飆股"].values
print(f"Training data shape: {df.shape}, Base features: {X.shape[1]}")
print_memory_usage()

# Apply wavelet transform
feature_names = df.drop(columns=["ID", "飆股"]).columns.tolist()
rsi_idx = (
    feature_names.index("技術指標_週RSI(5)")
    if "技術指標_週RSI(5)" in feature_names
    else 0
)
fii_idx = (
    feature_names.index("外資券商_分點進出")
    if "外資券商_分點進出" in feature_names
    else 0
)
print(f"Wavelet indices - RSI: {rsi_idx}, FII: {fii_idx}")

wavelet_rsi = wavelet_transform(X, rsi_idx, min(20, X.shape[1] - rsi_idx))
wavelet_fii = wavelet_transform(X, fii_idx, min(20, X.shape[1] - fii_idx))
print(f"Wavelet RSI shape: {wavelet_rsi.shape}, Wavelet FII shape: {wavelet_fii.shape}")

X_transformed = np.hstack((X, wavelet_rsi, wavelet_fii))
print(f"Transformed feature count: {X_transformed.shape[1]}")
# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_transformed)
joblib.dump(scaler, scaler_file)
print(f"Scaler trained on {X_scaled.shape[1]} features, saved to {scaler_file}")


# Optuna objective
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 5,
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 5),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
        "n_jobs": 8,
        "verbose": -1,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        val_preds_proba = model.predict(X_val)
        thresholds = np.linspace(0.01, 0.5, 50)
        best_f1 = 0
        for thresh in thresholds:
            val_preds = (val_preds_proba > thresh).astype(int)
            f1 = f1_score(y_val, val_preds)
            if f1 > best_f1:
                best_f1 = f1
        f1_scores.append(best_f1)
    return np.mean(f1_scores)


# Run Optuna
print("Optimizing hyperparameters...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best params: {best_params}, Best F1: {study.best_value:.4f}")

# Train final model
final_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": best_params["learning_rate"],
    "max_depth": int(best_params["max_depth"]),
    "num_leaves": int(best_params["num_leaves"]),
    "feature_fraction": best_params["feature_fraction"],
    "bagging_fraction": best_params["bagging_fraction"],
    "bagging_freq": 5,
    "scale_pos_weight": best_params["scale_pos_weight"],
    "min_child_weight": best_params["min_child_weight"],
    "min_child_samples": int(best_params["min_child_samples"]),
    "lambda_l1": best_params["lambda_l1"],
    "lambda_l2": best_params["lambda_l2"],
    "n_jobs": 8,
    "verbose": -1,
}
train_data = lgb.Dataset(X_scaled, label=y)
final_model = lgb.train(
    final_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)],
)
joblib.dump(final_model, model_file)
print(f"Model saved to {model_file}")
print_memory_usage()
print("Model training completed!")
