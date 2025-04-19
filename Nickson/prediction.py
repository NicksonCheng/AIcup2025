import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import psutil
import pywt
import os

# File paths
public_test_file = (
    "../Stock_Competition/38_Public_Test_Set_and_Submmision_Template/public_x.csv"
)
private_test_file = (
    "../Stock_Competition/38_Private_Test_Set_and_Submission_Template_V2/private_x.csv"
)
output_dir = "output"
selected_features_file = os.path.join(output_dir, "selected_features_improved.csv")
scaler_file = os.path.join(output_dir, "robust_scaler.joblib")
model_file = os.path.join(output_dir, "lightgbm_final.joblib")
target_positive_rules = {
    "public": 176,
    "private": 176,
}


# Memory usage function
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024**3:.2f} GB")


# Wavelet transform function with fixed output size
def wavelet_transform(X, start_idx, length=20, target_output_features=20):
    series = X[:, start_idx : start_idx + length]
    if series.shape[1] < length:
        padding = np.zeros((series.shape[0], length - series.shape[1]))
        series = np.hstack((series, padding))
    cA, cD = pywt.dwt(series, "db1", axis=1)
    result = np.hstack((cA, cD))
    current_features = result.shape[1]
    if current_features < target_output_features:
        padding = np.zeros((result.shape[0], target_output_features - current_features))
        result = np.hstack((result, padding))
    elif current_features > target_output_features:
        result = result[:, :target_output_features]
    return result


# Load selected features
features_df = pd.read_csv(selected_features_file)
selected_features = features_df["Feature"].tolist()
print(f"Loaded {len(selected_features)} selected features")
# Load scaler and model
scaler = joblib.load(scaler_file)
model = joblib.load(model_file)
print(f"Loaded scaler and model from {scaler_file}, {model_file}")
print(f"Scaler expects {scaler.n_features_in_} features")

# Process test sets
for test_file, name in [(public_test_file, "public"), (private_test_file, "private")]:
    print(f"Predicting on {name} test set...")
    df_test = pd.read_csv(test_file)

    # Compute median for numeric columns only
    numeric_cols = df_test.select_dtypes(include=["float64", "int64"]).columns
    medians = df_test[numeric_cols].median()

    # Fill missing values
    df_test = df_test.copy()
    for col in numeric_cols:
        df_test[col] = df_test[col].fillna(medians[col])

    # Ensure all selected features are present
    missing_cols = [col for col in selected_features if col not in df_test.columns]
    print(f"Missing features in {name} test set: {missing_cols}")
    for col in missing_cols:
        df_test[col] = 0
    X_test = df_test[selected_features].values
    print(f"Base test features: {X_test.shape[1]}")
    if X_test.shape[1] != len(selected_features):
        raise ValueError(
            f"Base feature mismatch: Expected {len(selected_features)} features, got {X_test.shape[1]}"
        )

    # Apply wavelet transform
    rsi_idx = (
        selected_features.index("技術指標_週RSI(5)")
        if "技術指標_週RSI(5)" in selected_features
        else 0
    )
    fii_idx = (
        selected_features.index("外資券商_分點進出")
        if "外資券商_分點進出" in selected_features
        else 0
    )
    print(f"Wavelet indices - RSI: {rsi_idx}, FII: {fii_idx}")

    wavelet_rsi = wavelet_transform(
        X_test, rsi_idx, length=20, target_output_features=20
    )
    wavelet_fii = wavelet_transform(
        X_test, fii_idx, length=20, target_output_features=20
    )
    print(
        f"Wavelet RSI shape: {wavelet_rsi.shape}, Wavelet FII shape: {wavelet_fii.shape}"
    )
    X_test_transformed = np.hstack((X_test, wavelet_rsi, wavelet_fii))
    print(f"Transformed test feature count: {X_test_transformed.shape[1]}")

    # Verify feature count
    expected_features = scaler.n_features_in_
    if X_test_transformed.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: X_test_transformed has {X_test_transformed.shape[1]} features, "
            f"but scaler expects {expected_features} features"
        )

    # Scale
    X_test_scaled = scaler.transform(X_test_transformed)

    # Predict
    test_preds_proba = model.predict(X_test_scaled)

    # Dynamic threshold
    thresholds = np.linspace(0.01, 0.5, 100)
    positive_counts = [sum(test_preds_proba > thresh) for thresh in thresholds]
    target_idx = np.argmin(
        [abs(count - target_positive_rules[name]) for count in positive_counts]
    )
    optimal_threshold = thresholds[target_idx]
    test_preds = (test_preds_proba > optimal_threshold).astype(int)
    print(
        f"{name} positive predictions: {sum(test_preds)}, Threshold: {optimal_threshold:.3f}"
    )

    # Save predictions
    output_df = pd.DataFrame({"ID": df_test["ID"].values, "飆股": test_preds})
    output_file = os.path.join(output_dir, f"submission_{name}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print_memory_usage()

# Combine submissions
public_df = pd.read_csv(os.path.join(output_dir, "submission_public.csv"))
private_df = pd.read_csv(os.path.join(output_dir, "submission_private.csv"))
combined_df = pd.concat([public_df, private_df])
combined_df.to_csv(
    os.path.join(output_dir, "submission_template_public_and_private.csv"), index=False
)
print(
    f"Combined submission saved to {output_dir}/submission_template_public_and_private.csv"
)
print_memory_usage()
print("Prediction completed!")
