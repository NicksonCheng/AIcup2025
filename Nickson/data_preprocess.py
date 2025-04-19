import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
import psutil
import os

# File paths
input_file = "../Stock_Competition/38_Training_Data_Set/training.csv"
selected_features_file = "output/selected_features_improved.csv"
output_dir = "output"
selected_training_file = os.path.join(output_dir, "selected_training_improved.csv")
chunksize = 10000
target_positive_samples = 10000  # Adjusted for balance


# Memory usage function
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024**3:.2f} GB")


# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load selected features
print("Loading selected features...")
features_df = pd.read_csv(selected_features_file)
selected_features = features_df["Feature"].tolist()
columns_to_keep = ["ID", "飆股"] + selected_features
print(f"Keeping {len(columns_to_keep)} columns")

# Process data in chunks
print("Processing training data...")
all_positive = []
all_negative = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    # Select columns
    chunk_filtered = chunk[columns_to_keep]

    # Compute median for numeric columns only
    numeric_cols = chunk_filtered.select_dtypes(include=["float64", "int64"]).columns
    medians = chunk_filtered[numeric_cols].median()

    # Fill missing values: numeric with median, non-numeric with default (if needed)
    chunk_filtered = chunk_filtered.copy()
    for col in numeric_cols:
        chunk_filtered[col] = chunk_filtered[col].fillna(medians[col])

    # Split into positive and negative samples
    positive_chunk = chunk_filtered[chunk_filtered["飆股"] == 1]
    negative_chunk = chunk_filtered[chunk_filtered["飆股"] == 0]
    all_positive.append(positive_chunk)
    all_negative.append(negative_chunk)
    print_memory_usage()

# Combine positive and negative samples
df_positive = pd.concat(all_positive, ignore_index=True)
df_negative = pd.concat(all_negative, ignore_index=True)
print(f"Positive samples: {len(df_positive)}, Negative samples: {len(df_negative)}")

# Prepare for balancing
X_all = pd.concat([df_positive, df_negative]).drop(columns=["ID", "飆股"])
y_all = pd.concat([df_positive, df_negative])["飆股"]

# Apply SMOTE + Tomek Links
print("Balancing classes...")
smote = SMOTE(sampling_strategy={1: target_positive_samples}, random_state=42)
tomek = TomekLinks(sampling_strategy="majority")
pipeline = Pipeline([("smote", smote), ("tomek", tomek)])
X_resampled, y_resampled = pipeline.fit_resample(X_all, y_all)
print(
    f"After SMOTE+Tomek: Positive {sum(y_resampled == 1)}, Negative {sum(y_resampled == 0)}"
)

# Create balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X_all.columns)
df_balanced["飆股"] = y_resampled
df_balanced["ID"] = range(len(df_balanced))

# Add derived features
df_balanced["FII_RSI_interaction"] = df_balanced.get(
    "外資券商_分點進出", 0
) * df_balanced.get("技術指標_週RSI(5)", 0)
df_balanced["Mainforce_Accel"] = df_balanced.get(
    "個股主力買賣超統計_近1日主力買賣超", 0
) / (df_balanced.get("個股主力買賣超統計_近5日主力買賣超", 0) + 1e-6)

# Shuffle and save
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced.to_csv(selected_training_file, index=False)
print(f"Balanced data saved to {selected_training_file}")
print_memory_usage()
print("Data preprocessing completed!")
