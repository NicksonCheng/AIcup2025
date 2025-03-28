import pandas as pd
TRAIN_PATH = "../Stock_Competition/38_Training_Data_Set/training.csv"
df = pd.read_csv(TRAIN_PATH, nrows=10000)  # 先讀少量數據
print(df.describe())
df = df.drop(columns=['ID'])
print("=========================")
print(df.isna().sum())
print(df.columns[df.var() == 0])