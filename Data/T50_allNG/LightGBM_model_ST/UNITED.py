import pandas as pd

# 讀取兩個 CSV 檔案
file1_path = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST_voting_Optuna/submission_optuna_test.csv"
file2_path = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST/submission_BT.csv"
output_path = "/home/r1419r1419/Stock_Competition/Data/T50_allNG/LightGBM_model_ST/submission_UNITED3.csv"

# 讀取檔案
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 假設 '飆股' 是要進行聯集的欄位，提取這一欄的唯一值
set1 = set(df1['飆股'])
set2 = set(df2['飆股'])

# 計算聯集
union_set = set1.union(set2)

# 找出只有在第一個檔案中的值
only_in_file1 = set1 - set2
# 找出只有在第二個檔案中的值
only_in_file2 = set2 - set1

# 輸出結果
print("只有在第一個檔案中的飆股值:", only_in_file1)
print("只有在第二個檔案中的飆股值:", only_in_file2)

# 合併兩個 DataFrame，並移除重複的行
united_df = pd.concat([df1, df2]).drop_duplicates()

# 儲存聯集結果到新的 CSV 檔案
united_df.to_csv(output_path, index=False)
print(f"聯集結果已儲存到: {output_path}")