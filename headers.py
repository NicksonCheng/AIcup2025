import csv

def extract_headers(input_csv, output_csv):
    """
    讀取指定的 CSV 檔案，提取所有的欄位名稱（header），並儲存至另一個 CSV 檔案。
    
    :param input_csv: str, 輸入的 CSV 檔案名稱
    :param output_csv: str, 輸出的 CSV 檔案名稱
    """
    try:
        with open(input_csv, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)  # 取得第一行作為 header
        
        with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)  # 寫入 header 至新 CSV 檔案
        
        print(f"Header 已成功儲存至 {output_csv}")
    except Exception as e:
        print(f"發生錯誤: {e}")

# 使用範例
input_csv = "Stock_Competition/38_Training_Data_Set/training.csv"  # 請替換為你的 CSV 檔案
output_csv = "Stock_Competition/38_Training_Data_Set/headers.csv"  # 產出的檔案名稱
extract_headers(input_csv, output_csv)