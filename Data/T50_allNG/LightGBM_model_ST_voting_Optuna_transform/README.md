# LightGBM 飆股預測模型

## 專案概述
本專案旨在使用 LightGBM 模型預測飆股，結合特徵選擇、小波轉換與 SMOTE + Tomek Links 數據平衡技術，目標是提升模型的 F1-score 至 0.9。特徵從 8 種方法中提取，共 217 個，並保留全部負樣本進行訓練。

- **日期**: 2025 年 3 月 27 日  
- **目標**: F1-score 達到 0.9  
- **當前最佳**: F1 = 0.7427

---

## 環境需求
- **Python**: 3.8+
- **依賴庫**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imblearn`
  - `lightgbm`
  - `optuna`
  - `pywt`
  - `psutil`

安裝依賴：
```bash
pip install -r requirements.txt
```

---

## 專案結構
```
/home/r1419r1419/Stock_Competition/
├── 38_Training_Data_Set/
│   └── training.csv                      # 原始訓練數據
├── Data/
│   └── T50_allNG/
│       ├── selected_features8.csv        # 特徵選擇結果（217 個特徵）
│       └── LightGBM_model_ST_voting_Optuna_transform/
│           ├── selected_training8_ST.csv  # 處理後訓練數據
│           ├── minmax_scaler.pkl          # 標準化器
│           ├── best_params.pkl            # Optuna 最佳參數
│           ├── lightgbm_final.pkl         # 最終模型
│           └── submission_optuna.csv      # 預測結果
├── preprocess.py                         # 數據預處理腳本
├── train.py                              # 模型訓練腳本
└── predict.py                            # 預測腳本
```

---

## 數據處理流程（`preprocess.py`）

- **特徵讀取**：
  - 檔案：`selected_features8.csv`
  - 保留欄位：`ID`、`飆股`、特徵

- **數據處理**：
  - 檔案：`training.csv`，每塊 10,000 行
  - 缺失值填補：0
  - 分離樣本：正樣本（飆股=1）與負樣本（飆股=0）

- **數據平衡**：
  - 保留所有負樣本（約 199,385 筆）
  - 使用 SMOTE 生成正樣本至約 14,000 筆
  - 使用 Tomek Links 移除多數類別邊界樣本
  - 最終數據：約 14,000 正樣本 + 稍少的負樣本

- **輸出檔案**：
  - `selected_training8_ST.csv`

執行：
```bash
python preprocess.py
```

---

## 模型訓練流程（`train.py`）

- 輸入數據：`selected_training8_ST.csv`
- 缺失值填 0

- **特徵轉換**：
  - 應用小波轉換 (db1) 於：
    - 上市加權指數前 1~20 天收盤價（索引 51-70）
    - 成交量（索引 63-82）
  - 新增約 40 個小波特徵（cA + cD）

- **標準化**：
  - MinMaxScaler (0~1)
  - 儲存為 `minmax_scaler.pkl`

- **Optuna 優化**：
  - 目標：最大化 F1-score
  - 交叉驗證：10 折
  - 試驗次數：100 次
  - 參數範圍：
    - `learning_rate`: 0.005 ~ 0.2
    - `max_depth`: 3 ~ 15
    - `num_leaves`: 15 ~ 63
    - `scale_pos_weight`: 5 ~ 200
    - `min_child_samples`: 5 ~ 50
  - 儲存最佳參數為 `best_params.pkl`

- **最終訓練**：
  - 使用最佳參數訓練 LightGBM
  - 儲存模型：`lightgbm_final.pkl`

執行：
```bash
python train.py
```

---

## 預測流程（`predict.py`）

- 輸入測試資料：`selected_public_x8.csv`
- 缺失值填 0
- 特徵轉換：應用小波轉換（共 257 個特徵）
- 標準化：使用 `minmax_scaler.pkl`
- 載入模型：`lightgbm_final.pkl`

- **預測**：
  - 動態調整閾值（0.1~0.5）
  - 目標正樣本數：176

- **輸出結果**：`submission_optuna.csv`

執行：
```bash
python predict.py
```

---

## 使用說明

- 數據預處理：
  ```bash
  python preprocess.py
  ```

- 訓練模型：
  ```bash
  python train.py
  ```

- 生成預測：
  ```bash
  python predict.py
  ```

- 提交結果：檢查 `submission_optuna.csv`

---

## 注意事項

- **記憶體需求**：25~30 GB
- **計算時間**：
  - 預處理：1~2 小時
  - 訓練：6~8 小時（Optuna 100 次試驗）
  - 預測：10~20 分鐘
- **特徵限制**：僅使用 217 個預選特徵，無原始時間序列

---

## 下一步計畫

- 提升 F1-score
  - 若 F1 < 0.85，則篩選特徵至 150 個
  - 嘗試 ensemble（聯集）或 stacking 方法

- **加速 Optuna**：
  - 折數降至 5 折
  - 試驗次數降至 50
