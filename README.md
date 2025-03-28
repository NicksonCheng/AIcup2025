# Stock Competition - 飆股預測項目

**因主辦單位要求賽後需刪除資料集，為避免爭議故本項目不提供資料集**

## 項目概述
本項目旨在利用機器學習模型，基於股票技術指標與市場數據，預測哪些股票可能成為「飆股」（標籤為 1）。項目使用 TensorFlow 與雙 GPU（RTX 3090）進行訓練，包含數據前處理、模型訓練與預測階段。

- **訓練數據**: `./38_Training_Data_Set/training.csv`（約 200,864 筆，12GB）
- **測試數據**: `./38_Public_Test_Set_and_Submmision_Template/public_x.csv`25,108 筆股票
- **根目錄**: `Stock_Competition`

---

## 專案結構
```
/home/r1419r1419/Stock_Competition/
├── 38_Public_Test_Set_and_Submmision_Template/
│   ├── public_x.csv                      # 原始測試數據
│   └── submission_template_public.csv    # 原始提交模板
├── 38_Training_Data_Set/
│   ├── headers.csv                       # 總headers
│   ├── headers.py                        # 從原始訓練數據提出headers
│   └── training.csv                      # 原始訓練數據
├── LightGBM_model_ST_voting_Optuna_transform/
│   ├── Dataset/
│   │    ├── analysis8.py                 # 特徵選擇
│   │    ├── selected_features8.csv       # 特徵選擇結果（217 個特徵）
│   │    ├── selected_public_x8.csv       # 測試集特徵選擇結果（217 個特徵）
│   │    ├── selected_training8.csv       # 訓練集特徵選擇結果（217 個特徵）
│   │    ├── testing_data_generation.py   # 測試集產生
│   │    └── training_data_generation.py  # 訓練集產生
│   ├── best_params.pkl                   # Optuna 最佳參數
│   ├── lightgbm_final.pkl                # 最終模型
│   ├── minmax_scaler.pkl                 # 標準化器
│   ├── model.py                          # 訓練模型
│   ├── predict.py                        # 模型預測
│   └── submission_optuna.csv             # 預測結果
└── README.md                             # 說明
```

---

## 進行流程

### 1. 數據前處理
#### 1.1 特徵選擇與數據過濾
- **程式**: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/analysis8.py`
- **方法**:
  - 方法 1: LightGBM 特徵重要性
  - 方法 2: 互信息
  - 方法 3: Spearman 相關性
  - 方法 4: 隨機森林特徵重要性
  - 方法 5: 皮爾森相關係數
  - 方法 6: 卡方檢定
  - 方法 7: Lasso 回歸
  - 方法 8: Auto-Encoder
  - 以上特徵各取前 **50** 個與 **飆股** 相關特徵
- **輸入**:
  - 原始數據: `./38_Training_Data_Set/training.csv`
  - 特徵列表: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_features8.csv`（取聯集 217 個特徵）
- **處理**:
  - 從原始數據中選取指定特徵（`ID`, `飆股` 與 217 個技術指標）。
  - 分塊讀取（`chunksize=15000`）以減少記憶體壓力。
- **輸出**: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_training8.csv`

#### 1.2 數據平衡
- **程式**: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/training_data_generation.py`
- **問題**: 原始數據中 `飆股=1`（正樣本）約 1,470 筆，`飆股=0`（負樣本）約 199,394 筆，比例約 0.73%，極度不平衡。
- **方法**: 
  - 保留所有正樣本（1,470 筆）。
  - 保留所有負樣本（約 199,385 筆）
  - 使用 SMOTE 生成正樣本至約 14,000 筆
  - 使用 Tomek Links 移除多數類別邊界樣本
  - 最終數據：約 14,000 正樣本 + 所有負樣本
- **輸出**: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_training8.csv`

---

### 2. 模型訓練

- 輸入數據：`./LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_features8.csv`
- 缺失值填 0

#### 2.1 環境設置
- **硬體**: 雙 RTX 3090。
- **框架**: TensorFlow，啟用 `MirroredStrategy` 支援多 GPU。
- **記憶體管理**: 設置 `set_memory_growth` 避免 CUDA 記憶體溢出。

#### 2.2 模型架構
- **程式**: `./LightGBM_model_ST_voting_Optuna_transform/model.py`
- **模型**: LightGBM

- **特徵轉換**：
  - 應用小波轉換 (db1) 於：
    - 上市加權指數前 1~20 天收盤價（索引 51-70）
    - 成交量（索引 63-82）
  - 新增約 40 個小波特徵（cA + cD）

- **標準化**：
  - MinMaxScaler (0~1)
  - 儲存為 `./LightGBM_model_ST_voting_Optuna_transform/minmax_scaler.pkl`

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
  - 儲存模型：`./LightGBM_model_ST_voting_Optuna_transform/lightgbm_final.pkl`

---

### 3. 預測
- **程式**: `predict.py`
- **輸入**: 
  - 測試數據: `./LightGBM_model_ST_voting_Optuna_transform/Dataset/selected_public_x8.csv`
  - 缺失值填 0
  - 特徵轉換：應用小波轉換（共 257 個特徵）
  - 標準化：使用 `minmax_scaler.pkl`
  - 載入模型：`lightgbm_final.pkl`
- **預測**：
  - 動態調整閾值（0.1~0.5）
  - 目標正樣本數：176
- **輸出**: `./LightGBM_model_ST_voting_Optuna_transform/submission_optuna.csv`
  - 格式: `ID, 飆股`（0 或 1）

---

## 結果分析

### 結果（手動閾值 0.03）
- **測試集**: 25,108 筆
- **方法**:
  - 特徵：217 個（8 種方法提取）+ 小波轉換（總 258）
  - 數據處理：SMOTE + Tomek Links，正樣本 14,000，負樣本全用（約 199,385）
  - 模型：LightGBM + Optuna + 10 折驗證
  - 預測閾值：0.03
  - 正樣本數：177
- **推算**:
  - TP = 0.8079 * 177 ≈ 143。
  - 實際正樣本 = 143 / 0.8125 ≈ 176。
  - FP = 177 - 143 = 34。
  - FN = 176 - 143 = 33。
- **成績**:
  - Precision: 0.8079
  - Recall: 0.8125
  - F1-score: 0.8102
- **分析**:
  - 高召回率（93.18%）：捕捉大部分正樣本（約 164/176）
  - 低精確率（6.48%）：預測正樣本 2,531 個，假陽性約 2,367
  - F1 分數偏低，因精確率拖累
- **問題**: 
  - Recall (0.8125) 和 Precision (0.8079) 均低於目標 0.9
  - FN = 33，漏掉約 19% 正樣本；FP = 34，影響 Precision

---

## 下一步計畫

### 🎯 目標分析
- **目標 F1 = 0.9**
  - 預估需提升：
    - TP 增加約 15 個
    - FP 減少約 16 個
    - FN 減少約 15 個

---

### 1. 微調閾值

- **問題**：  
  閾值 0.03 時，正樣本數接近 176，但 FP 偏高（34）。

- **建議**：  
  測試 0.04 - 0.06 之間的閾值，尋找 Precision 和 Recall 的平衡點。

- **示例**：
  ```python
  manual_threshold = 0.04  # 試 0.04, 0.045, 0.05
  test_preds = (test_preds_proba > manual_threshold).astype(int)
  ```

- **預期結果**：
  - 正樣本數維持 170~180
  - FP 減至 20~25
  - F1 提升至 0.82~0.85

---

### 2. 增加正樣本數量

- **問題**：  
  現有正樣本 14,000（占比 6.6%）仍可能不足以捕捉特性。

- **建議**：  
  使用 SMOTE 增至 20,000，控制正負比為 1:10。

- **預期結果**：
  - Recall 提升至 0.85~0.9
  - F1 提升至 0.83~0.87

---

### 3. 特徵篩選

- **建議**：  
  使用 LightGBM 的 `feature_importance()` 篩選前 150~200 個重要特徵。

- **示例**：
  ```python
  importances = model.feature_importance()
  top_indices = np.argsort(importances)[-150:]
  X_test_scaled = X_test_scaled[:, top_indices]
  ```

- **注意**：  
  訓練集也應同步進行特徵選擇。

- **預期結果**：
  - Precision 提升至 0.85~0.9
  - F1 提升至 0.85~0.87

---

### 4. Stacking 集成

- **建議**：  

- **模型結構**：
  - 基模型 1：
  - 基模型 2：
  - 元模型：

- **預期結果**：
  - Recall 提升至 0.9
  - Precision 維持 0.85~0.9
  - F1 提升至 0.87~0.9

---

### 5. 調整訓練參數

- **建議**：  
  在 Optuna 中加大 `scale_pos_weight`（建議範圍 50~300），提升正樣本權重。

- **預期結果**：
  - Recall 提升 0.05~0.1
  - F1 提升至 0.83~0.85