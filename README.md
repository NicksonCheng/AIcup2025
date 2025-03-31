# Stock Competition - 飆股預測項目

**因主辦單位要求賽後需刪除資料集，為避免爭議故本項目不提供資料集**

## 項目概述
本項目旨在利用機器學習模型，基於股票技術指標與市場數據，預測哪些股票可能成為「飆股」（標籤為 1）。項目使用 TensorFlow 與雙 GPU（RTX 4090）進行訓練，包含數據前處理、模型訓練與預測階段。

- **訓練數據**: `./38_Training_Data_Set/training.csv`（約 200,864 筆，12GB）
- **測試數據**: 25,108 筆股票
- **根目錄**: `Stock_Competition`

---

### 檔案內容
1. ** catboost_classifier_gpu_model.py **:
   - ** 使用 GPU 進行 optuna 選參，218個features單次 trail 須跑3~4分鐘；depth=10的情況下，VRAM約為6G，以下為當前最好三組參數。預測正樣本約為80-100，可進一步對depth做調整。
   - 0.9827961505953834 - {'learning_rate': 0.19985445314053876, 'depth': 10, 'l2_leaf_reg': 1.1912568427873744}
   - 0.9825979796769116 - {'learning_rate': 0.13729598218654218, 'depth': 10, 'l2_leaf_reg': 1.3600113909655966}
   - 0.9819134704647062 - {'learning_rate': 0.08457584692887682, 'depth': 10, 'l2_leaf_reg': 1.9359795254566103}
2. ** training_data_generation.py **:
--- 使用 ADASYN(邊界), SMOTE(均勻) 生成正樣本

## 下一步建議

