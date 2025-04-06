# Stock Competition - 飆股預測項目

**因主辦單位要求賽後需刪除資料集，為避免爭議故本項目不提供資料集**

## 項目概述
本項目旨在利用機器學習模型，基於股票技術指標與市場數據，預測哪些股票可能成為「飆股」（標籤為 1）。項目使用 TensorFlow 與雙 GPU（RTX 4090）進行訓練，包含數據前處理、模型訓練與預測階段。

- **訓練數據**: `./38_Training_Data_Set/training.csv`（約 200,864 筆，12GB）
- **測試數據**: 25,108 筆股票
- **根目錄**: `Stock_Competition`

---

### 檔案內容
1. **catboost_classifier_gpu_model.py**:
   - 使用 GPU 進行 optuna 選參，218個features 單次 trail 須跑3~4分鐘；depth=10的情況下，VRAM約為6G，以下為當前最好三組參數。預測正樣本約為80-100。
   - 0.9827961505953834 - {'learning_rate': 0.19985445314053876, 'depth': 10, 'l2_leaf_reg': 1.1912568427873744}
   
2. **training_data_generation.py**:
   - 使用 ADASYN(邊界), SMOTE(均勻) 生成正樣本 + Tomek Links 下採樣
   
## 結果分析
Precision =\cfrac{TP}{TP+FP}
Recall =\cfrac{TP}{TP+FN}
### 使用ADASYN+SMOTE+TomekLinks 生成資料
1.  **catboost_classifier預測82個正樣本**:
   - 使用參數{'learning_rate': 0.147296968768365, 'depth': 14, 'l2_leaf_reg': 2.144381755329559}
   - 預測結果：Precision 0.9512 , Recall 0.4432
   - 預測正樣本約錯4個，可容許範圍內。下一步可結合其他模型增加預測正樣本數，或是篩選出FP。

2.  **使用Classifier(Catboost,LGBMBoost,XGBoost)個別預測取聯集，預測68個正樣本**
   - 預測結果：Precision 1 , Recall 0.3864
   - 預測正樣本全對，可以針對此結果進一步加強最後模型權重。
     
3.  **上一個方法取交集，預測113個正樣本**
   - 預測結果：Precision 0.9558 , Recall 0.6136
   - 預測正樣本約錯5個，LGBM(98)與 XGB(95)預測比 Catboost更精準，且有微小差異性(相較1.都再多抓1X個）。可 以此為基準，繼續增加差異性(個別擅長範圍)，或用其他方法抓更多正樣本做篩選。
     
## 下一步建議
- 218個特徵使用分類的Boost目前上限應該為0.75，可以增加資料多樣性(更多特徵、特徵轉換 或 不同採樣方法)提高結果。

