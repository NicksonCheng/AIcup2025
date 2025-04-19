# run lgb
nohup python train_model.py > log/feature_selection_$(date +\%Y-\%m-\%d-\%H\%M\%S).txt 2>&1 &
# run xgboost
#nohup python xgboost_model.py > log/xgboost_$(date +\%Y-\%m-\%d-).txt 2>&1 &
