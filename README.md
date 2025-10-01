# 1001LGBM_Base_0.123457
# AI CUP LGBM Baseline

本專案使用 LightGBM 建立帳戶風險預測模型，流程包含：
1. 特徵聚合
2. 訓練與驗證
3. 推論（預測帳戶標籤）



使用 `build_features_fast.py` 一次生成三份資料：
- `account_features_train.csv`  
- `account_features_valid.csv`  
- `account_features_predict.csv`

- 指令：<br>

```powershell
python build_features_fast.py `
  --txn_csv .\dataset\acct_transaction.csv `
  --fx_json .\fx_rates.json `
  --alert_csv .\dataset\acct_alert.csv `
  --exclude_acct_list .\dataset\acct_predict.csv `
  --acct_list_predict .\dataset\acct_predict.csv `
  --out_dir .\training_data `
  --valid_ratio 0.2

```powershell
python training.py `
  --train_csv .\training_data\account_features_train.csv `
  --valid_csv .\training_data\account_features_valid.csv `
  --feature_spec .\feature_spec.json `
  --output_dir .\output\lgbm_v1 `
  --learning_rate 0.05 `
  --n_estimators 5000 `
  --num_leaves 63 `
  --min_child_samples 100 `
  --max_bin 127 `
  --early_stopping_rounds 200 `
  --eval_metric auc `
  --verbose 50
