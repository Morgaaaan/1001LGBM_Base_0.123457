# inference.py
import argparse
import json
from pathlib import Path
import pandas as pd
from model import ModelLGBM
from agg_features import aggregate_single_account, aggregate_many_accounts, load_fx_rates

def load_feature_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    feats = obj.get("features", {})
    return {
        "numeric": feats.get("numeric", []),
        "categorical": feats.get("categorical", []),
        "label": obj.get("label"),
        "id_cols": obj.get("id_cols", []),
        "drop": obj.get("drop", []),
    }

def apply_spec(df: pd.DataFrame, spec: dict, for_training: bool = False):
    df = df.copy()
    for col in spec.get("drop", []) or []:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    for col in spec.get("categorical", []) or []:
        if col in df.columns:
            df[col] = df[col].astype("category")
    feature_cols = (spec.get("numeric", []) or []) + (spec.get("categorical", []) or [])
    X = df[feature_cols]
    return X

def parse_args():
    ap = argparse.ArgumentParser(description="Inference for per-account model")
    ap.add_argument("--model_path", required=True, help="模型檔 .pkl")
    ap.add_argument("--feature_spec", required=True, help="feature_spec.json")
    ap.add_argument("--output_dir", default="./output/predictions", help="輸出資料夾")
    ap.add_argument("--threshold", type=float, default=0.5, help="二值化門檻")

    # 三種模式
    ap.add_argument("--input_csv", help="已聚合好的特徵表")
    ap.add_argument("--acct", help="單帳號 ID (需 txn_csv, fx_json)")
    ap.add_argument("--acct_list", help="帳號名單 CSV (需 txn_csv, fx_json)，必須含 'acct' 欄")
    ap.add_argument("--txn_csv", help="原始交易檔 acct_transaction.csv")
    ap.add_argument("--fx_json", help="fx_rates.json 匯率表")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 載入模型與規格
    model = ModelLGBM(); model.load(args.model_path)
    spec = load_feature_spec(args.feature_spec)

    # ===== 準備輸入資料 =====
    raw_for_ids = None
    if args.input_csv:
        # 批次模式
        df = pd.read_csv(args.input_csv)
        raw_for_ids = df.copy()
        X = apply_spec(df, spec, for_training=False)

    elif args.acct:
        # 單帳號 on-the-fly
        if not (args.txn_csv and args.fx_json):
            raise ValueError("單帳號模式需要 --txn_csv 與 --fx_json")
        df_txn = pd.read_csv(args.txn_csv)
        fx_rates = load_fx_rates(args.fx_json)
        feat_row = aggregate_single_account(df_txn, args.acct, fx_rates)
        raw_for_ids = feat_row.copy()
        X = apply_spec(feat_row, spec, for_training=False)

    elif args.acct_list:
        # 帳號名單模式
        if not (args.txn_csv and args.fx_json):
            raise ValueError("帳號名單模式需要 --txn_csv 與 --fx_json")
        acct_df = pd.read_csv(args.acct_list)
        if "acct" not in acct_df.columns:
            raise ValueError("acct_list 檔案必須有一欄叫 'acct'")
        acct_list = acct_df["acct"].dropna().astype(str).tolist()

        df_txn = pd.read_csv(args.txn_csv)
        fx_rates = load_fx_rates(args.fx_json)
        feat_df = aggregate_many_accounts(df_txn, acct_list, fx_rates)
        raw_for_ids = feat_df.copy()
        X = apply_spec(feat_df, spec, for_training=False)

    else:
        raise ValueError("必須指定 --input_csv 或 --acct 或 --acct_list")

    # ===== 推論 =====
    probs = model.predict_proba(X)
    thr = float(args.threshold)
    preds = (probs >= thr).astype(int)

    # 儲存 predict_raw.csv
    pd.DataFrame({"prob": probs}).to_csv(out_dir/"predict_raw.csv", index=False)

    # 儲存 predict.csv（包含 acct）
    out_df = pd.DataFrame()
    id_cols = spec.get("id_cols") or []
    for c in id_cols:
        if c in raw_for_ids.columns:
            out_df[c] = raw_for_ids[c]
    out_df["label"] = preds
    out_df.to_csv(out_dir/"predict.csv", index=False)

    print(f"[INFO] predict_raw.csv saved: {out_dir/'predict_raw.csv'}")
    print(f"[INFO] predict.csv saved: {out_dir/'predict.csv'}")
    print(f"[INFO] threshold={thr}, #1s={int(preds.sum())}/{len(preds)}")

if __name__ == "__main__":
    main()
