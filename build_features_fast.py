# build_features_fast.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 原始欄位
COL_FROM = "from_acct"
COL_TO   = "to_acct"
COL_AMT  = "txn_amt"
COL_DATE = "txn_date"
COL_TIME = "txn_time"
COL_CURR = "currency_type"
COL_CH   = "channel_type"
COL_SELF = "is_self_txn"

def load_fx_factor_series(fx_json_path: str | Path) -> pd.Series:
    with open(fx_json_path, "r", encoding="utf-8") as f:
        fx = json.load(f)
    rows = []
    for cur, info in fx["rates"].items():
        unit = info["unit"]; twd = info["twd"]
        factor = float(twd) if unit == "per_1" else (float(twd)/100.0 if unit == "per_100" else 1.0)
        rows.append((cur, factor))
    s = pd.Series(dict(rows), name="fx_factor").astype(float)
    if "TWD" not in s: s.loc["TWD"] = 1.0
    return s

def value_counts_mode(series: pd.Series) -> object:
    if series.size == 0: return "UNK"
    vc = series.value_counts(dropna=True)
    return vc.index[0] if len(vc) else "UNK"

def read_acct_list(path: str | None) -> set[str] | None:
    if not path: return None
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    if p.suffix.lower() in {".txt", ".list"}:
        accts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return set(accts)
    else:
        df = pd.read_csv(p)
        col = "acct" if "acct" in df.columns else df.columns[0]
        return set(df[col].astype(str).dropna())

def aggregate_accounts(df_txn: pd.DataFrame, fx_factor: pd.Series, acct_keep: set[str] | None,
                       merge_label: bool, alert_csv: str | None) -> pd.DataFrame:
    """向量化聚合指定帳號（acct_keep=None 表示用全部帳號）"""
    df = df_txn
    if acct_keep is not None:
        mask = df[COL_FROM].astype(str).isin(acct_keep) | df[COL_TO].astype(str).isin(acct_keep)
        df = df.loc[mask].copy()

    # 換匯、時間、自轉自
    df["fx_factor"] = df[COL_CURR].map(fx_factor).fillna(1.0)
    df["amt_twd"] = df[COL_AMT].astype(float) * df["fx_factor"]
    df["hour"] = pd.to_datetime(df[COL_TIME], format="%H:%M:%S", errors="coerce").dt.hour
    df["self_flag"] = df[COL_SELF].map({"Y":1.0,"N":0.0}).astype(float)

    # 帳號視角長表
    A = pd.DataFrame({
        "acct": df[COL_FROM].astype(str),
        "partner": df[COL_TO].astype(str),
        "amt_twd": df["amt_twd"],
        "txn_date": df[COL_DATE].astype(float),
        "hour": df["hour"],
        "self_flag": df["self_flag"],
        "channel_type": df[COL_CH].astype(str),
        "currency_type": df[COL_CURR].astype(str),
        "is_out": 1
    })
    B = pd.DataFrame({
        "acct": df[COL_TO].astype(str),
        "partner": df[COL_FROM].astype(str),
        "amt_twd": df["amt_twd"],
        "txn_date": df[COL_DATE].astype(float),
        "hour": df["hour"],
        "self_flag": df["self_flag"],
        "channel_type": df[COL_CH].astype(str),
        "currency_type": df[COL_CURR].astype(str),
        "is_out": 0
    })
    G = pd.concat([A,B], ignore_index=True)
    if acct_keep is not None:
        G = G[G["acct"].isin(acct_keep)].copy()

    grp = G.groupby("acct", sort=False)

    # 基本統計
    txn_count = grp.size().rename("txn_count")
    total_amt = grp["amt_twd"].sum(min_count=1).rename("total_amt_twd")
    avg_amt   = grp["amt_twd"].mean().rename("avg_amt_twd")
    max_amt   = grp["amt_twd"].max().rename("max_amt_twd")
    std_amt   = grp["amt_twd"].std().rename("amt_twd_std")
    self_ratio= grp["self_flag"].mean().rename("self_txn_ratio")

    # 方向/金額
    incoming_mask = (G["is_out"]==0); outgoing_mask = (G["is_out"]==1)
    incoming_count = G.loc[incoming_mask].groupby("acct")["is_out"].count().rename("incoming_count")
    outgoing_count = G.loc[outgoing_mask].groupby("acct")["is_out"].count().rename("outgoing_count")
    incoming_amt   = G.loc[incoming_mask].groupby("acct")["amt_twd"].sum().rename("incoming_amt_twd_sum")
    outgoing_amt   = G.loc[outgoing_mask].groupby("acct")["amt_twd"].sum().rename("outgoing_amt_twd_sum")
    io_df = pd.concat([incoming_count, outgoing_count], axis=1).fillna(0)
    io_df["incoming_outgoing_ratio"] = io_df["incoming_count"] / io_df["outgoing_count"].replace(0, np.nan)

    # 對手/時間
    uniq_partner = grp["partner"].nunique().rename("unique_counterparty")
    hour_mean   = grp["hour"].mean().rename("hour_mean")
    night_ratio = grp["hour"].apply(lambda h: ((h>=0)&(h<6)).mean()).rename("night_ratio")

    # interarrival
    Gs = G.sort_values(["acct","txn_date"], kind="stable")
    diffs = Gs.groupby("acct")["txn_date"].diff()
    interarrival_mean = diffs.groupby(Gs["acct"]).mean().rename("interarrival_days_mean")
    interarrival_std  = diffs.groupby(Gs["acct"]).std().rename("interarrival_days_std")

    # 視窗特徵
    base_index = txn_count.index
    max_day = grp["txn_date"].transform("max"); G["max_day"] = max_day
    def win_feats(w):
        m = G["txn_date"] >= (G["max_day"] - w)
        sub = G[m]
        return (sub.groupby("acct").size().rename(f"txn_count_{w}d").reindex(base_index, fill_value=0),
                sub.groupby("acct")["amt_twd"].sum().rename(f"amt_twd_sum_{w}d").reindex(base_index, fill_value=0.0))
    cnt7,sum7 = win_feats(7); cnt30,sum30 = win_feats(30); cnt90,sum90 = win_feats(90)

    # 大額次數：median + 3*std
    stats = grp["amt_twd"].agg(["median","std"]); stats["thr"] = stats["median"] + 3*stats["std"].fillna(0)
    G2 = G[["acct","amt_twd"]].merge(stats["thr"], left_on="acct", right_index=True)
    big_flag = (G2["amt_twd"] >= G2["thr"]).astype(int)
    big_cnt = big_flag.groupby(G2["acct"]).sum().rename("big_amt_count_twd").reindex(base_index, fill_value=0).astype(int)

    # 通路/幣別比例
    unique_channel = grp["channel_type"].nunique().rename("unique_channel")
    channel_mode = grp["channel_type"].agg(value_counts_mode).rename("channel_type_mode")
    counts = G.groupby(["acct","currency_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=base_index, fill_value=0)
    tot = counts.sum(axis=1).replace(0, np.nan)
    ratio_twd = (counts.get("TWD",0)/tot).rename("ratio_twd")
    ratio_usd = (counts.get("USD",0)/tot).rename("ratio_usd")
    ratio_other = (1.0 - ratio_twd.fillna(0) - ratio_usd.fillna(0)).rename("ratio_other")

    feats = pd.concat([
        txn_count, total_amt, avg_amt, max_amt, std_amt, big_cnt,
        incoming_count.reindex(base_index, fill_value=0),
        outgoing_count.reindex(base_index, fill_value=0),
        incoming_amt.reindex(base_index, fill_value=0.0),
        outgoing_amt.reindex(base_index, fill_value=0.0),
        io_df["incoming_outgoing_ratio"].reindex(base_index),
        hour_mean, night_ratio, interarrival_mean, interarrival_std,
        cnt7, sum7, cnt30, sum30, cnt90, sum90,
        unique_channel, channel_mode, uniq_partner,
        self_ratio, ratio_twd, ratio_usd, ratio_other
    ], axis=1)

    feats.index.name="acct"; feats.reset_index(inplace=True)
    feats["channel_type_mode"] = feats["channel_type_mode"].astype("category")

    if merge_label and alert_csv:
        df_alert = pd.read_csv(alert_csv)
        if "acct" not in df_alert.columns:
            for cand in ["account","acct_id","id","from_acct","to_acct"]:
                if cand in df_alert.columns:
                    df_alert = df_alert.rename(columns={cand:"acct"}); break
        lab = df_alert[["acct"]].drop_duplicates().assign(label=1)
        feats = feats.merge(lab, on="acct", how="left").fillna({"label":0})
        feats["label"] = feats["label"].astype(int)

    return feats

def parse_args():
    ap = argparse.ArgumentParser(description="Fast builder: per-account aggregated features (vectorized).")
    ap.add_argument("--txn_csv", required=True)
    ap.add_argument("--fx_json", required=True)
    ap.add_argument("--alert_csv", required=True)
    ap.add_argument("--out_dir", default="training_data", help="輸出資料夾")
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    # 訓練/驗證資料：排除名單（通常是 acct_predict.csv）
    ap.add_argument("--exclude_acct_list", help="從 train/valid 排除的帳號（txt 或 CSV，需含 acct 欄）")
    # 預測資料：只聚合這份名單
    ap.add_argument("--acct_list_predict", help="要聚合成 predict 特徵的帳號名單（txt 或 CSV，需含 acct 欄）")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 讀交易與匯率
    df_all = pd.read_csv(args.txn_csv)
    fx_factor = load_fx_factor_series(args.fx_json)

    # ========= 產生 train/valid （排除 predict 名單）=========
    exclude_set = read_acct_list(args.exclude_acct_list)
    df_trainvalid = df_all.copy()
    if exclude_set:
        m_from = ~df_trainvalid[COL_FROM].astype(str).isin(exclude_set)
        m_to   = ~df_trainvalid[COL_TO].astype(str).isin(exclude_set)
        df_trainvalid = df_trainvalid[m_from & m_to].copy()

    feats_tv = aggregate_accounts(df_trainvalid, fx_factor, acct_keep=None, merge_label=True, alert_csv=args.alert_csv)

    if exclude_set:
        feats_tv = feats_tv[~feats_tv["acct"].isin(exclude_set)].copy()

    # split 8:2
    accts = feats_tv["acct"].tolist()
    train_accts, valid_accts = train_test_split(accts, test_size=args.valid_ratio, random_state=42)
    train_df = feats_tv[feats_tv["acct"].isin(train_accts)].copy()
    valid_df = feats_tv[feats_tv["acct"].isin(valid_accts)].copy()

    train_out = out_dir/"account_features_train.csv"
    valid_out = out_dir/"account_features_valid.csv"
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)
    print(f"[OK] train={train_out} ({len(train_df)} rows), valid={valid_out} ({len(valid_df)} rows)")

    # ========= 產生 predict （只針對名單聚合；不合併 label）=========
    if args.acct_list_predict:
        predict_set = read_acct_list(args.acct_list_predict)
        if not predict_set:
            raise ValueError("acct_list_predict 檔案為空或缺少 acct 欄")
        feats_pred = aggregate_accounts(df_all, fx_factor, acct_keep=predict_set, merge_label=False, alert_csv=None)
        pred_out = out_dir/"account_features_predict.csv"
        feats_pred.to_csv(pred_out, index=False)
        print(f"[OK] predict={pred_out} ({len(feats_pred)} rows)")

if __name__=="__main__":
    main()
