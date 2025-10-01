# agg_features.py
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---- 欄位名對照（依 acct_transaction.csv 規格表） ----
COL_FROM = "from_acct"
COL_TO = "to_acct"
COL_AMT = "txn_amt"
COL_DATE = "txn_date"    # 日序號 (1起算)
COL_TIME = "txn_time"    # HH:MM:SS
COL_CURR = "currency_type"
COL_CH = "channel_type"
COL_SELF = "is_self_txn"
# ------------------------------------------------------

def load_fx_rates(path: str | Path) -> dict:
    """載入 fx_rates.json"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["rates"]

def convert_to_twd(df: pd.DataFrame, fx_rates: dict) -> pd.Series:
    """將每筆交易金額換成 TWD"""
    twd_values = []
    for cur, amt in zip(df[COL_CURR], df[COL_AMT]):
        if cur not in fx_rates:
            # 不在表內 → 當作其他，1:1
            twd_values.append(amt)
            continue
        rate_info = fx_rates[cur]
        unit, twd_rate = rate_info["unit"], rate_info["twd"]
        if unit == "per_1":
            twd_values.append(amt * twd_rate)
        elif unit == "per_100":
            twd_values.append(amt * twd_rate / 100.0)
        else:
            twd_values.append(amt)
    return pd.Series(twd_values, index=df.index, name="amt_twd")

def aggregate_single_account(df_txn: pd.DataFrame, acct: str, fx_rates: dict) -> pd.DataFrame:
    """聚合單一 acct 的交易紀錄 → 一列特徵"""
    sub = df_txn[(df_txn[COL_FROM] == acct) | (df_txn[COL_TO] == acct)].copy()
    if sub.empty:
        return pd.DataFrame([{"acct": acct}])  # 空帳號安全處理

    # 金額換匯
    sub["amt_twd"] = convert_to_twd(sub, fx_rates)

    # 自轉自
    sub[COL_SELF] = sub[COL_SELF].map({"Y": 1, "N": 0}).astype("float")

    # ---- 基本統計 ----
    row = {
        "acct": acct,
        "txn_count": len(sub),
        "total_amt_twd": sub["amt_twd"].sum(),
        "avg_amt_twd": sub["amt_twd"].mean(),
        "max_amt_twd": sub["amt_twd"].max(),
        "amt_twd_std": sub["amt_twd"].std(),
        "self_txn_ratio": sub[COL_SELF].mean(),
    }

    # 大額次數（帳號內 median+3*std 門檻）
    thr = sub["amt_twd"].median() + 3 * (sub["amt_twd"].std() or 0)
    row["big_amt_count_twd"] = int((sub["amt_twd"] >= thr).sum())

    # ---- 方向與資金流 ----
    incoming = (sub[COL_TO] == acct)
    outgoing = (sub[COL_FROM] == acct)
    row["incoming_count"] = incoming.sum()
    row["outgoing_count"] = outgoing.sum()
    row["incoming_amt_twd_sum"] = sub.loc[incoming, "amt_twd"].sum()
    row["outgoing_amt_twd_sum"] = sub.loc[outgoing, "amt_twd"].sum()
    row["incoming_outgoing_ratio"] = row["incoming_count"] / max(1, row["outgoing_count"])

    # ---- 對手帳號 ----
    partners = pd.concat([
        sub.loc[incoming, COL_FROM],
        sub.loc[outgoing, COL_TO]
    ], axis=0)
    row["unique_counterparty"] = partners.nunique(dropna=True)

    # ---- 時間分布 ----
    # txn_time → 小時
    hours = pd.to_datetime(sub[COL_TIME], format="%H:%M:%S", errors="coerce").dt.hour
    row["hour_mean"] = hours.mean()
    row["night_ratio"] = ((hours >= 0) & (hours < 6)).mean()

    # txn_date → 相鄰間隔
    d_sorted = sub[COL_DATE].sort_values()
    if len(d_sorted) >= 2:
        gaps = d_sorted.diff().dropna()
        row["interarrival_days_mean"] = gaps.mean()
        row["interarrival_days_std"] = gaps.std()
    else:
        row["interarrival_days_mean"] = np.nan
        row["interarrival_days_std"] = np.nan

    # ---- 視窗統計 ----
    max_day = sub[COL_DATE].max()
    for days in (7, 30, 90):
        cutoff = max_day - days
        recent = sub[sub[COL_DATE] >= cutoff]
        row[f"txn_count_{days}d"] = len(recent)
        row[f"amt_twd_sum_{days}d"] = recent["amt_twd"].sum()

    # ---- 通路 ----
    row["unique_channel"] = sub[COL_CH].nunique(dropna=True)
    row["channel_type_mode"] = str(sub[COL_CH].mode().iloc[0]) if not sub[COL_CH].mode().empty else "UNK"

    # ---- 幣別比例（筆數占比）----
    total_cnt = len(sub)
    row["ratio_twd"] = (sub[COL_CURR] == "TWD").sum() / total_cnt
    row["ratio_usd"] = (sub[COL_CURR] == "USD").sum() / total_cnt
    row["ratio_other"] = 1.0 - row["ratio_twd"] - row["ratio_usd"]

    return pd.DataFrame([row])

def aggregate_many_accounts(df_txn: pd.DataFrame, acct_list: list[str], fx_rates: dict) -> pd.DataFrame:
    """多帳號聚合"""
    feats = []
    for a in acct_list:
        feats.append(aggregate_single_account(df_txn, a, fx_rates))
    out = pd.concat(feats, ignore_index=True)
    out["channel_type_mode"] = out["channel_type_mode"].astype("category")
    return out
