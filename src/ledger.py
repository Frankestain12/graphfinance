# -*- coding: utf-8 -*-
"""Tahmin defteri — sistemin hafızası.
Her çalıştırmada: (1) yeni tahminler eklenir, (2) ufku dolan eski tahminler
gerçekleşen fiyatla çözülür. Hiçbir satır silinmez.
"""
import os
import numpy as np
import pandas as pd

COLS = ["made_on", "asset", "aclass", "close", "p_up", "direction",
        "horizon_td", "resolved", "resolve_date", "realized_ret", "correct"]


def load_ledger(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["made_on", "resolve_date"])
        return df
    return pd.DataFrame(columns=COLS)


def append_predictions(led: pd.DataFrame, preds: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    new = preds.rename(columns={"date": "made_on"}).copy()
    new["direction"] = np.where(new["p_up"] >= 0.5, "up", "down")
    new["horizon_td"] = horizon
    new["resolved"] = 0
    new["resolve_date"] = pd.NaT
    new["realized_ret"] = np.nan
    new["correct"] = np.nan
    new = new[["made_on", "asset", "aclass", "close", "p_up", "direction",
               "horizon_td", "resolved", "resolve_date", "realized_ret", "correct"]]
    # ayni gun ayni varlik icin tekrar ekleme
    key = led["made_on"].astype(str) + "|" + led["asset"] if len(led) else pd.Series(dtype=str)
    new_key = new["made_on"].astype(str) + "|" + new["asset"]
    new = new[~new_key.isin(set(key))]
    return pd.concat([led, new], ignore_index=True)


def resolve(led: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Ufku dolan tahminleri panel fiyatlarıyla çöz."""
    if led.empty:
        return led
    led = led.copy()
    px = {a: g.set_index("date")["close"].sort_index() for a, g in panel.groupby("asset")}
    for i, r in led[led["resolved"] == 0].iterrows():
        s = px.get(r["asset"])
        if s is None:
            continue
        future = s[s.index > r["made_on"]]
        if len(future) >= r["horizon_td"]:
            end_px = future.iloc[int(r["horizon_td"]) - 1]
            ret = float(np.log(end_px / r["close"]))
            led.at[i, "resolved"] = 1
            led.at[i, "resolve_date"] = future.index[int(r["horizon_td"]) - 1]
            led.at[i, "realized_ret"] = ret
            led.at[i, "correct"] = int((ret > 0) == (r["direction"] == "up"))
    return led


def rolling_hit(led: pd.DataFrame, window_days: int = 180) -> pd.DataFrame:
    """Kenar bekçisi girdisi: varlık başına son N gündeki gerçekleşen isabet."""
    done = led[led["resolved"] == 1].copy()
    if done.empty:
        return pd.DataFrame(columns=["asset", "n", "hit"])
    cutoff = done["made_on"].max() - pd.Timedelta(days=window_days)
    done = done[done["made_on"] >= cutoff]
    g = done.groupby("asset")["correct"].agg(["size", "mean"]).reset_index()
    g.columns = ["asset", "n", "hit"]
    return g
