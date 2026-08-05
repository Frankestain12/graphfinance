"""GraphFinance — dürüst değerlendirme.
- İsabet oranı vs 'hep yukarı' baseline (drift avantajını ayıklar)
- AUC, yüksek güvenli (p>0.55 / p<0.45) kovalar
- Basit strateji: p>esik -> H gün long, maliyet dahil; karşılaştırma buy&hold
Not: 5 günlük ufuklar örtüşür; strateji simülasyonu örtüşmesiz (her H günde bir karar) yapılır.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = 5
COST = {"crypto": 0.0010, "fx": 0.0002, "commodity": 0.0005}  # tek yön, oran


def per_asset_metrics(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in oos.dropna(subset=["y", "fwd_ret"]).groupby("asset"):
        g = g.sort_values("date")
        base = max(g["y"].mean(), 1 - g["y"].mean())  # çoğunluk sınıfı
        pred_dir = (g["p_up"] > 0.5).astype(int)
        hit = (pred_dir == g["y"]).mean()
        try:
            auc = roc_auc_score(g["y"], g["p_up"])
        except ValueError:
            auc = np.nan
        conf = g[(g["p_up"] > 0.55) | (g["p_up"] < 0.45)]
        conf_dir = (conf["p_up"] > 0.5).astype(int)
        conf_hit = (conf_dir == conf["y"]).mean() if len(conf) else np.nan

        # örtüşmesiz strateji: her H günde bir, p>0.55 ise long
        s = g.iloc[::H]
        cost = COST[g["aclass"].iloc[0]]
        strat_r = np.where(s["p_up"] > 0.55, s["fwd_ret"] - 2 * cost, 0.0)
        bh_r = s["fwd_ret"].values
        ann = 252 / H
        def sharpe(x):
            return np.mean(x) / np.std(x) * np.sqrt(ann) if np.std(x) > 0 else np.nan
        eq = np.exp(np.cumsum(strat_r))
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        rows.append(dict(
            asset=asset, aclass=g["aclass"].iloc[0], n_oos=len(g),
            date_start=g["date"].min().date(), date_end=g["date"].max().date(),
            base_hit=round(base, 4), hit=round(hit, 4), auc=round(auc, 4),
            n_conf=len(conf), conf_hit=round(conf_hit, 4) if len(conf) else None,
            strat_totret=round(float(np.exp(strat_r.sum()) - 1), 4),
            bh_totret=round(float(np.exp(bh_r.sum()) - 1), 4),
            strat_sharpe=round(float(sharpe(strat_r)), 2),
            bh_sharpe=round(float(sharpe(bh_r)), 2),
            strat_maxdd=round(float(dd), 4),
            time_in_market=round(float((s["p_up"] > 0.55).mean()), 3),
        ))
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def calibration(oos: pd.DataFrame) -> pd.DataFrame:
    g = oos.dropna(subset=["y"])
    bins = pd.cut(g["p_up"], [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0])
    return g.groupby(bins, observed=True).agg(
        n=("y", "size"), gercek_yukari_orani=("y", "mean"),
        ort_tahmin=("p_up", "mean")).round(4)
