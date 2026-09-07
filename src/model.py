"""GraphFinance — pooled LightGBM + walk-forward doğrulama.
Aylık yeniden eğitim, embargo ile sızıntı önleme:
  T ayında eğitim: feature tarihi <= T - 10 takvim günü (etiket t+5 işlem günü bittiğinden emin)
  T ayında tahmin: T <= tarih < T+1 ay  (tamamen out-of-sample)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

EMBARGO_DAYS = 10  # takvim günü; 5 işlem günlük etiket ufkunu güvenle kapsar

PARAMS = dict(
    objective="binary", n_estimators=250, learning_rate=0.04,
    num_leaves=31, min_child_samples=60, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=-1, n_jobs=-1,
)

RETRAIN_MONTHS = 3  # Actions calisma suresi icin ceyreklik yeniden egitim


def walk_forward(df: pd.DataFrame, feat_cols: list[str],
                 oos_start: str = "2019-01-01", min_train: int = 5000,
                 target: str = "y") -> pd.DataFrame:
    """Her çeyrek başında yeniden eğit, o çeyreği tahmin et. OOS tahminleri döndürür.
    target='y' mutlak yön; target='y_rel' göreli (sınıf medyanına göre) yön — çıktı sütunları
    aynı ('y', 'fwd_ret', 'p_up') ki metrik kodu ortak kalsın; göreli modelde fwd_ret = fwd_rel."""
    ret_col = "fwd_rel" if target == "y_rel" else "fwd_ret"
    df = df.sort_values("date").reset_index(drop=True)
    months = pd.date_range(oos_start, df["date"].max(), freq=f"{RETRAIN_MONTHS}MS")
    preds = []
    for T in months:
        train = df[(df["date"] <= T - pd.Timedelta(days=EMBARGO_DAYS)) & df[target].notna() & df[ret_col].notna()]
        test = df[(df["date"] >= T) & (df["date"] < T + pd.offsets.MonthBegin(RETRAIN_MONTHS))]
        if len(train) < min_train or test.empty:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(train[feat_cols], train[target].astype(int))
        p = m.predict_proba(test[feat_cols])[:, 1]
        out = test[["date", "asset", "aclass", "close"]].copy()
        out["fwd_ret"] = test[ret_col].values
        out["y"] = test[target].values
        out["p_up"] = p
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def train_final(df: pd.DataFrame, feat_cols: list[str], target: str = "y") -> lgb.LGBMClassifier:
    """Güncel tahmin için tüm etiketli veriyle eğit."""
    train = df[df[target].notna() & df["fwd_ret"].notna()]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(train[feat_cols], train[target].astype(int))
    return m


def naive_hit(oos: pd.DataFrame) -> float:
    """'Hep cogunluk sinifi' tabani: modelin gercek kenari = isabet - bu."""
    g = oos.dropna(subset=["y"])
    return float(max(g["y"].mean(), 1 - g["y"].mean())) if len(g) else float("nan")


def latest_predictions(df: pd.DataFrame, feat_cols: list[str], model) -> pd.DataFrame:
    """Her varlığın son gözlem günü için ileriye dönük tahmin + sürücü açıklamaları."""
    idx = df.groupby("asset")["date"].idxmax()
    last = df.loc[idx].copy()
    last["p_up"] = model.predict_proba(last[feat_cols])[:, 1]
    last["drivers"] = top_drivers(model, last, feat_cols)
    return (last[["asset", "aclass", "date", "close", "p_up", "vol21", "drivers"]]
            .sort_values("p_up", ascending=False))


def top_drivers(model, X: pd.DataFrame, feat_cols: list[str], k: int = 3) -> list[str]:
    """Tahmin başına en etkili k özellik (SHAP benzeri katkı). 'feat:+' / 'feat:-' listesi."""
    contrib = model.booster_.predict(X[feat_cols].to_numpy(dtype=float), pred_contrib=True)
    out = []
    for row in contrib:
        vals = row[:-1]  # son sütun bias
        order = np.argsort(-np.abs(vals))[:k]
        out.append("|".join(f"{feat_cols[i]}:{'+' if vals[i] > 0 else '-'}" for i in order))
    return out
