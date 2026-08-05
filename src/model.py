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
    objective="binary", n_estimators=400, learning_rate=0.03,
    num_leaves=31, min_child_samples=60, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=-1, n_jobs=4,
)


def walk_forward(df: pd.DataFrame, feat_cols: list[str],
                 oos_start: str = "2019-01-01", min_train: int = 5000) -> pd.DataFrame:
    """Her ay başında yeniden eğit, o ayı tahmin et. OOS tahminleri döndürür."""
    df = df.sort_values("date").reset_index(drop=True)
    months = pd.date_range(oos_start, df["date"].max(), freq="MS")
    preds = []
    for T in months:
        train = df[(df["date"] <= T - pd.Timedelta(days=EMBARGO_DAYS)) & df["y"].notna() & df["fwd_ret"].notna()]
        test = df[(df["date"] >= T) & (df["date"] < T + pd.offsets.MonthBegin(1))]
        if len(train) < min_train or test.empty:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(train[feat_cols], train["y"])
        p = m.predict_proba(test[feat_cols])[:, 1]
        out = test[["date", "asset", "aclass", "close", "fwd_ret", "y"]].copy()
        out["p_up"] = p
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def train_final(df: pd.DataFrame, feat_cols: list[str]) -> lgb.LGBMClassifier:
    """Güncel tahmin için tüm etiketli veriyle eğit."""
    train = df[df["y"].notna() & df["fwd_ret"].notna()]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(train[feat_cols], train["y"])
    return m


def latest_predictions(df: pd.DataFrame, feat_cols: list[str], model) -> pd.DataFrame:
    """Her varlığın son gözlem günü için ileriye dönük tahmin."""
    idx = df.groupby("asset")["date"].idxmax()
    last = df.loc[idx].copy()
    last["p_up"] = model.predict_proba(last[feat_cols])[:, 1]
    return last[["asset", "aclass", "date", "close", "p_up", "vol21"]].sort_values("p_up", ascending=False)
