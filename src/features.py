"""GraphFinance — özellik mühendisliği (sadece kapanış fiyatı gerektirir).
Tüm özellikler t anındaki ve öncesi veriyi kullanır; hedef t→t+H yönü.
"""
import numpy as np
import pandas as pd

HORIZON = 5  # işlem günü


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _asset_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]
    lr = np.log(c).diff()
    g["r1"] = lr
    for k in (5, 10, 21, 63):
        g[f"r{k}"] = np.log(c / c.shift(k))
    g["vol21"] = lr.rolling(21).std()
    g["vol_ratio"] = lr.rolling(5).std() / g["vol21"]
    g["rsi14"] = _rsi(c)
    for k in (20, 50, 200):
        g[f"sma{k}_gap"] = c / c.rolling(k).mean() - 1
    g["hi52_dist"] = c / c.rolling(252).max() - 1
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    g["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    # hedef: t -> t+H log getiri yönü (SADECE gelecek, sızıntı yok)
    g["fwd_ret"] = np.log(c.shift(-HORIZON) / c)
    g["y"] = (g["fwd_ret"] > 0).astype(int)
    return g


def build_features(panel: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = pd.concat([_asset_feats(g) for _, g in panel.groupby("asset")], ignore_index=True)

    # --- piyasa genel özellikleri (her tarih için, sadece geçmiş) ---
    vix_df = pd.DataFrame({"vix": vix})
    vix_df["vix_z"] = (vix_df["vix"] - vix_df["vix"].rolling(252).mean()) / vix_df["vix"].rolling(252).std()
    vix_df["vix_chg5"] = vix_df["vix"].pct_change(5)
    df = df.merge(vix_df[["vix_z", "vix_chg5"]], left_on="date", right_index=True, how="left")

    for feat_asset, colname in [("WTI", "oil_r5"), ("EURUSD", "eur_r5"), ("BTC", "btc_r21")]:
        s = (df[df["asset"] == feat_asset].set_index("date")["r5" if "r5" in colname else "r21"]
             .rename(colname))
        df = df.merge(s, left_on="date", right_index=True, how="left")

    # piyasa özellikleri hafta sonu/tatil boşluklarında son değerle doldurulur (geçmişe bakmaz)
    df = df.sort_values(["asset", "date"])
    for c in ("vix_z", "vix_chg5", "oil_r5", "eur_r5", "btc_r21"):
        df[c] = df.groupby("asset")[c].ffill()

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    for cls in ("crypto", "fx", "commodity", "stock"):
        df[f"is_{cls}"] = (df["aclass"] == cls).astype(int)

    feat_cols = [c for c in df.columns if c not in
                 ("date", "asset", "aclass", "close", "fwd_ret", "y")]
    # temel özellikler eksikse satırı at (ilk ~252 gün)
    core = ["r1", "r21", "vol21", "rsi14", "sma200_gap", "hi52_dist"]
    df = df.dropna(subset=core)
    return df.reset_index(drop=True), feat_cols


if __name__ == "__main__":
    from fetch import load_panel
    panel, vix = load_panel()
    df, cols = build_features(panel, vix)
    print(df.shape, "features:", cols)
    print(df.groupby("asset")["date"].agg(["min", "max", "size"]))
