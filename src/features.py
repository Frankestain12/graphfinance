# -*- coding: utf-8 -*-
"""GraphFinance — özellik mühendisliği.
v2: grafik-okuma paketi (hacim, destek/direnç, sıkışma, rejim, çapraz-varlık).
Tüm özellikler t anındaki ve öncesi veriyi kullanır; hedef t→t+H yönü.
BASE_FEATS = v1 şampiyonu; EXT_FEATS = v1 + grafik-okuma (rakip).
"""
import numpy as np
import pandas as pd

HORIZON = 5  # işlem günü
FEATURES_VERSION = "v8-haber"  # değişince aylık doğrulama + A/B yeniden tetiklenir

FEAT_TR = {
    "r1": "dünkü getiri", "r5": "5 günlük momentum", "r10": "10 günlük momentum",
    "r21": "aylık momentum", "r63": "3 aylık momentum",
    "vol21": "21 günlük oynaklık", "vol_ratio": "kısa/uzun oynaklık oranı",
    "rsi14": "RSI (aşırı alım/satım)", "sma20_gap": "20g ortalamaya uzaklık",
    "sma50_gap": "50g ortalamaya uzaklık", "sma200_gap": "200g ortalamaya uzaklık",
    "hi52_dist": "52 haftalık zirveye uzaklık", "macd_hist": "MACD histogramı",
    "vix_z": "VIX korku endeksi (z)", "vix_chg5": "VIX 5 günlük değişim",
    "vix_pct": "VIX yıllık yüzdelik dilimi", "oil_r5": "petrol ivmesi",
    "eur_r5": "EUR/USD ivmesi", "btc_r21": "BTC aylık momentumu",
    "dow": "haftanın günü", "month": "ay (mevsimsellik)",
    "volu_z": "hacim anomalisi (z)", "volu_trend": "hacim trendi",
    "obv_slope": "birikimli hacim eğimi (OBV)", "pv_diverge": "fiyat-hacim uyumsuzluğu",
    "boll_b": "Bollinger konumu", "dd20": "20g tepeden düşüş",
    "du20": "20g dipten yükseliş", "range_pos60": "60g bandındaki konum",
    "round_dist": "psikolojik seviyeye uzaklık", "squeeze": "oynaklık sıkışması",
    "streak": "ardışık gün serisi", "reversal": "dönüş sinyali",
    "skew21": "getiri çarpıklığı", "kurt63": "kuyruk riski (basıklık)",
    "trendiness": "trend/ortalamaya-dönüş rejimi",
    "corr_spy63": "S&P 500 ile korelasyon", "gold_spy_mom": "altın-borsa makası",
    "risk_off": "riskten kaçış rejimi", "try_r5": "dolar/TL ivmesi",
    "evt_iran": "İran/Orta Doğu haber ısısı", "evt_china": "Çin ambargo haber ısısı",
    "evt_sanction": "yaptırım haber ısısı", "evt_tariff": "tarife haber ısısı",
    "evt_fed": "Fed haber ısısı", "evt_oil": "OPEC/petrol arzı haber ısısı",
    "news_cnt": "haber yoğunluğu (24s)", "news_cnt_z": "haber yoğunluğu anomalisi",
    "news_sent1": "haber duygusu (24s)", "news_sent3": "haber duygusu (3g)",
    "is_crypto": "kripto sınıfı", "is_fx": "döviz sınıfı",
    "is_commodity": "emtia sınıfı", "is_stock": "hisse sınıfı",
}

BASE_FEATS = [
    "r1", "r5", "r10", "r21", "r63", "vol21", "vol_ratio", "rsi14",
    "sma20_gap", "sma50_gap", "sma200_gap", "hi52_dist", "macd_hist",
    "vix_z", "vix_chg5", "oil_r5", "eur_r5", "btc_r21", "dow", "month",
    "is_crypto", "is_fx", "is_commodity", "is_stock",
]
CHART_FEATS = [
    # hacim (yalnizca hisse/kripto; digerlerinde NaN — LightGBM NaN'i dogal isler)
    "volu_z", "volu_trend", "obv_slope", "pv_diverge",
    # destek/direnc & konum
    "boll_b", "dd20", "du20", "range_pos60", "round_dist",
    # rejim & oruntu
    "squeeze", "streak", "reversal", "skew21", "kurt63", "trendiness",
    # capraz-varlik / piyasa
    "vix_pct", "corr_spy63", "gold_spy_mom", "risk_off", "try_r5",
    # haber/olay isisi (GDELT z-skor) — A/B karar verir
    "evt_iran", "evt_china", "evt_sanction", "evt_tariff", "evt_fed", "evt_oil",
    # sembol bazli haber akisi (Alpaca/Benzinga, sozluk duygusu)
    "news_cnt", "news_cnt_z", "news_sent1", "news_sent3",
]
EXT_FEATS = BASE_FEATS + CHART_FEATS


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _asset_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]
    lr = np.log(c.where(c > 0)).diff()
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

    # --- v2: hacim ---
    v = g["volume"] if "volume" in g.columns else pd.Series(np.nan, index=g.index)
    vm21 = v.rolling(21).mean()
    g["volu_z"] = (v - vm21) / v.rolling(21).std()
    g["volu_trend"] = v.rolling(5).mean() / vm21 - 1
    obv = (np.sign(lr).fillna(0) * v.fillna(0)).cumsum()
    g["obv_slope"] = np.where(vm21 > 0, (obv - obv.shift(10)) / (vm21 * 10), np.nan)
    g["pv_diverge"] = np.sign(g["r5"]) * g["volu_z"]

    # --- v2: destek/direnc & konum ---
    std20 = c.rolling(20).std()
    g["boll_b"] = (c - c.rolling(20).mean()) / (2 * std20.replace(0, np.nan))
    g["dd20"] = c / c.rolling(20).max() - 1
    g["du20"] = c / c.rolling(20).min() - 1
    rng_hi, rng_lo = c.rolling(60).max(), c.rolling(60).min()
    g["range_pos60"] = (c - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)
    mag = 10 ** np.floor(np.log10(c.where(c > 0)))
    nearest_round = np.round(c / mag * 2) / 2 * mag  # yarim-onluk psikolojik seviyeler
    g["round_dist"] = (c - nearest_round) / c

    # --- v2: rejim & oruntu ---
    g["squeeze"] = lr.rolling(5).std() / lr.rolling(63).std()
    sgn = np.sign(lr).fillna(0)
    blocks = (sgn != sgn.shift()).cumsum()
    g["streak"] = (sgn * (sgn.groupby(blocks).cumcount() + 1)).clip(-10, 10)
    g["reversal"] = np.where(sgn * sgn.shift(1) < 0,
                             np.sign(lr) * (lr.abs() + lr.shift(1).abs()), 0.0)
    g["skew21"] = lr.rolling(21).skew()
    g["kurt63"] = lr.rolling(63).kurt()
    g["trendiness"] = lr.rolling(21).corr(lr.shift(1))  # +: trend, -: ortalamaya donus

    # hedef: t -> t+H log getiri yönü (SADECE gelecek, sızıntı yok)
    g["fwd_ret"] = np.log(c.shift(-HORIZON) / c)
    g["y"] = (g["fwd_ret"] > 0).astype(int)
    return g


def build_features(panel: pd.DataFrame, vix: pd.Series, events: pd.DataFrame | None = None,
                   news: pd.DataFrame | None = None):
    df = pd.concat([_asset_feats(g) for _, g in panel.groupby("asset")], ignore_index=True)

    # --- piyasa genel özellikleri (her tarih için, sadece geçmiş) ---
    vix_df = pd.DataFrame({"vix": vix})
    vix_df["vix_z"] = (vix_df["vix"] - vix_df["vix"].rolling(252).mean()) / vix_df["vix"].rolling(252).std()
    vix_df["vix_chg5"] = vix_df["vix"].pct_change(5)
    vix_df["vix_pct"] = vix_df["vix"].rolling(252).rank(pct=True)
    df = df.merge(vix_df[["vix_z", "vix_chg5", "vix_pct"]],
                  left_on="date", right_index=True, how="left")
    EVT = ["evt_iran", "evt_china", "evt_sanction", "evt_tariff", "evt_fed", "evt_oil"]
    if events is not None and len(events):
        ev = events.reindex(columns=EVT)
        df = df.merge(ev, left_on="date", right_index=True, how="left")
    else:
        for c in EVT:
            df[c] = np.nan

    NEWSF = ["news_cnt", "news_cnt_z", "news_sent1", "news_sent3"]
    if news is not None and len(news):
        df = df.merge(news[["date", "asset"] + NEWSF], on=["date", "asset"], how="left")
    else:
        for c in NEWSF:
            df[c] = np.nan

    def series_of(asset, col):
        s = df[df["asset"] == asset].set_index("date")[col]
        return s[~s.index.duplicated()]

    for feat_asset, src_col, colname in [("WTI", "r5", "oil_r5"), ("EURUSD", "r5", "eur_r5"),
                                         ("BTC", "r21", "btc_r21")]:
        df = df.merge(series_of(feat_asset, src_col).rename(colname),
                      left_on="date", right_index=True, how="left")
    if "USDTRY" in set(df["asset"].unique()):
        df = df.merge(series_of("USDTRY", "r5").rename("try_r5"),
                      left_on="date", right_index=True, how="left")
    else:
        df["try_r5"] = np.nan

    # --- v2: capraz-varlik ---
    have = set(df["asset"].unique())
    spy_r1 = series_of("SPY", "r1").rename("spy_r1") if "SPY" in have else None
    if spy_r1 is not None:
        df = df.merge(spy_r1, left_on="date", right_index=True, how="left")
        df = df.sort_values(["asset", "date"])
        df["corr_spy63"] = (df.groupby("asset", group_keys=False)
                              .apply(lambda x: x["r1"].rolling(63).corr(x["spy_r1"]),
                                     include_groups=False).reset_index(level=0, drop=True))
        spy_r21 = series_of("SPY", "r21")
        gold_r21 = series_of("XAUUSD", "r21") if "XAUUSD" in have else None
        gs = (gold_r21 - spy_r21).rename("gold_spy_mom") if gold_r21 is not None else None
        if gs is not None:
            df = df.merge(gs, left_on="date", right_index=True, how="left")
        else:
            df["gold_spy_mom"] = np.nan
        spy_r21_al = spy_r21.reindex(df["date"]).to_numpy()
        df["risk_off"] = ((df["vix_z"] > 1) & (spy_r21_al < 0)).astype(int)
        df = df.drop(columns=["spy_r1"])
    else:
        df["corr_spy63"] = np.nan
        df["gold_spy_mom"] = np.nan
        df["risk_off"] = ((df["vix_z"] > 1)).astype(int)

    # piyasa özellikleri hafta sonu/tatil boşluklarında son değerle doldurulur (geçmişe bakmaz)
    df = df.sort_values(["asset", "date"])
    for c in ("vix_z", "vix_chg5", "vix_pct", "oil_r5", "eur_r5", "btc_r21",
              "corr_spy63", "gold_spy_mom", "try_r5",
              "evt_iran", "evt_china", "evt_sanction", "evt_tariff", "evt_fed", "evt_oil"):
        df[c] = df.groupby("asset")[c].ffill()

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    for cls in ("crypto", "fx", "commodity", "stock"):
        df[f"is_{cls}"] = (df["aclass"] == cls).astype(int)

    core = ["r1", "r21", "vol21", "rsi14", "sma200_gap", "hi52_dist"]
    df = df.dropna(subset=core)
    return df.reset_index(drop=True), list(EXT_FEATS)


if __name__ == "__main__":
    from fetch import load_panel
    panel, vix = load_panel()
    df, cols = build_features(panel, vix)
    print(df.shape, "ozellik:", len(cols))
    na = df[CHART_FEATS].isna().mean().round(2)
    print("v2 ozellik NaN oranlari:\n", na.to_string())
