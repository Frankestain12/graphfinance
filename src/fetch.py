"""GraphFinance — veri katmanı.
Sandbox'tan erişilebilen ücretsiz kaynaklar: raw.githubusercontent.com (datahub + coinmetrics).
Çıktı: tek panel DataFrame [date, asset, close] + VIX ayrı seri (özellik için).
"""
import io
import os
import requests
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (GraphFinance; personal research)"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CM_BASE = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{sym}.csv"
DATAHUB = {
    "VIX":    ("https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv", "DATE", "CLOSE"),
    "WTI":    ("https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv", "Date", "Price"),
    "BRENT":  ("https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv", "Date", "Price"),
    "NATGAS": ("https://raw.githubusercontent.com/datasets/natural-gas/main/data/daily.csv", "Date", "Price"),
}
FX_URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
# Fed H.10 kotasyon yönü korunuyor (parite adına yansıtıldı)
FX_MAP = {
    "Euro": "EURUSD", "United Kingdom": "GBPUSD", "Australia": "AUDUSD",
    "Japan": "USDJPY", "Switzerland": "USDCHF", "China": "USDCNY",
}
CRYPTO = ["btc", "eth", "xrp"]  # sol: coinmetrics community fiyat kolonu boş


def _get(url: str, cache_name: str, refresh: bool = True) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, cache_name)
    if refresh or not os.path.exists(path):
        try:
            r = requests.get(url, headers=H, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            if not os.path.exists(path):
                raise
            print(f"  ! {cache_name}: indirme hatasi ({e}), cache kullaniliyor")
    with open(path, "r") as f:
        return f.read()


def load_crypto() -> pd.DataFrame:
    frames = []
    for sym in CRYPTO:
        txt = _get(CM_BASE.format(sym=sym), f"cm_{sym}.csv")
        df = pd.read_csv(io.StringIO(txt), low_memory=False)
        cands = [c for c in ("PriceUSD", "ReferenceRateUSD", "ReferenceRate") if c in df.columns]
        if not cands:
            continue
        price_col = max(cands, key=lambda c: df[c].notna().sum())
        out = df[["time", price_col]].rename(columns={"time": "date", price_col: "close"})
        out["asset"] = sym.upper()
        out["aclass"] = "crypto"
        frames.append(out.dropna(subset=["close"]))
    return pd.concat(frames, ignore_index=True)


def load_datahub() -> pd.DataFrame:
    frames = []
    for name, (url, dcol, vcol) in DATAHUB.items():
        if name == "VIX":
            continue  # VIX sadece ozellik
        txt = _get(url, f"dh_{name.lower()}.csv")
        df = pd.read_csv(io.StringIO(txt)).rename(columns={dcol: "date", vcol: "close"})
        df["asset"] = name
        df["aclass"] = "commodity"
        frames.append(df[["date", "asset", "aclass", "close"]].dropna())
    return pd.concat(frames, ignore_index=True)


def load_fx() -> pd.DataFrame:
    txt = _get(FX_URL, "fx_daily.csv")
    df = pd.read_csv(io.StringIO(txt))
    df = df[df["Country"].isin(FX_MAP)].copy()
    df["asset"] = df["Country"].map(FX_MAP)
    df = df.rename(columns={"Date": "date", "Exchange rate": "close"})
    df["aclass"] = "fx"
    return df[["date", "asset", "aclass", "close"]].dropna()


def load_vix() -> pd.Series:
    url, dcol, vcol = DATAHUB["VIX"]
    txt = _get(url, "dh_vix.csv")
    df = pd.read_csv(io.StringIO(txt)).rename(columns={dcol: "date", vcol: "vix"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["vix"].astype(float).sort_index()


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    panel = pd.concat([load_crypto(), load_datahub(), load_fx()], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel = (panel.dropna(subset=["close"])
                  .sort_values(["asset", "date"])
                  .drop_duplicates(["asset", "date"], keep="last")
                  .reset_index(drop=True))
    return panel, load_vix()


if __name__ == "__main__":
    panel, vix = load_panel()
    print(panel.groupby("asset").agg(n=("close", "size"),
                                     start=("date", "min"), last=("date", "max")))
    print("VIX last:", vix.index[-1].date(), vix.iloc[-1])
