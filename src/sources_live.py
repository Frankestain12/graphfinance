# -*- coding: utf-8 -*-
"""Canlı veri kaynakları — GitHub Actions üzerinde çalışır (tam internet erişimi).
Cowork sandbox'ında bu kaynaklar kapalıdır; run_live otomatik olarak fetch.py
kaynaklarına (GitHub-mirror) düşer. Her kaynak yedeklidir: Stooq -> Yahoo, Binance -> Bybit.
"""
import io
import time
import requests
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (GraphFinance personal research; github actions)"}

# sembol, sınıf, görünen ad
STOCKS = [
    ("aapl.us", "AAPL", "Apple"), ("msft.us", "MSFT", "Microsoft"),
    ("nvda.us", "NVDA", "Nvidia"), ("googl.us", "GOOGL", "Alphabet"),
    ("amzn.us", "AMZN", "Amazon"), ("meta.us", "META", "Meta"),
    ("tsla.us", "TSLA", "Tesla"), ("avgo.us", "AVGO", "Broadcom"),
    ("spy.us", "SPY", "S&P 500 ETF"), ("qqq.us", "QQQ", "Nasdaq 100 ETF"),
]
FX_EXTRA = [("xauusd", "XAUUSD", "Altın (ons)"), ("usdtry", "USDTRY", "USD/TRY")]
CRYPTO_LIVE = [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"),
               ("XRPUSDT", "XRP"), ("BNBUSDT", "BNB")]


def stooq_daily(sym: str) -> pd.DataFrame:
    """Stooq tam geçmiş CSV. Kişisel kullanım için düşük hacimli indirme."""
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        raise ValueError(f"stooq bos: {sym}")
    return df.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]


def yahoo_daily(sym: str) -> pd.DataFrame:
    """Yahoo chart API yedeği (10 yıl, günlük)."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range=10y&interval=1d&events=div%2Csplit")
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    ts = j["timestamp"]
    close = j["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").date, "close": close})
    return df.dropna()


def binance_daily(sym: str, days: int = 3000) -> pd.DataFrame:
    """Binance public klines, sayfalayarak ~days günlük geçmiş."""
    frames, end = [], None
    for _ in range(1 + days // 1000):
        url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=1000"
        if end:
            url += f"&endTime={end}"
        r = requests.get(url, headers=H, timeout=45)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame({
            "date": pd.to_datetime([x[0] for x in rows], unit="ms").date,
            "close": [float(x[4]) for x in rows]}))
        end = rows[0][0] - 1
        if len(rows) < 1000:
            break
        time.sleep(0.3)
    if not frames:
        raise ValueError(f"binance bos: {sym}")
    return pd.concat(frames).drop_duplicates("date").sort_values("date")


def bybit_daily(sym: str) -> pd.DataFrame:
    """Bybit yedeği (spot, 1000 gün)."""
    url = (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={sym}"
           f"&interval=D&limit=1000")
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    rows = r.json()["result"]["list"]
    return pd.DataFrame({
        "date": pd.to_datetime([int(x[0]) for x in rows], unit="ms").date,
        "close": [float(x[4]) for x in rows]}).sort_values("date")


def _norm(df: pd.DataFrame, asset: str, aclass: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["asset"], out["aclass"] = asset, aclass
    return out.dropna(subset=["close"])[["date", "asset", "aclass", "close"]]


def load_live_panel(log=print) -> pd.DataFrame:
    """Hisse + altın + USDTRY + güncel kripto. Erişim yoksa boş döner (sandbox)."""
    frames = []
    for sym, asset, _name in STOCKS:
        try:
            frames.append(_norm(stooq_daily(sym), asset, "stock"))
        except Exception as e1:
            try:
                frames.append(_norm(yahoo_daily(asset), asset, "stock"))
                log(f"  {asset}: stooq yok ({type(e1).__name__}), yahoo kullanildi")
            except Exception as e2:
                log(f"  ! {asset}: veri yok ({type(e2).__name__})")
    for sym, asset, _name in FX_EXTRA:
        aclass = "commodity" if asset == "XAUUSD" else "fx"
        try:
            frames.append(_norm(stooq_daily(sym), asset, aclass))
        except Exception as e:
            log(f"  ! {asset}: veri yok ({type(e).__name__})")
    for sym, asset in CRYPTO_LIVE:
        try:
            frames.append(_norm(binance_daily(sym), asset, "crypto"))
        except Exception:
            try:
                frames.append(_norm(bybit_daily(sym), asset, "crypto"))
                log(f"  {asset}: binance yok, bybit kullanildi")
            except Exception as e2:
                log(f"  ! {asset}: canli kripto yok ({type(e2).__name__})")
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "aclass", "close"])
    return pd.concat(frames, ignore_index=True)


ASSET_NAMES_LIVE = {a: n for _s, a, n in STOCKS} | {a: n for _s, a, n in FX_EXTRA} | {"BNB": "BNB"}
