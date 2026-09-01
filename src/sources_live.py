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
    # pasif gelir enstrümanları
    ("schd.us", "SCHD", "Temettü ETF (SCHD)"),
    ("jepi.us", "JEPI", "Aylık Gelir ETF (JEPI)"),
    ("o.us", "O", "Realty Income (aylık temettü)"),
]
AI_INFRA = [  # (yahoo_sembol, varlik, ad) — AI altyapi/sebeke sepeti (26 Agu 2026 tezi)
    ("GEV", "GEV", "GE Vernova"), ("ETN", "ETN", "Eaton"),
    ("VRT", "VRT", "Vertiv"), ("HUBB", "HUBB", "Hubbell"),
    ("PWR", "PWR", "Quanta Services"), ("GRID", "GRID", "Şebeke ETF (GRID)"),
]
SECTOR_ETFS = [  # ABD sektorleri (Alpaca'da islem gorur)
    ("XLK", "XLK", "Teknoloji ETF (XLK)"), ("XLE", "XLE", "Enerji ETF (XLE)"),
    ("XLF", "XLF", "Finans ETF (XLF)"), ("XLV", "XLV", "Sağlık ETF (XLV)"),
    ("XLU", "XLU", "Altyapı/Utilities ETF (XLU)"), ("XLI", "XLI", "Sanayi ETF (XLI)"),
    ("XLY", "XLY", "Tüketim ETF (XLY)"), ("XLP", "XLP", "Temel Tüketim ETF (XLP)"),
    ("XLB", "XLB", "Hammadde ETF (XLB)"), ("XLRE", "XLRE", "Gayrimenkul ETF (XLRE)"),
    ("XLC", "XLC", "İletişim ETF (XLC)"),
]
COMMODITY_ETFS = [  # olay kitabinin islem yapilabilir karsiliklari
    ("GLD", "GLD", "Altın ETF (GLD)"), ("SLV", "SLV", "Gümüş ETF (SLV)"),
    ("URA", "URA", "Uranyum ETF (URA)"), ("COPX", "COPX", "Bakır ETF (COPX)"),
    ("LIT", "LIT", "Lityum ETF (LIT)"), ("DBA", "DBA", "Tarım ETF (DBA)"),
]
GLOBAL_MARKETS = [  # (yahoo_sembol, varlik, ad) — ülke ETF'leri (USD, işlem yapılabilir) + dev endeksler
    ("MCHI", "MCHI", "Çin ETF (MCHI)"), ("EWJ", "EWJ", "Japonya ETF (EWJ)"),
    ("EWG", "EWG", "Almanya ETF (EWG)"), ("EWU", "EWU", "İngiltere ETF (EWU)"),
    ("EWQ", "EWQ", "Fransa ETF (EWQ)"), ("EWY", "EWY", "G. Kore ETF (EWY)"),
    ("INDA", "INDA", "Hindistan ETF (INDA)"), ("EWZ", "EWZ", "Brezilya ETF (EWZ)"),
    ("^N225", "N225", "Nikkei 225"), ("^GDAXI", "DAX", "DAX 40"),
    ("^HSI", "HSI", "Hang Seng"), ("^FTSE", "FTSE", "FTSE 100"),
]
BIST = [  # (yahoo_sembol, varlik, ad) — TRY cinsinden
    ("XU100.IS", "XU100", "BIST 100"), ("THYAO.IS", "THYAO", "Türk Hava Yolları"),
    ("GARAN.IS", "GARAN", "Garanti BBVA"), ("ASELS.IS", "ASELS", "Aselsan"),
    ("AKBNK.IS", "AKBNK", "Akbank"), ("EREGL.IS", "EREGL", "Ereğli Demir Çelik"),
    ("TUPRS.IS", "TUPRS", "Tüpraş"), ("BIMAS.IS", "BIMAS", "BİM"),
    ("SISE.IS", "SISE", "Şişecam"), ("KCHOL.IS", "KCHOL", "Koç Holding"),
]
FX_EXTRA = [("xauusd", "XAUUSD", "Altın (ons)"), ("usdtry", "USDTRY", "USD/TRY")]
YAHOO_FX_FALLBACK = {"XAUUSD": "GC=F", "USDTRY": "TRY=X"}  # altın vadeli, USD/TRY
# (binance_sym, asset, coinbase_product) — Binance/Bybit ABD sunucularından engelli
CRYPTO_LIVE = [("BTCUSDT", "BTC", "BTC-USD"), ("ETHUSDT", "ETH", "ETH-USD"),
               ("SOLUSDT", "SOL", "SOL-USD"), ("XRPUSDT", "XRP", "XRP-USD"),
               ("BNBUSDT", "BNB", None)]


def stooq_daily(sym: str) -> pd.DataFrame:
    """Stooq tam geçmiş CSV. Kişisel kullanım için düşük hacimli indirme."""
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        raise ValueError(f"stooq bos: {sym}")
    df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    keep = ["date", "close"] + (["volume"] if "volume" in df.columns else [])
    return df[keep]


def yahoo_daily(sym: str) -> pd.DataFrame:
    """Yahoo chart API yedeği (10 yıl, günlük)."""
    from urllib.parse import quote
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}"
           f"?range=10y&interval=1d&events=div%2Csplit")
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    ts = j["timestamp"]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").date,
                       "close": q["close"], "volume": q.get("volume")})
    return df.dropna(subset=["close"])


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
            "close": [float(x[4]) for x in rows],
            "volume": [float(x[5]) for x in rows]}))
        end = rows[0][0] - 1
        if len(rows) < 1000:
            break
        time.sleep(0.3)
    if not frames:
        raise ValueError(f"binance bos: {sym}")
    return pd.concat(frames).drop_duplicates("date").sort_values("date")


def coinbase_daily(product: str, days: int = 3000) -> pd.DataFrame:
    """Coinbase Exchange public API (ABD dostu, anahtarsız). 300 mum/istek, sayfalı."""
    import datetime as dt
    end = dt.datetime.now(dt.timezone.utc)
    frames = []
    for _ in range(days // 300 + 1):
        start = end - dt.timedelta(days=300)
        url = (f"https://api.exchange.coinbase.com/products/{product}/candles"
               f"?granularity=86400&start={start.isoformat()}&end={end.isoformat()}")
        r = requests.get(url, headers=H, timeout=45)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame({
            "date": pd.to_datetime([x[0] for x in rows], unit="s").date,
            "close": [float(x[4]) for x in rows],
            "volume": [float(x[5]) for x in rows]}))
        end = start
        time.sleep(0.35)
    if not frames:
        raise ValueError(f"coinbase bos: {product}")
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
        "close": [float(x[4]) for x in rows],
        "volume": [float(x[5]) for x in rows]}).sort_values("date")


def _norm(df: pd.DataFrame, asset: str, aclass: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = float("nan")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["asset"], out["aclass"] = asset, aclass
    return out.dropna(subset=["close"])[["date", "asset", "aclass", "close", "volume"]]


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
    for ysym, asset, _name in BIST + GLOBAL_MARKETS + AI_INFRA + SECTOR_ETFS + COMMODITY_ETFS:
        try:
            frames.append(_norm(yahoo_daily(ysym), asset, "stock"))
        except Exception as e:
            log(f"  ! {asset}: veri yok ({type(e).__name__})")
    for sym, asset, _name in FX_EXTRA:
        aclass = "commodity" if asset == "XAUUSD" else "fx"
        try:
            frames.append(_norm(stooq_daily(sym), asset, aclass))
        except Exception:
            try:
                frames.append(_norm(yahoo_daily(YAHOO_FX_FALLBACK[asset]), asset, aclass))
                log(f"  {asset}: stooq yok, yahoo kullanildi")
            except Exception as e2:
                log(f"  ! {asset}: veri yok ({type(e2).__name__})")
    for sym, asset, cb in CRYPTO_LIVE:
        got = False
        for fn, src in ((lambda: binance_daily(sym), "binance"),
                        (lambda: bybit_daily(sym), "bybit"),
                        ((lambda: coinbase_daily(cb)) if cb else None, "coinbase")):
            if fn is None:
                continue
            try:
                frames.append(_norm(fn(), asset, "crypto"))
                if src != "binance":
                    log(f"  {asset}: {src} kullanildi")
                got = True
                break
            except Exception:
                continue
        if not got:
            log(f"  ! {asset}: canli kripto yok (tum kaynaklar engelli)")
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "aclass", "close"])
    return pd.concat(frames, ignore_index=True)


ASSET_NAMES_LIVE = ({a: n for _s, a, n in STOCKS} | {a: n for _s, a, n in FX_EXTRA}
                    | {a: n for _s, a, n in BIST} | {a: n for _s, a, n in GLOBAL_MARKETS} | {a: n for _s, a, n in AI_INFRA}
                    | {a: n for _s, a, n in SECTOR_ETFS} | {a: n for _s, a, n in COMMODITY_ETFS}
                    | {"BNB": "BNB"})
