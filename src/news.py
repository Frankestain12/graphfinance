# -*- coding: utf-8 -*-
"""GraphFinance — anlık haber akışı (Alpaca News API, Benzinga kaynaklı, ücretsiz).

- Sembol bazlı haberler; 2 yıl geriye doldurma + günlük artımlı önbellek (data_store/news_daily.csv)
- Duygu: finans sözlüğü (Loughran-McDonald tarzı) — hafif, tutarlı, geriye doldurulabilir.
  (FinBERT gibi transformer modeli v9 adayı; geçmiş biriktikçe A/B'ye girer.)
- Özellikler (varlık-gün): news_cnt (log1p), news_cnt_z (30g), news_sent1, news_sent3
- Kötü-haber filtresi: son 24 saatte >=3 haber ve ortalama duygu < -0.5 → o gün alım yok
Anahtar yoksa / erişim yoksa: önbellek varsa onu kullanır, yoksa boş döner.
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data_store", "news_daily.csv")
LATEST = os.path.join(ROOT, "data_store", "news_latest.json")
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
BACKFILL_START = "2024-08-01"
NEWS_FEATS = ["news_cnt", "news_cnt_z", "news_sent1", "news_sent3"]

# haber takip edilecek semboller (sirket/endeks; ETF haberleri seyrek)
NEWS_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "SPY", "QQQ",
                "GEV", "ETN", "VRT", "HUBB", "PWR", "JEPI", "O", "SCHD"]

POS = set("""beat beats beating surge surges surged record upgrade upgraded upgrades strong stronger growth
profit profits gain gains rally rallies outperform outperforms bullish raise raises raised exceed exceeds
boost boosts jump jumps soar soars buyback breakthrough approval approved wins win expands expansion
accelerate accelerates momentum optimistic upbeat robust resilient tops topped surpass surpasses""".split())
NEG = set("""miss misses missed cut cuts downgrade downgraded downgrades weak weaker loss losses decline declines
drop drops plunge plunges lawsuit probe investigation recall fraud bankruptcy layoffs layoff warning warns
slump slumps tumble tumbles bearish delay delays halt halted fine fines sanction sanctions tariff tariffs
disappoint disappointing disappoints fall falls fell selloff sell-off crash crashes risk risks concern concerns
scrutiny outage breach shortfall guidance-cut""".split())
_TOK = re.compile(r"[a-z][a-z\-]+")


def lexicon_score(text: str) -> float:
    w = _TOK.findall((text or "").lower())
    p = sum(1 for t in w if t in POS)
    n = sum(1 for t in w if t in NEG)
    return (p - n) / (p + n + 1.0)


def _hdr():
    return {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY_ID", "").strip(),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", "").strip()}


def _fetch_symbol(sym: str, start: str, log=print, max_pages: int = 120) -> list:
    out, token, pages = [], None, 0
    while pages < max_pages:
        params = {"symbols": sym, "start": f"{start}T00:00:00Z", "limit": 50,
                  "sort": "asc", "include_content": "false"}
        if token:
            params["page_token"] = token
        r = requests.get(NEWS_URL, headers=_hdr(), params=params, timeout=45)
        r.raise_for_status()
        j = r.json()
        for a in j.get("news", []):
            out.append(dict(asset=sym, ts=a.get("created_at"), headline=a.get("headline", ""),
                            summary=a.get("summary", ""), url=a.get("url", "")))
        token = j.get("next_page_token")
        pages += 1
        if not token:
            break
        time.sleep(0.35)
    return out


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Bos onbellekten concat sonrasi object dtype kalmasin (ilk calismada TypeError)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].astype(str)
    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0).astype(float)
    df["sent"] = pd.to_numeric(df["sent"], errors="coerce").fillna(0.0).astype(float)
    return df


def update_news(log=print) -> tuple[pd.DataFrame, dict]:
    """Önbelleği günceller; (gunluk_ozet_df, son_basliklar) döndürür."""
    cache = pd.read_csv(CACHE, parse_dates=["date"]) if os.path.exists(CACHE) else \
        pd.DataFrame(columns=["date", "asset", "cnt", "sent"])
    latest = {}
    if not _hdr()["APCA-API-KEY-ID"]:
        log("   haber: alpaca anahtari yok, onbellek kullaniliyor")
        return cache, (json.load(open(LATEST)) if os.path.exists(LATEST) else {})
    rows = []
    for sym in NEWS_SYMBOLS:
        have = cache[cache["asset"] == sym]["date"].max() if len(cache) else pd.NaT
        start = (have - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if pd.notna(have) else BACKFILL_START
        try:
            arts = _fetch_symbol(sym, start, log)
        except Exception as e:
            log(f"  haber {sym}: {type(e).__name__}")
            continue
        for a in arts:
            d = pd.to_datetime(a["ts"]).tz_convert(None).normalize()
            rows.append(dict(date=d, asset=sym, sent=lexicon_score(a["headline"] + " " + (a["summary"] or "")),
                             headline=a["headline"], url=a["url"], ts=a["ts"]))
        time.sleep(0.2)
    if rows:
        new = pd.DataFrame(rows)
        agg = new.groupby(["date", "asset"]).agg(cnt=("sent", "size"), sent=("sent", "mean")).reset_index()
        cache = pd.concat([cache, agg], ignore_index=True) if len(cache) else agg
        cache = cache.drop_duplicates(["date", "asset"], keep="last").sort_values(["asset", "date"])
        cache = _typed(cache)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        cache.to_csv(CACHE, index=False)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        rec = new[new["date"] >= cutoff].sort_values("ts", ascending=False)
        for sym, g in rec.groupby("asset"):
            latest[sym] = dict(cnt=int(len(g)), sent=float(g["sent"].mean()),
                               headlines=[dict(h=r["headline"][:140], u=r["url"], s=round(float(r["sent"]), 2))
                                          for _, r in g.head(3).iterrows()])
        json.dump(latest, open(LATEST, "w"), ensure_ascii=False)
    log(f"   haber akisi: {len(rows)} yeni baslik, onbellek {len(cache)} varlik-gun")
    return cache, latest


def news_features(daily: pd.DataFrame) -> pd.DataFrame:
    """(date, asset) indeksli özellikler. Boş DataFrame olabilir."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["date", "asset"] + NEWS_FEATS)
    daily = _typed(daily)
    out = []
    for a, g in daily.groupby("asset"):
        s = g.set_index("date")[["cnt", "sent"]].sort_index()
        idx = pd.date_range(s.index.min(), pd.Timestamp.today().normalize(), freq="D")
        s = s.reindex(idx)
        cnt = s["cnt"].fillna(0)
        sent = s["sent"]
        f = pd.DataFrame(index=idx)
        f["news_cnt"] = np.log1p(cnt)
        f["news_cnt_z"] = (cnt - cnt.rolling(30, min_periods=10).mean()) / (cnt.rolling(30, min_periods=10).std() + 1e-6)
        f["news_sent1"] = sent.fillna(0.0)
        f["news_sent3"] = sent.rolling(3, min_periods=1).mean().fillna(0.0)
        f["asset"] = a
        out.append(f.reset_index().rename(columns={"index": "date"}))
    return pd.concat(out, ignore_index=True)


def bad_news_assets(latest: dict, min_cnt: int = 3, thr: float = -0.5) -> set:
    return {a for a, v in (latest or {}).items() if v.get("cnt", 0) >= min_cnt and v.get("sent", 0) < thr}
