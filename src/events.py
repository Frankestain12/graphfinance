# -*- coding: utf-8 -*-
"""GraphFinance — olay radarı ve olay kitabı.

1) OLAY RADARI (GDELT DOC 2.0, ücretsiz, anahtarsız): tema başına günlük haber-yoğunluğu
   serisi (2017→bugün), z-skor "ısı" olarak modele özellik girer (A/B karar verir).
   Önbellek: data_store/events.csv — accession değişmediyse sadece son 10 gün yenilenir.
2) OLAY KİTABI: geçmiş gerçek olaylarda varlıkların 5 günlük tepkisi, KENDİ panel verimizden.
Sandbox'ta GDELT kapalı: radar boş/NaN döner, kitap yerel veriyle çalışır.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

H = {"User-Agent": "GraphFinance personal research bedirhan.icli@icloud.com"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data_store", "events.csv")
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"

THEMES = {  # ozellik adi -> (GDELT sorgusu, Turkce ad)
    "evt_iran":     ('(iran OR tehran OR hormuz) (strike OR attack OR war OR missile OR escalation)', "İran / Orta Doğu gerilimi"),
    "evt_china":    ('(china OR beijing) ("export ban" OR embargo OR "export controls" OR "rare earth")', "Çin ambargo / ihracat kısıtı"),
    "evt_sanction": ('sanctions (washington OR "united states" OR treasury OR "european union")', "ABD/AB yaptırımları"),
    "evt_tariff":   ('(tariff OR tariffs) (trump OR "white house" OR trade)', "Tarife / ticaret savaşı"),
    "evt_fed":      ('("federal reserve" OR powell) (rate OR rates OR inflation)', "Fed / faiz"),
    "evt_oil":      ('(opec OR "oil supply" OR "oil output" OR "oil production cut")', "OPEC / petrol arzı"),
}
EVENT_FEATS = list(THEMES)

# Olay kitabi: (tema, tarih, etiket, tur)  tur: 'tirmanma' | 'yatisma'
PLAYBOOK_EVENTS = [
    ("İran / Orta Doğu", "2019-09-14", "Abqaiq saldırısı (Suudi petrol tesisi)", "tirmanma"),
    ("İran / Orta Doğu", "2020-01-03", "Süleymani suikastı", "tirmanma"),
    ("İran / Orta Doğu", "2020-01-08", "Trump 'gerilim düşürme' konuşması", "yatisma"),
    ("İran / Orta Doğu", "2023-10-07", "7 Ekim saldırısı", "tirmanma"),
    ("İran / Orta Doğu", "2024-04-13", "İran'ın İsrail'e İHA/füze saldırısı", "tirmanma"),
    ("İran / Orta Doğu", "2024-10-01", "İran füze yaylımı", "tirmanma"),
    ("İran / Orta Doğu", "2025-06-13", "İsrail-İran savaşı başlangıcı", "tirmanma"),
    ("İran / Orta Doğu", "2025-06-24", "Ateşkes", "yatisma"),
    ("Tarife / Çin", "2018-03-22", "İlk Çin tarifeleri", "tirmanma"),
    ("Tarife / Çin", "2019-05-05", "Tarife artışı tweeti", "tirmanma"),
    ("Tarife / Çin", "2019-08-01", "Yeni %10 tarife duyurusu", "tirmanma"),
    ("Tarife / Çin", "2025-04-02", "'Kurtuluş Günü' tarifeleri", "tirmanma"),
    ("Tarife / Çin", "2025-04-09", "90 günlük tarife duraklaması", "yatisma"),
    ("Tarife / Çin", "2025-05-12", "Cenevre ABD-Çin ateşkesi", "yatisma"),
    ("Yaptırım / Rusya", "2022-02-24", "Rusya'nın Ukrayna işgali", "tirmanma"),
]
PLAYBOOK_ASSETS = ["WTI", "BRENT", "XAUUSD", "SPY", "QQQ", "BTC", "USDTRY", "EURUSD", "USDJPY",
                   "EWZ", "MCHI", "XLE", "GLD", "SLV", "URA", "COPX", "LIT", "XLU"]


def _timeline(query: str, start: str, end: str) -> pd.Series:
    r = requests.get(GDELT, params={"query": query, "mode": "timelinevol", "format": "json",
                                    "startdatetime": start, "enddatetime": end}, headers=H, timeout=60)
    r.raise_for_status()
    data = r.json().get("timeline", [{}])[0].get("data", [])
    s = pd.Series({pd.to_datetime(d["date"][:8]): float(d["value"]) for d in data})
    return s.groupby(s.index).mean().sort_index()


def load_event_features(log=print) -> pd.DataFrame:
    """Tarih indeksli DataFrame: EVENT_FEATS sütunları (z-skor, 90g). Boş olabilir."""
    cache = None
    if os.path.exists(CACHE):
        try:
            cache = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        except Exception:
            cache = None
    today = pd.Timestamp.today().normalize()
    end = today.strftime("%Y%m%d%H%M%S")
    raw = {}
    for feat, (q, _tr) in THEMES.items():
        try:
            if cache is not None and feat in cache.columns and cache[feat].notna().sum() > 500:
                start = (today - pd.Timedelta(days=12)).strftime("%Y%m%d%H%M%S")
                s_new = _timeline(q, start, end)
                s = pd.concat([cache[feat].dropna(), s_new])
                s = s[~s.index.duplicated(keep="last")].sort_index()
            else:
                s = _timeline(q, "20170101000000", end)  # tam geriye doldurma
            raw[feat] = s
            time.sleep(1.2)  # GDELT nezaket
        except Exception as e:
            if cache is not None and feat in cache.columns:
                raw[feat] = cache[feat].dropna()
                log(f"  olay {feat}: guncellenemedi ({type(e).__name__}), onbellek")
            else:
                log(f"  ! olay {feat}: {type(e).__name__}")
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw).sort_index()
    df = df.asfreq("D").ffill(limit=3)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE)
    # z-skor: log-yogunluk, 90 gunluk pencere
    lg = np.log(df + 1e-4)
    z = (lg - lg.rolling(90, min_periods=30).mean()) / lg.rolling(90, min_periods=30).std()
    log(f"   olay radari: {len(df.columns)} tema, {len(df)} gun")
    return z.clip(-4, 4)


def playbook(panel: pd.DataFrame, horizon: int = 5) -> list:
    """Her tema x tur icin varlik bazinda ortalama ileri getiri (kendi verimizden)."""
    px = {a: g.set_index("date")["close"].sort_index()
          for a, g in panel[panel["asset"].isin(PLAYBOOK_ASSETS)].groupby("asset")}
    rows = {}
    for theme, d, label, kind in PLAYBOOK_EVENTS:
        d = pd.Timestamp(d)
        key = (theme, kind)
        for a, s in px.items():
            after = s[s.index >= d]
            if len(after) <= horizon:
                continue
            r = float(after.iloc[horizon] / after.iloc[0] - 1)
            rows.setdefault(key, {}).setdefault(a, []).append(r)
    out = []
    for (theme, kind), assets in rows.items():
        stats = {a: (float(np.mean(v)), len(v)) for a, v in assets.items() if len(v) >= 2}
        if stats:
            out.append(dict(theme=theme, kind=kind, stats=stats))
    return out


def current_heat(z: pd.DataFrame) -> list:
    if z is None or z.empty:
        return []
    last = z.dropna(how="all").iloc[-1]
    return sorted([dict(feat=f, name=THEMES[f][1], z=float(last[f]))
                   for f in EVENT_FEATS if f in last.index and pd.notna(last[f])],
                  key=lambda x: -x["z"])
