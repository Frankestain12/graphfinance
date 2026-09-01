# -*- coding: utf-8 -*-
"""GraphFinance — bilanço takvimi (Nasdaq, ücretsiz, anahtarsız).
Önümüzdeki N gün içinde bilanço açıklayacak sembolleri döndürür.
Ders (18 günlük otopsi): en büyük kayıplar bilanço günlerinde — bilançoya girerken pozisyon açma.
Erişim yoksa boş küme döner (filtre devre dışı, sistem çalışmaya devam eder).
"""
import time

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (GraphFinance personal research)",
     "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}


def upcoming_earnings(days: int = 4, log=print) -> set:
    out = set()
    today = pd.Timestamp.today().normalize()
    for i in range(days + 1):
        d = (today + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={d}",
                             headers=H, timeout=30)
            r.raise_for_status()
            rows = (r.json().get("data") or {}).get("rows") or []
            out |= {str(x.get("symbol", "")).upper() for x in rows if x.get("symbol")}
            time.sleep(0.4)
        except Exception as e:
            log(f"  bilanco {d}: {type(e).__name__}")
    return out
