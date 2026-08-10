# -*- coding: utf-8 -*-
"""GraphFinance — pasif gelir modülü (temettü radarı).
Yahoo chart API'nin temettü olaylarından son 12 ayın GERÇEKLEŞEN verimini hesaplar.
Sandbox'ta Yahoo kapalı — boş tablo döner, pano kartı gizlenir; Actions'ta dolar.
"""
import os
import time

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (GraphFinance personal research; github actions)"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "reports")

# (yahoo_sembol, varlik) — temettü taranacaklar
DIV_UNIVERSE = [
    ("SCHD", "SCHD"), ("JEPI", "JEPI"), ("O", "O"), ("SPY", "SPY"),
    ("AAPL", "AAPL"), ("MSFT", "MSFT"), ("AVGO", "AVGO"),
    ("THYAO.IS", "THYAO"), ("GARAN.IS", "GARAN"), ("ASELS.IS", "ASELS"),
    ("AKBNK.IS", "AKBNK"), ("EREGL.IS", "EREGL"), ("TUPRS.IS", "TUPRS"),
    ("BIMAS.IS", "BIMAS"), ("SISE.IS", "SISE"), ("KCHOL.IS", "KCHOL"),
]


def yahoo_dividends(sym: str) -> list[tuple[pd.Timestamp, float]]:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range=3y&interval=1mo&events=div")
    r = requests.get(url, headers=H, timeout=45)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    divs = (j.get("events") or {}).get("dividends") or {}
    return sorted((pd.to_datetime(int(v["date"]), unit="s"), float(v["amount"]))
                  for v in divs.values())


def freq_label(n_pay_12m: int) -> str:
    if n_pay_12m >= 10:
        return "Aylık"
    if n_pay_12m >= 3:
        return "3 Aylık"
    if n_pay_12m >= 1:
        return "Yıllık"
    return "—"


def build_income(panel: pd.DataFrame, log=print) -> pd.DataFrame:
    """panel: [date, asset, close] — son fiyatlar verim hesabında kullanılır."""
    last_close = (panel.sort_values("date").groupby("asset")["close"].last())
    now = pd.Timestamp.today()
    rows = []
    for sym, asset in DIV_UNIVERSE:
        if asset not in last_close.index:
            continue
        try:
            divs = yahoo_dividends(sym)
        except Exception as e:
            log(f"  ! temettu {asset}: {type(e).__name__}")
            continue
        last12 = [(d, a) for d, a in divs if d >= now - pd.Timedelta(days=365)]
        ttm = sum(a for _, a in last12)
        if ttm <= 0:
            continue
        px = float(last_close[asset])
        rows.append(dict(
            asset=asset, ttm_div=round(ttm, 4), price=px,
            yield_ttm=round(ttm / px, 5), n_pay=len(last12),
            freq=freq_label(len(last12)),
            last_pay=str(max(d for d, _ in last12).date()) if last12 else "",
        ))
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("yield_ttm", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(REP, "income.csv"), index=False)
    log(f"   temettu radari: {len(df)} varlik")
    return df
