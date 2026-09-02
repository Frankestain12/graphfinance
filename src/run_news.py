# -*- coding: utf-8 -*-
"""GraphFinance — SAATLİK haber koşusu (hafif).

Model eğitimi/tahmin yok. Yapılanlar:
  1) Alpaca haber akışından yeni başlıkları çek (artımlı), duygu skorla, önbelleğe ekle
  2) Kötü-haber filtresini güncelle (data_store/bad_news.json) — günlük koşu da bunu okur
  3) Panoyu son haberlerle yeniden kur (docs/index.html); model çıktıları günlük koşudan gelir
Günlük koşunun kaydettiği reports/dash_state.json yoksa pano adımı atlanır.
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "reports")
STORE = os.path.join(ROOT, "data_store")
DOCS = os.path.join(ROOT, "docs")


def main():
    t0 = time.time()
    print("haber kosusu (saatlik)...")
    from news import update_news, bad_news_assets
    news_daily, news_latest = update_news()
    bad_news = bad_news_assets(news_latest)
    json.dump({"asof": pd.Timestamp.now('UTC').isoformat(), "bad_news": sorted(bad_news),
               "symbols": {a: dict(cnt=v.get("cnt", 0), sent=round(float(v.get("sent", 0)), 3))
                           for a, v in news_latest.items()}},
              open(os.path.join(STORE, "bad_news.json"), "w"), ensure_ascii=False)
    if bad_news:
        print(f"   kotu haber filtresi: {sorted(bad_news)}")

    state_path = os.path.join(REP, "dash_state.json")
    if not os.path.exists(state_path):
        print("   pano durumu yok (gunluk kosu henuz v8.2 degil), pano atlandi")
        return
    st = json.load(open(state_path))
    import build_dashboard as BD
    import ledger as L
    from sources_live import ASSET_NAMES_LIVE
    met, preds, oos, imp = BD.load()
    led = L.load_ledger(os.path.join(STORE, "ledger.csv"))
    inc_path = os.path.join(REP, "income.csv")
    try:
        income_df = pd.read_csv(inc_path) if os.path.getsize(inc_path) > 5 else pd.DataFrame()
    except Exception:
        income_df = pd.DataFrame()
    html = BD.build(met, preds, oos, imp, extra_names=ASSET_NAMES_LIVE, led=led,
                    yorum=st.get("yorum"), income=income_df, usdtry=st.get("usdtry"),
                    whales=st.get("whales") or [], paper=st.get("paper"),
                    suspended=set(st.get("suspended", [])), cooldown=set(st.get("cooldown", [])),
                    heat=st.get("heat") or [], book=st.get("book") or [],
                    earnings_soon=set(st.get("earnings_soon", [])),
                    news_latest=news_latest, bad_news=bad_news)
    os.makedirs(DOCS, exist_ok=True)
    for out in (os.path.join(DOCS, "index.html"), os.path.join(REP, "graphfinance_panosu.html")):
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"   pano tazelendi ({len(news_latest)} sembolde haber)")
    print(f"bitti — {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
