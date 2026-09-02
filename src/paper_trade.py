# -*- coding: utf-8 -*-
"""GraphFinance — Alpaca PAPER trading (sahte para, gerçek borsa mekaniği).

Kurallar (sinyal portföyü simülasyonuyla aynı):
  - Sadece kanıtlanmış kenarlı varlıklarda, güven > %55 YUKARI çağrıları
  - Sadece ABD'de işlem gören semboller (Alpaca evreni)
  - Pozisyon: hesap değerinin ~%18'i, en fazla 5 eşzamanlı pozisyon
  - Her koşuda pozisyon gözden geçirme: 5 gün doldu / zarar durdur %4 / sinyal tersine döndü (<%45) /
    kötü haber / kâr al (%6 + zayıf güven) → sat; aksi halde tut (gerekçe karar günlüğüne yazılır)
  - KILL-SWITCH: hesap başlangıcın %90'ının altına inerse her şeyi sat, dur, panoda kırmızı bant

GÜVENLİK: Bu modül SADECE paper-api.alpaca.markets ile konuşur (sahte para).
Canlı URL kodda yoktur. Anahtarlar GitHub secrets'tan gelir; yoksa modül sessizce atlanır.
"""
import json
import os

import numpy as np
import pandas as pd
import requests

BASE = "https://paper-api.alpaca.markets"  # ASLA canli URL kullanma
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE_PATH = os.path.join(ROOT, "data_store", "paper_state.json")
OUT_PATH = os.path.join(ROOT, "reports", "paper.json")

SLICE = 0.18          # (eski sabit dilim — artik referans; asil boyut asagida)
MAX_POS = 5
RISK_PER_POS = 0.012  # hedef: pozisyon basina gunluk ~%1,2 portfoy oynakligi
SLICE_MIN, SLICE_MAX = 0.06, 0.20  # 5 x %20 = %100, marj yok
CLUSTERS = {  # korelasyon tavani: kume basina en fazla 2 pozisyon
    "yari_iletken": {"NVDA", "AVGO", "VRT"},
    "mega_tek": {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"},
    "endeks": {"SPY", "QQQ"},
    "ulke": {"MCHI", "EWJ", "EWG", "EWU", "EWQ", "EWY", "INDA", "EWZ"},
    "sebeke": {"GEV", "ETN", "HUBB", "PWR", "GRID"},
    "gelir": {"SCHD", "JEPI", "O"},
    "sektor": {"XLK", "XLE", "XLF", "XLV", "XLU", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLC"},
    "emtia": {"GLD", "SLV", "URA", "COPX", "LIT", "DBA"},
}
MAX_PER_CLUSTER = 2


def position_fraction(p_up: float, vol21: float) -> float:
    """Oynaklik hedefli boyut x guven carpani. vol21: gunluk log-getiri std."""
    v = float(vol21) if vol21 and vol21 > 0 else 0.02
    base = RISK_PER_POS / v                      # sakin varlik -> buyuk, cilgin -> kucuk
    conf_mult = 0.6 + 2.0 * (float(p_up) - 0.55)  # p.55->0.6x, .65->0.8x, .75->1.0x, .85->1.2x
    return float(min(SLICE_MAX, max(SLICE_MIN, base * conf_mult)))


def _cluster_of(sym: str):
    return next((k for k, v in CLUSTERS.items() if sym in v), None)
HOLD_TDAYS = 5
KILL_DD = 0.10        # baslangictan %10 dusus -> tam durdurma
# Pozisyon gozden gecirme (her kosuda; ozellikle acilis 16:50 TR ve kapanis 23:23 TR):
STOP_LOSS = -0.04     # pozisyon %4 zarardaysa sat (zarar durdur)
TAKE_PROFIT = 0.06    # %6 kardaysa ve guven artik zayifsa sat (kar al)
REVERSAL_P = 0.45     # modelin guncel yukari olasiligi bunun altina dustuyse sat (sinyal tersine dondu)

# Alpaca'da islem gorebilen varliklarimiz (ABD borsalari)
TRADEABLE = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "SPY", "QQQ", "SCHD", "JEPI", "O",
    "MCHI", "EWJ", "EWG", "EWU", "EWQ", "EWY", "INDA", "EWZ",
    "GEV", "ETN", "VRT", "HUBB", "PWR", "GRID",
    "XLK", "XLE", "XLF", "XLV", "XLU", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLC",
    "GLD", "SLV", "URA", "COPX", "LIT", "DBA",
}


def _hdr():
    return {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY_ID", "").strip(),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", "").strip()}


def _api(method, path, **kw):
    r = requests.request(method, BASE + path, headers=_hdr(), timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.text else {}


def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {"start_equity": None, "halted": False, "positions": {}}


def _save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    json.dump(st, open(STATE_PATH, "w"))


def _tdays_since(datestr: str) -> int:
    try:
        return int(np.busday_count(np.datetime64(datestr), np.datetime64("today")))
    except Exception:
        return 0


def run_paper(preds: pd.DataFrame, met: pd.DataFrame, log=print, skip=None, bad_news=None) -> dict | None:
    skip = skip or set()
    bad_news = bad_news or set()
    decisions = []  # karar gunlugu: pano "neden" sutunu
    if not _hdr()["APCA-API-KEY-ID"]:
        log("   paper: anahtar yok, atlandi")
        return None
    st = _load_state()
    try:
        acct = _api("GET", "/v2/account")
        equity = float(acct["equity"])
        if st["start_equity"] is None:
            st["start_equity"] = equity
        positions = {p["symbol"]: p for p in _api("GET", "/v2/positions")}
        orders_yapilan = []

        # --- KILL-SWITCH ---
        if st.get("halted"):
            log("   paper: KILL-SWITCH aktif, islem yok")
        elif equity < st["start_equity"] * (1 - KILL_DD):
            log(f"   paper: KILL-SWITCH TETIKLENDI ({equity:.0f} < %90) — her sey satiliyor")
            for sym, p in positions.items():
                try:
                    _api("POST", "/v2/orders", json={"symbol": sym, "qty": p["qty"],
                         "side": "sell", "type": "market", "time_in_force": "day"})
                    orders_yapilan.append(f"SAT {sym} (kill-switch)")
                except Exception as e:
                    log(f"   paper satis hatasi {sym}: {type(e).__name__}")
            st["halted"] = True
        else:
            # --- 1) suresi dolan pozisyonlari sat ---
            for sym in list(st["positions"]):  # dolmamis/iptal olmus emir kalintilarini temizle (3 is gunu)
                if sym not in positions and _tdays_since(st["positions"][sym].get("entry", "")) >= 3:
                    st["positions"].pop(sym, None)
                    log(f"   paper: {sym} emri dolmamis, kayit temizlendi")
            # --- 1b) POZISYON GOZDEN GECIRME: her acik pozisyon icin sat/tut karari + gerekce ---
            p_now = preds.set_index("asset")["p_up"].to_dict()
            for sym in list(positions):
                entry = st["positions"].get(sym, {}).get("entry")
                days = _tdays_since(entry) if entry else 0
                plpc = float(positions[sym].get("unrealized_plpc") or 0)
                p = p_now.get(sym)
                p_txt = f"guven %{100*p:.0f}" if p is not None else "guven ?"
                reason = None
                if entry and days >= HOLD_TDAYS:
                    reason = f"{HOLD_TDAYS} gun doldu"
                elif plpc <= STOP_LOSS:
                    reason = f"zarar durdur ({100*plpc:+.1f}%)"
                elif p is not None and p < REVERSAL_P:
                    reason = f"sinyal tersine dondu ({p_txt})"
                elif sym in skip and sym in (bad_news or set()):
                    reason = "kotu haber akisi"
                elif plpc >= TAKE_PROFIT and (p is None or p < 0.55):
                    reason = f"kar al ({100*plpc:+.1f}%, {p_txt})"
                if reason:
                    try:
                        _api("POST", "/v2/orders", json={"symbol": sym, "qty": positions[sym]["qty"],
                             "side": "sell", "type": "market", "time_in_force": "day"})
                        orders_yapilan.append(f"SAT {sym} ({reason})")
                        decisions.append(dict(sym=sym, action="SAT", why=reason, plpc=plpc, days=days, p=p))
                        st["positions"].pop(sym, None)
                    except Exception as e:
                        log(f"   paper satis hatasi {sym}: {type(e).__name__}")
                else:
                    why = f"{p_txt}, {100*plpc:+.1f}%, {days}/{HOLD_TDAYS} gun"
                    decisions.append(dict(sym=sym, action="TUT", why=why, plpc=plpc, days=days, p=p))

            # --- 2) yeni sinyalleri al ---
            met_idx = met.set_index("asset")
            # bekleyen (henuz dolmamis) emirler de slot sayar: kapali piyasada verilen gunluk emirler acilista dolar
            # acik (dolmamis) emirler: slot sayar, nakitten duser; fazlasi iptal (marja dusmemek icin)
            try:
                open_orders = [o for o in _api("GET", "/v2/orders", params={"status": "open", "limit": 100})
                               if o.get("side") == "buy"]
            except Exception:
                open_orders = []
            open_orders.sort(key=lambda o: o.get("submitted_at", ""))  # eski once
            cash_left = float(acct.get("cash") or 0)
            keep = []
            for o in open_orders:
                notional = float(o.get("qty") or 0) * float(o.get("limit_price") or preds.set_index("asset")["close"].get(o["symbol"], 0) or 0)
                if len(positions) + len(keep) < MAX_POS and notional <= cash_left:
                    keep.append(o["symbol"])
                    cash_left -= notional
                else:
                    try:
                        _api("DELETE", f"/v2/orders/{o['id']}")
                        st["positions"].pop(o["symbol"], None)
                        orders_yapilan.append(f"IPTAL {o['symbol']} (slot/nakit asimi)")
                    except Exception as e:
                        log(f"   paper iptal hatasi {o['symbol']}: {type(e).__name__}")
            held = set(positions) | set(keep)
            slots = MAX_POS - len(held)
            for _, r in preds.sort_values("p_up", ascending=False).iterrows():
                if slots <= 0:
                    break
                a = r["asset"]
                if (a not in TRADEABLE or a in positions or a in st["positions"]
                        or a in skip  # canli bekci: aski/soguma listesi
                        or r["p_up"] <= 0.55 or a not in met_idx.index
                        or met_idx.loc[a, "auc"] < 0.53):
                    continue
                # kume tavani (ayni temaya yigilma)
                cl = _cluster_of(a)
                if cl and sum(1 for s in list(positions) + list(st["positions"]) if _cluster_of(s) == cl) >= MAX_PER_CLUSTER:
                    continue
                frac = position_fraction(r["p_up"], r.get("vol21", 0.02))
                budget = min(equity * frac, cash_left * 0.98)
                qty = int(budget // float(r["close"]))
                if qty < 1:
                    continue
                try:
                    _api("POST", "/v2/orders", json={"symbol": a, "qty": str(qty),
                         "side": "buy", "type": "market", "time_in_force": "day"})
                    st["positions"][a] = {"entry": str(pd.Timestamp.today().date())}
                    orders_yapilan.append(f"AL {qty}x {a} (guven %{100*r['p_up']:.0f}, boyut %{100*frac:.0f})")
                    decisions.append(dict(sym=a, action="AL", why=f"guven %{100*r['p_up']:.0f}, boyut %{100*frac:.0f}, {qty} adet",
                                          plpc=0.0, days=0, p=float(r["p_up"])))
                    slots -= 1
                    cash_left -= qty * float(r["close"])
                except Exception as e:
                    log(f"   paper alis hatasi {a}: {type(e).__name__}")

        _save_state(st)
        out = {
            "equity": equity, "cash": float(acct["cash"]),
            "start_equity": st["start_equity"], "halted": st["halted"],
            "pnl": equity - st["start_equity"],
            "pnl_pct": (equity / st["start_equity"] - 1) if st["start_equity"] else 0.0,
            "positions": [{"symbol": s, "qty": p["qty"],
                           "value": float(p["market_value"]),
                           "pnl_pct": float(p.get("unrealized_plpc") or 0)}
                          for s, p in positions.items()],
            "orders": orders_yapilan,
            "decisions": decisions,
            "asof": str(pd.Timestamp.today().date()),
            "asof_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False)
        log(f"   paper: hesap ${equity:,.0f} | {len(positions)} pozisyon | {len(orders_yapilan)} emir")
        return out
    except Exception as e:
        log(f"   ! paper hata: {type(e).__name__} {str(e)[:120]}")
        return None
