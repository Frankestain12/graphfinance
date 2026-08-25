# -*- coding: utf-8 -*-
"""GraphFinance — Alpaca PAPER trading (sahte para, gerçek borsa mekaniği).

Kurallar (sinyal portföyü simülasyonuyla aynı):
  - Sadece kanıtlanmış kenarlı varlıklarda, güven > %55 YUKARI çağrıları
  - Sadece ABD'de işlem gören semboller (Alpaca evreni)
  - Pozisyon: hesap değerinin ~%18'i, en fazla 5 eşzamanlı pozisyon
  - 5 işlem günü sonra satış
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

SLICE = 0.18          # pozisyon basina hesap orani
MAX_POS = 5
HOLD_TDAYS = 5
KILL_DD = 0.10        # baslangictan %10 dusus -> tam durdurma

# Alpaca'da islem gorebilen varliklarimiz (ABD borsalari)
TRADEABLE = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "SPY", "QQQ", "SCHD", "JEPI", "O",
    "MCHI", "EWJ", "EWG", "EWU", "EWQ", "EWY", "INDA", "EWZ",
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


def run_paper(preds: pd.DataFrame, met: pd.DataFrame, log=print, skip=None) -> dict | None:
    skip = skip or set()
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
            for sym in list(positions):
                entry = st["positions"].get(sym, {}).get("entry")
                if entry and _tdays_since(entry) >= HOLD_TDAYS:
                    try:
                        _api("POST", "/v2/orders", json={"symbol": sym, "qty": positions[sym]["qty"],
                             "side": "sell", "type": "market", "time_in_force": "day"})
                        orders_yapilan.append(f"SAT {sym} ({HOLD_TDAYS} gun doldu)")
                        st["positions"].pop(sym, None)
                    except Exception as e:
                        log(f"   paper satis hatasi {sym}: {type(e).__name__}")

            # --- 2) yeni sinyalleri al ---
            met_idx = met.set_index("asset")
            slots = MAX_POS - len(positions)
            for _, r in preds.sort_values("p_up", ascending=False).iterrows():
                if slots <= 0:
                    break
                a = r["asset"]
                if (a not in TRADEABLE or a in positions or a in st["positions"]
                        or a in skip  # canli bekci: aski/soguma listesi
                        or r["p_up"] <= 0.55 or a not in met_idx.index
                        or met_idx.loc[a, "auc"] < 0.53):
                    continue
                qty = int((equity * SLICE) // float(r["close"]))
                if qty < 1:
                    continue
                try:
                    _api("POST", "/v2/orders", json={"symbol": a, "qty": str(qty),
                         "side": "buy", "type": "market", "time_in_force": "day"})
                    st["positions"][a] = {"entry": str(pd.Timestamp.today().date())}
                    orders_yapilan.append(f"AL {qty}x {a} (guven %{100*r['p_up']:.0f})")
                    slots -= 1
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
            "asof": str(pd.Timestamp.today().date()),
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False)
        log(f"   paper: hesap ${equity:,.0f} | {len(positions)} pozisyon | {len(orders_yapilan)} emir")
        return out
    except Exception as e:
        log(f"   ! paper hata: {type(e).__name__} {str(e)[:120]}")
        return None
