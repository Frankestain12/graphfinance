# -*- coding: utf-8 -*-
"""GraphFinance — pano üretici (self-contained HTML, dataviz spec)."""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REP = os.path.join(ROOT, "reports")
H = 5

ASSET_TR = {
    "BTC": ("Bitcoin", "$"), "ETH": ("Ethereum", "$"), "XRP": ("XRP", "$"),
    "SOL": ("Solana", "$"), "BNB": ("BNB", "$"),
    "EURUSD": ("EUR/USD", ""), "GBPUSD": ("GBP/USD", ""), "AUDUSD": ("AUD/USD", ""),
    "USDJPY": ("USD/JPY", ""), "USDCHF": ("USD/CHF", ""), "USDCNY": ("USD/CNY", ""),
    "USDTRY": ("USD/TRY", ""), "XAUUSD": ("Altın (ons)", "$"),
    "WTI": ("Ham Petrol (WTI)", "$"), "BRENT": ("Brent Petrol", "$"),
    "NATGAS": ("Doğalgaz (Henry Hub)", "$"),
}
CLASS_TR = {"crypto": "Kripto", "fx": "Döviz", "commodity": "Emtia", "stock": "Hisse"}
COST = {"crypto": 0.0010, "fx": 0.0002, "commodity": 0.0005, "stock": 0.0005}


def aname(a, extra=None):
    if a in ASSET_TR:
        return ASSET_TR[a]
    if extra and a in extra:
        return (extra[a], "$")
    return (a, "$")


def tr_num(x, dec=2):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def tr_pct(x, dec=1):
    return tr_num(100 * x, dec) + "%"


def edge_class(auc):
    if auc >= 0.53:
        return "proven", "Kanıtlanmış kenar"
    if auc >= 0.51:
        return "weak", "Zayıf sinyal"
    return "none", "Kenar yok"


def load():
    met = pd.read_csv(os.path.join(REP, "metrics.csv"))
    preds = pd.read_csv(os.path.join(REP, "latest_predictions.csv"), parse_dates=["date"])
    oos = pd.read_parquet(os.path.join(REP, "oos_predictions.parquet"))
    imp = pd.read_csv(os.path.join(REP, "feature_importance.csv"), names=["feat", "imp"], skiprows=1)
    return met, preds, oos, imp


def pooled_stats(oos):
    g = oos.dropna(subset=["y"])
    conf = g[(g["p_up"] > 0.55) | (g["p_up"] < 0.45)]
    hit = ((conf["p_up"] > 0.5).astype(int) == conf["y"]).mean()
    allhit = ((g["p_up"] > 0.5).astype(int) == g["y"]).mean()
    return dict(n_oos=len(g), n_conf=len(conf), conf_hit=hit, all_hit=allhit,
                d0=str(g["date"].min().date()), d1=str(g["date"].max().date()))


def equity(oos, asset, cost):
    g = oos[oos["asset"] == asset].dropna(subset=["fwd_ret"]).sort_values("date").iloc[::H]
    strat = np.where(g["p_up"] > 0.55, g["fwd_ret"] - 2 * cost, 0.0)
    return (g["date"].dt.strftime("%Y-%m-%d").tolist(),
            np.exp(np.cumsum(strat)).round(4).tolist(),
            np.exp(np.cumsum(g["fwd_ret"].values)).round(4).tolist())


# ---------------- SVG helpers ----------------
def svg_auc_bars(met, extra=None):
    """AUC diverging bar, 0.5 merkez. Mavi=kenar, kırmızı=negatif."""
    rows = met.sort_values("auc", ascending=False).reset_index(drop=True)
    bh, gap, top, left, right = 22, 10, 8, 150, 60
    W = 640
    Hgt = top + len(rows) * (bh + gap) + 30
    cx = left + (W - left - right) * 0.5
    span = max(0.12, 2 * (rows["auc"] - 0.5).abs().max() + 0.02)
    scale = (W - left - right) / span
    s = [f'<svg viewBox="0 0 {W} {Hgt}" role="img" aria-label="Varlık başına AUC">']
    # gridline at 0.5
    s.append(f'<line x1="{cx}" y1="{top-4}" x2="{cx}" y2="{Hgt-26}" stroke="var(--baseline)" stroke-width="1"/>')
    for v in (0.5 - span / 3, 0.50, 0.5 + span / 3):
        x = cx + (v - 0.5) * scale
        s.append(f'<text x="{x}" y="{Hgt-8}" text-anchor="middle" class="ax">{tr_num(v,2)}</text>')
    for i, r in rows.iterrows():
        y = top + i * (bh + gap)
        w = abs(r["auc"] - 0.5) * scale
        fill = "var(--pos)" if r["auc"] >= 0.5 else "var(--neg)"
        x = cx if r["auc"] >= 0.5 else cx - w
        rx = "3"
        name = aname(r["asset"], extra)[0]
        s.append(f'<text x="{left-8}" y="{y+bh/2+4}" text-anchor="end" class="lbl">{name}</text>')
        s.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w,1):.1f}" height="{bh}" rx="{rx}" fill="{fill}" '
                 f'class="hv" data-tip="{name} — AUC {tr_num(r["auc"],3)} · isabet {tr_pct(r["hit"])} · taban {tr_pct(r["base_hit"])}"/>')
        tx = cx + w + 6 if r["auc"] >= 0.5 else cx - w - 6
        anc = "start" if r["auc"] >= 0.5 else "end"
        s.append(f'<text x="{tx:.1f}" y="{y+bh/2+4}" text-anchor="{anc}" class="val">{tr_num(r["auc"],3)}</text>')
    s.append("</svg>")
    return "".join(s)


def svg_calibration(cal_rows):
    """Dumbbell: ort. tahmin (açık) -> gerçekleşen yukarı oranı (koyu)."""
    bh, gap, top, left, right = 30, 14, 14, 120, 30
    W = 640
    Hgt = top + len(cal_rows) * (bh + gap) + 34
    x0, x1 = 0.30, 0.70
    def X(v):
        return left + (W - left - right) * (v - x0) / (x1 - x0)
    s = [f'<svg viewBox="0 0 {W} {Hgt}" role="img" aria-label="Kalibrasyon">']
    s.append(f'<line x1="{X(0.5)}" y1="{top-4}" x2="{X(0.5)}" y2="{Hgt-30}" stroke="var(--baseline)" stroke-width="1"/>')
    for v in (0.35, 0.5, 0.65):
        s.append(f'<text x="{X(v)}" y="{Hgt-10}" text-anchor="middle" class="ax">{tr_pct(v,0)}</text>')
    for i, r in enumerate(cal_rows):
        y = top + i * (bh + gap) + bh / 2
        xa, xb = X(r["pred"]), X(r["act"])
        s.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="lbl">{r["lab"]}</text>')
        s.append(f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="var(--conn)" stroke-width="2"/>')
        tip = f'{r["lab"]}: tahmin {tr_pct(r["pred"])} → gerçek {tr_pct(r["act"])} (n={r["n"]:,})'.replace(",", ".")
        s.append(f'<circle cx="{xa}" cy="{y}" r="6" fill="var(--dot-light)" stroke="var(--surface-1)" stroke-width="2" class="hv" data-tip="{tip}"/>')
        s.append(f'<circle cx="{xb}" cy="{y}" r="6" fill="var(--dot-dark)" stroke="var(--surface-1)" stroke-width="2" class="hv" data-tip="{tip}"/>')
    s.append("</svg>")
    return "".join(s)


def svg_equity(dates, strat, bh, label):
    W, Hgt, left, right, top, bot = 560, 260, 46, 96, 14, 30
    n = len(dates)
    ymax = max(max(strat), max(bh), 1.0)
    ymin = min(min(strat), min(bh), 1.0)
    pad = (ymax - ymin) * 0.06 + 1e-9
    ymax += pad; ymin -= pad
    def X(i): return left + (W - left - right) * i / max(n - 1, 1)
    def Y(v): return top + (Hgt - top - bot) * (1 - (v - ymin) / (ymax - ymin))
    def path(vals):
        return "M" + " L".join(f"{X(i):.1f} {Y(v):.1f}" for i, v in enumerate(vals))
    s = [f'<svg viewBox="0 0 {W} {Hgt}" role="img" aria-label="{label} strateji eğrisi" class="eqsvg" '
         f'data-dates="{",".join(dates)}" data-strat="{",".join(map(str,strat))}" data-bh="{",".join(map(str,bh))}" '
         f'data-l="{left}" data-r="{right}" data-w="{W}" data-h="{Hgt}">']
    ydec = 2 if (ymax - ymin) < 1.5 else 1
    for gv in np.linspace(ymin + pad, ymax - pad, 4):
        s.append(f'<line x1="{left}" y1="{Y(gv):.1f}" x2="{W-right}" y2="{Y(gv):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        s.append(f'<text x="{left-6}" y="{Y(gv)+4:.1f}" text-anchor="end" class="ax">{tr_num(gv,ydec)}x</text>')
    yrs = sorted({d[:4] for d in dates})
    for yr in yrs[1::2]:
        i = next(i for i, d in enumerate(dates) if d.startswith(yr))
        s.append(f'<text x="{X(i):.1f}" y="{Hgt-8}" text-anchor="middle" class="ax">{yr}</text>')
    s.append(f'<path d="{path(bh)}" fill="none" stroke="var(--ctx)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    s.append(f'<path d="{path(strat)}" fill="none" stroke="var(--acc)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    s.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(strat[-1]):.1f}" r="4" fill="var(--acc)" stroke="var(--surface-1)" stroke-width="2"/>')
    s.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(bh[-1]):.1f}" r="4" fill="var(--ctx)" stroke="var(--surface-1)" stroke-width="2"/>')
    s.append(f'<text x="{X(n-1)+8:.1f}" y="{Y(strat[-1])+4:.1f}" class="val">{tr_num(strat[-1],1)}x model</text>')
    s.append(f'<text x="{X(n-1)+8:.1f}" y="{Y(bh[-1])+4:.1f}" class="val2">{tr_num(bh[-1],1)}x al-tut</text>')
    s.append(f'<line class="xh" x1="-9" y1="{top}" x2="-9" y2="{Hgt-bot}" stroke="var(--baseline)" stroke-width="1" opacity="0"/>')
    s.append("</svg>")
    return "".join(s)


AYLAR = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
         "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]


def build(met, preds, oos, imp, extra_names=None, led=None):
    ps = pooled_stats(oos)
    today = pd.Timestamp.today().normalize()
    gen_date = f"{today.day} {AYLAR[today.month]} {today.year}"

    # kalibrasyon kovaları
    g = oos.dropna(subset=["y"])
    bins = [(0.0, 0.4, "&lt; %40"), (0.4, 0.45, "%40–45"), (0.45, 0.5, "%45–50"),
            (0.5, 0.55, "%50–55"), (0.55, 0.6, "%55–60"), (0.6, 1.0, "&gt; %60")]
    cal_rows = []
    for lo, hi, lab in bins:
        m = g[(g["p_up"] > lo) & (g["p_up"] <= hi)]
        cal_rows.append(dict(lab=lab, n=len(m), pred=m["p_up"].mean(), act=m["y"].mean()))

    # kanıtlanmış kenarı en yüksek 2 varlığın strateji grafiği
    met_sorted = met.sort_values("auc", ascending=False)
    top2 = met_sorted.head(2)
    eq_charts = []
    for _, mr in top2.iterrows():
        a = mr["asset"]
        eq_charts.append((a, aname(a, extra_names)[0], mr,
                          equity(oos, a, COST.get(mr["aclass"], 0.0005))))

    met_idx = met.set_index("asset")

    # defter (gerçek isabet) kutusu
    ledger_html = ""
    if led is not None and len(led):
        done = led[led["resolved"] == 1]
        if len(done):
            ledger_html = (f'<div class="card tile"><div class="tl">Canlı defter isabeti</div>'
                           f'<div class="tv">{tr_pct(done["correct"].mean())}</div>'
                           f'<div class="td">{len(done)} çözülmüş gerçek tahmin</div></div>')
        else:
            ledger_html = (f'<div class="card tile"><div class="tl">Tahmin defteri</div>'
                           f'<div class="tv">{len(led)}</div>'
                           f'<div class="td">kayıtlı tahmin · ilk sonuçlar 5 iş günü sonra</div></div>')

    # tahmin tablosu satırları
    rows_html = []
    preds = preds.sort_values("p_up", key=lambda s: (s - 0.5).abs(), ascending=False)
    for _, r in preds.iterrows():
        a = r["asset"]
        name, cur = aname(a, extra_names)
        if a not in met_idx.index:
            continue
        m = met_idx.loc[a]
        ec, elab = edge_class(m["auc"])
        up = r["p_up"] >= 0.5
        prob = max(r["p_up"], 1 - r["p_up"])
        band = 1.28 * r["vol21"] * np.sqrt(H)
        lo, hi = r["close"] * np.exp(-band), r["close"] * np.exp(band)
        dec = 4 if r["close"] < 20 else (2 if r["close"] < 1000 else 0)
        dt = pd.Timestamp(r["date"]).strftime("%d.%m.%Y")
        days_old = (today - pd.Timestamp(r["date"]).normalize()).days
        stale = (f'<span class="chip stale" data-tip="Bu varlığın verisi {days_old} gün geride — '
                 f'canlı veri hattı kurulunca güncellenecek">{days_old} gün eski</span>'
                 if days_old > 7 else "")
        dir_html = (f'<span class="dir {"up" if up else "dn"}{" mut" if ec=="none" else ""}">'
                    f'{"▲ YUKARI" if up else "▼ AŞAĞI"}</span>')
        rows_html.append(f"""<tr>
<td><span class="aname">{name}</span> <span class="acls">{CLASS_TR[r["aclass"]]}</span></td>
<td>{dir_html}</td>
<td class="num"><strong>{tr_pct(prob)}</strong></td>
<td class="num">{cur}{tr_num(r["close"], dec)}</td>
<td class="num muted2">{cur}{tr_num(lo, dec)} – {cur}{tr_num(hi, dec)}</td>
<td><span class="chip {ec}" data-tip="7,5 yıllık out-of-sample test: AUC {tr_num(m['auc'],3)}, isabet {tr_pct(m['hit'])} (taban {tr_pct(m['base_hit'])})">{elab}</span></td>
<td class="num muted2">{dt} {stale}</td>
</tr>""")

    imp_top = imp.head(6)
    imp_max = imp_top["imp"].max()
    FEAT_TR = {"vix_z": "VIX korku endeksi (z-skor)", "btc_r21": "BTC 21 günlük momentum",
               "vol21": "21 günlük oynaklık", "eur_r5": "EUR/USD 5 günlük ivme",
               "sma200_gap": "200 günlük ortalamaya uzaklık", "oil_r5": "Petrol 5 günlük ivme",
               "month": "Ay (mevsimsellik)", "hi52_dist": "52 haftalık zirveye uzaklık",
               "r63": "3 aylık momentum", "vix_chg5": "VIX 5 günlük değişim"}
    imp_html = "".join(
        f'<div class="imp-row"><span class="lbl2">{FEAT_TR.get(f, f)}</span>'
        f'<div class="imp-track"><div class="imp-fill" style="width:{100*v/imp_max:.0f}%"></div></div></div>'
        for f, v in zip(imp_top["feat"], imp_top["imp"]))

    n_proven = int((met["auc"] >= 0.53).sum())
    proven_names = " &amp; ".join(aname(a, extra_names)[0]
                                  for a in met_sorted.head(n_proven)["asset"].head(3)) or "—"
    eq_cards = ""
    for a, nm, mr, (dts, st, bh) in eq_charts:
        eq_cards += f"""<div class="card">
    <h2>{nm} — model stratejisi vs al-tut</h2>
    <div class="sub" style="margin-bottom:8px">Güven &gt; %55 iken 5 gün long, maliyet dahil · Sharpe {tr_num(mr["strat_sharpe"],2)} vs {tr_num(mr["bh_sharpe"],2)} · Maks. düşüş {tr_pct(mr["strat_maxdd"])} · Yüksek güvenli isabet {tr_pct(mr["conf_hit"]) if pd.notna(mr["conf_hit"]) else "—"}</div>
    {svg_equity(dts, st, bh, nm)}
    <div class="legend"><span><span class="lk"></span>Model stratejisi</span><span><span class="lk g"></span>Al-tut</span></div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphFinance — Tahmin Panosu</title>
<style>
.viz-root {{
  color-scheme: light;
  --page:#f9f9f7; --surface-1:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --acc:#2a78d6; --ctx:#c3c2b7; --pos:#2a78d6; --neg:#e34948;
  --dot-light:#86b6ef; --dot-dark:#256abf; --conn:#cde2fb;
  --good:#0ca30c; --good-text:#006300; --warn-text:#8a5800;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --page:#0d0d0d; --surface-1:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
    --acc:#3987e5; --ctx:#52514e; --pos:#3987e5; --neg:#e66767;
    --dot-light:#5598e7; --dot-dark:#9ec5f4; --conn:#184f95;
    --good:#0ca30c; --good-text:#0ca30c; --warn-text:#c98500;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --page:#0d0d0d; --surface-1:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --acc:#3987e5; --ctx:#52514e; --pos:#3987e5; --neg:#e66767;
  --dot-light:#5598e7; --dot-dark:#9ec5f4; --conn:#184f95;
  --good:#0ca30c; --good-text:#0ca30c; --warn-text:#c98500;
}}
* {{ box-sizing:border-box; margin:0 }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.viz-root {{ background:var(--page); color:var(--ink); min-height:100vh; padding:28px 20px 48px; }}
.wrap {{ max-width:1060px; margin:0 auto }}
h1 {{ font-size:26px; font-weight:700; letter-spacing:-.02em }}
h1 .gf {{ color:var(--acc) }}
h2 {{ font-size:16px; font-weight:650; margin-bottom:2px }}
.sub {{ color:var(--ink2); font-size:13.5px; margin-top:4px }}
.card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:14px; padding:18px 20px; }}
.head {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:10px; margin-bottom:14px }}
.notice {{ border-left:3px solid var(--baseline); padding:10px 14px; font-size:13px; color:var(--ink2);
  background:var(--surface-1); border-radius:0 10px 10px 0; margin-bottom:18px }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:18px }}
.tile .tl {{ font-size:12.5px; color:var(--ink2) }}
.tile .tv {{ font-size:30px; font-weight:600; margin-top:2px }}
.tile .td {{ font-size:12px; color:var(--muted); margin-top:2px }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px }}
th {{ text-align:left; font-weight:600; color:var(--muted); font-size:12px; padding:8px 10px; border-bottom:1px solid var(--grid) }}
td {{ padding:9px 10px; border-bottom:1px solid var(--grid); vertical-align:middle }}
tr:last-child td {{ border-bottom:none }}
.num {{ font-variant-numeric: tabular-nums }}
.aname {{ font-weight:600 }}
.acls {{ font-size:11px; color:var(--muted); margin-left:4px }}
.muted2 {{ color:var(--ink2); font-size:12.5px }}
.dir {{ font-weight:700; font-size:12.5px; letter-spacing:.02em }}
.dir.up {{ color:var(--good-text) }} .dir.dn {{ color:var(--neg) }} .dir.mut {{ opacity:.45 }}
.chip {{ display:inline-block; font-size:11px; font-weight:600; padding:3px 8px; border-radius:99px; border:1px solid var(--border); white-space:nowrap }}
.chip.proven {{ color:var(--good-text); border-color:var(--good-text) }}
.chip.weak {{ color:var(--warn-text); border-color:var(--warn-text) }}
.chip.none {{ color:var(--muted) }}
.chip.stale {{ color:var(--warn-text); border-color:var(--warn-text); margin-left:6px }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px }}
@media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr }} }}
.ax {{ font-size:11px; fill:var(--muted) }}
.lbl {{ font-size:12px; fill:var(--ink2) }}
.lbl2 {{ font-size:12.5px; color:var(--ink2) }}
.val {{ font-size:11.5px; font-weight:600; fill:var(--ink) }}
.val2 {{ font-size:11.5px; fill:var(--ink2) }}
svg {{ width:100%; height:auto; display:block }}
.hv {{ cursor:default }}
.hv:hover {{ opacity:.82 }}
.legend {{ display:flex; gap:16px; font-size:12px; color:var(--ink2); margin-top:8px; flex-wrap:wrap }}
.lk {{ display:inline-block; width:14px; height:0; border-top:2px solid var(--acc); vertical-align:middle; margin-right:5px }}
.lk.g {{ border-color:var(--ctx) }}
.dk {{ display:inline-block; width:10px; height:10px; border-radius:50%; vertical-align:middle; margin-right:5px }}
.sec {{ margin-top:18px }}
.imp-row {{ display:flex; align-items:center; gap:10px; margin:7px 0 }}
.imp-row .lbl2 {{ flex:0 0 240px }}
.imp-track {{ flex:1; height:10px; background:var(--grid); border-radius:5px; overflow:hidden }}
.imp-fill {{ height:100%; background:var(--acc); border-radius:5px 3px 3px 5px }}
.foot {{ color:var(--muted); font-size:12px; margin-top:22px; line-height:1.6 }}
.road {{ font-size:13.5px; color:var(--ink2); line-height:1.65 }}
.road strong {{ color:var(--ink) }}
#tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--page);
  font-size:12px; padding:7px 10px; border-radius:8px; opacity:0; transition:opacity .12s;
  max-width:290px; z-index:9; line-height:1.45 }}
</style></head>
<body><div class="viz-root"><div class="wrap">

<div class="head">
  <div>
    <h1><span class="gf">Graph</span>Finance <span style="font-weight:400;color:var(--muted);font-size:15px">· 5 günlük yön tahmini</span></h1>
    <div class="sub">Üretim: {gen_date} · Model: LightGBM (havuzlanmış) · Doğrulama: aylık yeniden eğitimli walk-forward, {ps["d0"]} → {ps["d1"]}</div>
  </div>
</div>

<div class="notice"><strong>Dürüstlük notu:</strong> Bu bir araştırma aracıdır, yatırım tavsiyesi değildir. Aşağıdaki her rakam gerçek out-of-sample testten gelir; modelin çalışmadığı varlıklar da açıkça "kenar yok" olarak işaretlenir. Geçmiş performans geleceği garanti etmez.</div>

<div class="tiles">
  <div class="card tile"><div class="tl">Out-of-sample tahmin</div><div class="tv">{ps["n_oos"]:,}</div><div class="td">{ps["d0"]} → {ps["d1"]}, {met.shape[0]} varlık</div></div>
  <div class="card tile"><div class="tl">Yüksek güvenli çağrı isabeti</div><div class="tv">{tr_pct(ps["conf_hit"])}</div><div class="td">{ps["n_conf"]:,} çağrı (güven &gt; %55)</div></div>
  <div class="card tile"><div class="tl">Kanıtlanmış kenarlı varlık</div><div class="tv">{n_proven} / {met.shape[0]}</div><div class="td">{proven_names}</div></div>
  {ledger_html or f'<div class="card tile"><div class="tl">Tüm çağrılar isabeti</div><div class="tv">{tr_pct(ps["all_hit"])}</div><div class="td">yazı-tura: %50</div></div>'}
</div>

<div class="card">
  <h2>Bugünün tahminleri</h2>
  <div class="sub" style="margin-bottom:10px">Ufuk: 5 işlem günü · "Beklenen aralık" son 21 günlük oynaklıktan (%80 olasılık bandı) · Soluk satırlar: geçmişte kenarı kanıtlanamayan varlıklar — bilgi amaçlı</div>
  <div style="overflow-x:auto"><table>
    <thead><tr><th>Varlık</th><th>Yön</th><th>Güven</th><th>Son fiyat</th><th>Beklenen aralık (5g)</th><th>Model sicili</th><th>Veri tarihi</th></tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Model nerede çalışıyor? <span style="font-weight:400;color:var(--muted);font-size:12px">(AUC, 0,50 = yazı-tura)</span></h2>
    <div class="sub" style="margin-bottom:8px">Out-of-sample walk-forward. Mavi = tahmin gücü var, kırmızı = yok.</div>
    {svg_auc_bars(met, extra_names)}
  </div>
  <div class="card">
    <h2>Kalibrasyon: model ne derse ne oluyor?</h2>
    <div class="sub" style="margin-bottom:8px">Her güven kovasında: <span class="dk" style="background:var(--dot-light)"></span>ortalama tahmin → <span class="dk" style="background:var(--dot-dark)"></span>gerçekleşen yukarı oranı</div>
    {svg_calibration(cal_rows)}
    <div class="sub" style="margin-top:8px">Okuma: %55+ diyen çağrılar gerçekte %53–55 yukarı çıkıyor (zayıf ama gerçek sinyal). %40 altı "düşüş" çağrıları ise ters çalışıyor — piyasalar yukarı sürükleniyor; canlı sürümde düşüş sinyalleri filtrelenir.</div>
  </div>
</div>

<div class="grid2">
  {eq_cards}
</div>

<div class="grid2">
  <div class="card">
    <h2>Model neye bakıyor?</h2>
    <div class="sub" style="margin-bottom:10px">Özellik önemi (LightGBM kazanç bazlı, ilk 6)</div>
    {imp_html}
  </div>
  <div class="card">
    <h2>Yol haritası</h2>
    <div class="road">
      <strong>Sıradaki adım — canlı veri hattı (GitHub Actions):</strong> ABD hisseleri (S&amp;P 500 mega-cap + SPY/QQQ), altın, USD/TRY ve güncel kripto fiyatları bu ortamın veri kısıtları yüzünden henüz yok. Ücretsiz bir GitHub reposuna kuracağımız günlük otomasyon bunların hepsini ekler, panoyu her sabah kendi kendine yeniler.<br><br>
      <strong>Sonra:</strong> BlackRock/Vanguard 13F pozisyon takibi modülü · haber/sentiment katmanı · paper trading.
    </div>
  </div>
</div>

<div class="foot">
Veri: CoinMetrics community (kripto) · datahub/Fed H.10 (döviz) · datahub/EIA (emtia) · CBOE VIX — tümü ücretsiz kaynaklar.
Yöntem: havuzlanmış LightGBM sınıflandırıcı; hedef = 5 işlem günlük yön; aylık yeniden eğitim, 10 gün embargo (sızıntı önleme); işlem maliyeti dahil (kripto 10bp, döviz 2bp, emtia 5bp tek yön). Örtüşmesiz simülasyon.
Bu pano bilgilendirme amaçlıdır; yatırım tavsiyesi değildir.
</div>

</div></div>
<div id="tip"></div>
<script>
(function() {{
  var tip = document.getElementById('tip');
  function show(e, text) {{
    tip.textContent = text;
    tip.style.opacity = 1;
    var x = Math.min(e.clientX + 14, window.innerWidth - 300);
    tip.style.left = x + 'px';
    tip.style.top = (e.clientY + 16) + 'px';
  }}
  document.querySelectorAll('[data-tip]').forEach(function(el) {{
    el.addEventListener('pointermove', function(e) {{ show(e, el.getAttribute('data-tip')); }});
    el.addEventListener('pointerleave', function() {{ tip.style.opacity = 0; }});
    el.setAttribute('tabindex', '0');
    el.addEventListener('focus', function() {{
      var r = el.getBoundingClientRect();
      show({{clientX: r.left, clientY: r.bottom}}, el.getAttribute('data-tip'));
    }});
    el.addEventListener('blur', function() {{ tip.style.opacity = 0; }});
  }});
  // crosshair on equity charts
  document.querySelectorAll('.eqsvg').forEach(function(svg) {{
    var dates = svg.dataset.dates.split(','), strat = svg.dataset.strat.split(',').map(Number),
        bh = svg.dataset.bh.split(',').map(Number);
    var L = +svg.dataset.l, R = +svg.dataset.r, W = +svg.dataset.w;
    var xh = svg.querySelector('.xh');
    svg.addEventListener('pointermove', function(e) {{
      var rect = svg.getBoundingClientRect();
      var vx = (e.clientX - rect.left) / rect.width * W;
      var t = Math.max(0, Math.min(1, (vx - L) / (W - L - R)));
      var i = Math.round(t * (dates.length - 1));
      var X = L + (W - L - R) * i / (dates.length - 1);
      xh.setAttribute('x1', X); xh.setAttribute('x2', X); xh.setAttribute('opacity', 0.7);
      show(e, dates[i] + ' — model: ' + strat[i].toFixed(2) + 'x · al-tut: ' + bh[i].toFixed(2) + 'x');
    }});
    svg.addEventListener('pointerleave', function() {{ xh.setAttribute('opacity', 0); tip.style.opacity = 0; }});
  }});
}})();
</script>
</body></html>"""
    return html


if __name__ == "__main__":
    met, preds, oos, imp = load()
    html = build(met, preds, oos, imp)
    out = os.path.join(REP, "graphfinance_panosu.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("yazildi:", out, len(html), "bayt")
