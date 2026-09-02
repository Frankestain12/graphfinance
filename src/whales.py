# -*- coding: utf-8 -*-
"""GraphFinance — balina takibi (SEC EDGAR 13F).
Ünlü fonların son açıkladığı pozisyonlar + önceki çeyreğe göre değişimler.
- Veri ücretsiz ve resmî (SEC); 45 güne kadar gecikmeli, sadece ABD uzun pozisyonları.
- CIK'ler çalışma anında isim kontrolüyle doğrulanır; uyuşmayan fon atlanır.
- Önbellek: data_store/whales.json — accession değişmediyse XML yeniden çekilmez.
Sandbox'ta SEC kapalı — önbellek varsa onu kullanır, yoksa boş döner.
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

H = {"User-Agent": "GraphFinance personal research bedirhan.icli@icloud.com"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data_store", "whales.json")

# (cik, beklenen isim parçası, görünen ad)
FUNDS = [
    (1067983, "BERKSHIRE", "Berkshire Hathaway — Warren Buffett"),
    (1350694, "BRIDGEWATER", "Bridgewater Associates — Ray Dalio'nun fonu"),
    (1336528, "PERSHING", "Pershing Square — Bill Ackman"),
    (1536411, "DUQUESNE", "Duquesne Family Office — Stanley Druckenmiller"),
    (1649339, "SCION", "Scion Asset Management — Michael Burry"),
    (1656456, "APPALOOSA", "Appaloosa — David Tepper"),
]


def _get(url, timeout=60):
    r = requests.get(url, headers=H, timeout=timeout)
    r.raise_for_status()
    return r


def latest_13f_list(cik: int):
    j = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    name = j.get("name", "")
    rec = j.get("filings", {}).get("recent", {})
    out = []
    for form, acc, rdate, fdate in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                       rec.get("reportDate", []), rec.get("filingDate", [])):
        if form == "13F-HR":
            out.append(dict(acc=acc, report=rdate, filed=fdate))
    return name, out[:2]  # en yeni iki dosyalama


def _parse_infotable_regex(xml: str) -> dict:
    """XML cozumleyici (onek/namespace sorunlari) basarisiz olursa: duz metin desenleriyle oku."""
    holdings = {}
    def _tag(block, name):
        m = re.search(rf"<(?:[\w\-]+:)?{name}[^>]*>\s*([^<]*?)\s*</(?:[\w\-]+:)?{name}>", block, flags=re.I)
        return m.group(1).strip() if m else ""
    for blk in re.findall(r"<(?:[\w\-]+:)?infoTable[^>]*>(.*?)</(?:[\w\-]+:)?infoTable>", xml, flags=re.S | re.I):
        if _tag(blk, "putCall"):
            continue
        cusip, name, val = _tag(blk, "cusip"), _tag(blk, "nameOfIssuer"), _tag(blk, "value")
        try:
            val = float(val.replace(",", ""))
        except ValueError:
            continue
        if cusip:
            h = holdings.setdefault(cusip, {"name": name.title(), "value": 0.0})
            h["value"] += val
    return holdings


def _parse_infotable(xml: str) -> dict:
    xml = xml.lstrip("\ufeff \r\n\t")
    xml = re.sub(r'\sxmlns(:[\w\-]+)?="[^"]*"', "", xml)   # TUM namespace bildirimlerini at
    xml = re.sub(r'\s[\w\-]+:[\w\-]+="[^"]*"', "", xml)  # xsi:schemaLocation gibi onekli ozellikleri at
    xml = re.sub(r'<(/?)[\w\-]+:', r'<\1', xml)            # ns1: gibi etiket oneklerini at
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return _parse_infotable_regex(xml)
    holdings = {}
    for it in root.iter():
        if not it.tag.endswith("infoTable"):
            continue
        d = {c.tag.split("}")[-1]: c for c in it}
        try:
            if d.get("putCall") is not None and (d["putCall"].text or "").strip():
                continue  # opsiyonlar haric
            cusip = d["cusip"].text.strip()
            name = d["nameOfIssuer"].text.strip().title()
            val = float(d["value"].text)
            h = holdings.setdefault(cusip, {"name": name, "value": 0.0})
            h["value"] += val
        except (KeyError, AttributeError, ValueError):
            continue
    return holdings


def fetch_holdings(cik: int, acc: str) -> dict:
    """accession -> {cusip: {name, value}} (value: USD). Adaylari sirayla dener."""
    accn = acc.replace("-", "")
    idx = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json").json()
    # primary_doc SON aday: bazi fonlar (kucuk dosyalayanlar) infotable'i ana belgeye gomer
    cands = [it["name"] for it in idx.get("directory", {}).get("item", [])
             if it["name"].lower().endswith(".xml")]
    cands.sort(key=lambda n: ("infotable" not in n.lower(), "table" not in n.lower(),
                              "primary_doc" in n.lower(), n))
    last_err = None
    for cand in cands:
        try:
            xml = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{cand}").text
            if "infotable" not in xml.lower():
                continue
            h = _parse_infotable(xml)
            if h:
                return h
        except Exception as e:
            last_err = e
            continue
    # son care: tam dosyalama metni (.txt) icindeki <XML> bloklari
    try:
        txt = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{acc}.txt").text
        for blk in re.findall(r"<XML>(.*?)</XML>", txt, flags=re.S | re.I):
            if "infotable" in blk.lower():
                h = _parse_infotable(blk.strip())
                if h:
                    return h
    except Exception as e:
        last_err = e
    raise ValueError(f"infotable cozulemedi ({len(cands)} aday; son hata: {type(last_err).__name__ if last_err else 'yok'}: {str(last_err)[:80]})")


def summarize(cur: dict, prev: dict | None):
    total = sum(v["value"] for v in cur.values()) or 1.0
    top = sorted(cur.items(), key=lambda kv: -kv[1]["value"])[:5]
    top_rows = [dict(name=v["name"], pct=round(100 * v["value"] / total, 1)) for _, v in top]
    yeni, cikis = [], []
    if prev:
        pv_total = sum(v["value"] for v in prev.values()) or 1.0
        yeni = [cur[c]["name"] for c in sorted(set(cur) - set(prev),
                key=lambda c: -cur[c]["value"]) if cur[c]["value"] / total > 0.005][:4]
        cikis = [prev[c]["name"] for c in sorted(set(prev) - set(cur),
                 key=lambda c: -prev[c]["value"]) if prev[c]["value"] / pv_total > 0.005][:4]
    return dict(total_usd=round(total), top=top_rows, yeni=yeni, cikis=cikis)


def build_whales(log=print) -> list:
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = {w["cik"]: w for w in json.load(open(CACHE))}
        except Exception:
            cache = {}
    out = []
    for cik, expect, display in FUNDS:
        try:
            name, filings = latest_13f_list(cik)
            if expect not in name.upper():
                log(f"  ! balina {display}: CIK isim uyusmadi ({name}), atlandi")
                continue
            if not filings:
                log(f"  ! balina {display}: 13F yok")
                continue
            acc = filings[0]["acc"]
            if cik in cache and cache[cik].get("acc") == acc:
                out.append(cache[cik])  # degisiklik yok, onbellek
                continue
            cur = fetch_holdings(cik, acc)
            prev = fetch_holdings(cik, filings[1]["acc"]) if len(filings) > 1 else None
            time.sleep(0.4)
            s = summarize(cur, prev)
            out.append(dict(cik=cik, fund=display, acc=acc,
                            report=filings[0]["report"], filed=filings[0]["filed"],
                            n_pos=len(cur), **s))
            log(f"  balina {display}: {filings[0]['report']} donemi, {len(cur)} pozisyon")
        except Exception as e:
            if cik in cache:
                out.append(cache[cik])
                log(f"  balina {display}: guncellenemedi ({type(e).__name__}), onbellek")
            else:
                log(f"  ! balina {display}: {type(e).__name__} {str(e)[:110]}")
    if out:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(out, open(CACHE, "w"), ensure_ascii=False)
    log(f"   balina takibi: {len(out)} fon")
    return out
