# -*- coding: utf-8 -*-
"""GraphFinance — günlük canlı çalıştırma (GitHub Actions giriş noktası).

Akış:
  1. Veri: GitHub-mirror kaynaklar (her yerde çalışır) + canlı kaynaklar (Actions'ta)
  2. Aylık: tam walk-forward yeniden doğrulama + model yeniden eğitimi
     Günlük: kayıtlı modelle tahmin
  3. Tahmin defteri: yeni tahminleri ekle, ufku dolanları çöz
  4. Pano: docs/index.html (GitHub Pages)
"""
import json
import os
import sys
import time
from datetime import date

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fetch import load_panel
from sources_live import load_live_panel, ASSET_NAMES_LIVE
from features import build_features, BASE_FEATS, EXT_FEATS, FEATURES_VERSION
from model import walk_forward, train_final, latest_predictions
from backtest import per_asset_metrics, calibration
import ledger as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "reports")
DOCS = os.path.join(ROOT, "docs")
STORE = os.path.join(ROOT, "data_store")
MODELS = os.path.join(ROOT, "models")
for d in (REP, DOCS, STORE, MODELS):
    os.makedirs(d, exist_ok=True)

META_PATH = os.path.join(MODELS, "meta.json")
MODEL_PATH = os.path.join(MODELS, "model.joblib")
LEDGER_PATH = os.path.join(STORE, "ledger.csv")


def merge_panels(gh: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Canlı veri aynı varlığın mirror geçmişinin ÜZERİNE yazılır (tarih bazında)."""
    if live.empty:
        return gh
    both = pd.concat([gh, live], ignore_index=True)
    both = (both.sort_values(["asset", "date"])
                .drop_duplicates(["asset", "date"], keep="last"))
    return both.reset_index(drop=True)


def month_key(d=None):
    d = d or date.today()
    return f"{d.year}-{d.month:02d}"


def routed_predictions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Şampiyona göre tahmin: base / ext / karma (sınıf bazlı yönlendirme)."""
    from model import top_drivers
    idx = df.groupby("asset")["date"].idxmax()
    last = df.loc[idx].copy()
    champ = bundle.get("champion", "base")
    if champ == "mix":
        use_ext = last["aclass"].isin(set(bundle.get("ext_classes", [])))
    else:
        use_ext = pd.Series(champ == "ext", index=last.index)
    parts = []
    for flag, key in ((False, "base"), (True, "ext")):
        sub = last[use_ext == flag]
        if sub.empty or key not in bundle["models"]:
            continue
        m, fc = bundle["models"][key], bundle["feats"][key]
        sub = sub.copy()
        sub["p_up"] = m.predict_proba(sub[fc])[:, 1]
        sub["drivers"] = top_drivers(m, sub, fc)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return (out[["asset", "aclass", "date", "close", "p_up", "vol21", "drivers"]]
            .sort_values("p_up", ascending=False))


def main():
    t0 = time.time()
    print("1/5 veri katmani...")
    gh_panel, vix = load_panel()
    live_panel = load_live_panel()
    n_live = live_panel["asset"].nunique() if not live_panel.empty else 0
    print(f"   mirror: {gh_panel['asset'].nunique()} varlik | canli: {n_live} varlik")
    panel = merge_panels(gh_panel, live_panel)
    panel.to_parquet(os.path.join(STORE, "panel.parquet"))

    print("1.5/5 olay radari (GDELT)...")
    ev_z, heat, book = pd.DataFrame(), [], []
    try:
        from events import load_event_features, current_heat, playbook
        ev_z = load_event_features()
        heat = current_heat(ev_z)
        book = playbook(panel)
    except Exception as e:
        print(f"   ! olay radari atlandi: {type(e).__name__}")
    print("1.7/5 haber akisi (Alpaca)...")
    news_daily, news_latest = pd.DataFrame(), {}
    news_feats = pd.DataFrame()
    try:
        from news import update_news, news_features, bad_news_assets
        news_daily, news_latest = update_news()
        news_feats = news_features(news_daily)
        bad_news = bad_news_assets(news_latest)
        bn_path = os.path.join(STORE, "bad_news.json")
        if os.path.exists(bn_path):  # saatlik kosunun daha taze filtresi
            try:
                bj = json.load(open(bn_path))
                if (pd.Timestamp.now('UTC') - pd.Timestamp(bj["asof"])).total_seconds() < 6 * 3600:
                    bad_news |= set(bj.get("bad_news", []))
            except Exception:
                pass
        if bad_news:
            print(f"   kotu haber filtresi: {sorted(bad_news)}")
    except Exception as e:
        bad_news = set()
        print(f"   ! haber akisi atlandi: {type(e).__name__}")
    df, feat_cols = build_features(panel, vix, events=ev_z, news=news_feats)
    assets = set(df["asset"].unique())
    print(f"   toplam {len(assets)} varlik, {len(df)} satir")

    # --- aylik dogrulama / egitim karari ---
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    met_path = os.path.join(REP, "metrics.csv")
    need_full = (
        not os.path.exists(met_path)
        or not os.path.exists(MODEL_PATH)
        or meta.get("validated_month") != month_key()
        or meta.get("features_version") != FEATURES_VERSION
        or not assets.issubset(set(pd.read_csv(met_path)["asset"])) )

    if need_full:
        print("2/5 AYLIK dogrulama: sampiyon/rakip A/B testi...")
        ab_rows, results = [], {}
        for name, feats in (("base", BASE_FEATS), ("ext", EXT_FEATS)):
            oos_i = walk_forward(df, feats)
            met_i = per_asset_metrics(oos_i)
            g = oos_i.dropna(subset=["y"])
            conf = g[(g["p_up"] > 0.55) | (g["p_up"] < 0.45)]
            conf_hit = float(((conf["p_up"] > 0.5).astype(int) == conf["y"]).mean())
            auc_mean = float(met_i["auc"].mean())
            results[name] = dict(oos=oos_i, met=met_i, auc=auc_mean, conf=conf_hit)
            print(f"   {name}: ort.AUC={auc_mean:.4f}  guvenli-isabet={conf_hit:.4f}  ({len(feats)} ozellik)")
            for _, r in met_i.iterrows():
                ab_rows.append(dict(model=name, asset=r["asset"], auc=r["auc"], hit=r["hit"]))
        # 1) saf rakip anlamli farkla kazanirsa tahta gecer
        # 2) degilse KARMA denenir: ext'in sinif bazinda acik ara kazandigi siniflara ext,
        #    digerlerine base — karma da ancak havuzda base'i yenerse secilir
        champion, ext_classes = "base", []
        if (results["ext"]["auc"] >= results["base"]["auc"] + 0.002
                and results["ext"]["conf"] >= results["base"]["conf"] - 0.005):
            champion = "ext"
        else:
            bc = results["base"]["met"].groupby("aclass")["auc"].mean()
            ec = results["ext"]["met"].groupby("aclass")["auc"].mean()
            ext_classes = sorted(c for c in ec.index if ec[c] > bc.get(c, 1.0) + 0.005)
            if ext_classes:
                ob, oe = results["base"]["oos"], results["ext"]["oos"]
                mix = pd.concat([oe[oe["aclass"].isin(ext_classes)],
                                 ob[~ob["aclass"].isin(ext_classes)]], ignore_index=True)
                met_mix = per_asset_metrics(mix)
                if float(met_mix["auc"].mean()) >= results["base"]["auc"] + 0.002:
                    champion = "mix"
                    results["mix"] = dict(oos=mix, met=met_mix)
                    print(f"   karma kabul: {ext_classes} siniflari grafik-paketli modele gecti")
                else:
                    print(f"   karma denendi ({ext_classes}) ama havuzda base'i yenemedi")
        print(f"   SAMPIYON: {champion}")
        pd.DataFrame(ab_rows).to_csv(os.path.join(REP, "ab_test.csv"), index=False)
        oos, met = results[champion]["oos"], results[champion]["met"]
        oos.to_parquet(os.path.join(REP, "oos_predictions.parquet"))
        met.to_csv(met_path, index=False)
        calibration(oos).to_csv(os.path.join(REP, "calibration.csv"))
        models = {"base": train_final(df, BASE_FEATS)}
        feats = {"base": BASE_FEATS}
        if champion in ("ext", "mix"):
            models["ext"] = train_final(df, EXT_FEATS)
            feats["ext"] = EXT_FEATS
        bundle = {"models": models, "feats": feats,
                  "champion": champion, "ext_classes": ext_classes}
        joblib.dump(bundle, MODEL_PATH)
        imp_key = "ext" if champion == "ext" else "base"
        imp = pd.Series(models[imp_key].feature_importances_,
                        index=feats[imp_key]).sort_values(ascending=False)
        imp.to_csv(os.path.join(REP, "feature_importance.csv"))
        meta = {"validated_month": month_key(), "features_version": FEATURES_VERSION,
                "champion": champion, "ext_classes": ext_classes,
                "trained_at": str(date.today()),
                "n_assets": len(assets), "n_rows": len(df)}
        json.dump(meta, open(META_PATH, "w"))
    else:
        print("2/5 gunluk mod: kayitli model kullaniliyor "
              f"(dogrulama: {meta.get('validated_month')}, sampiyon: {meta.get('champion')})")
        bundle = joblib.load(MODEL_PATH)

    print("3/5 guncel tahminler...")
    bundle = joblib.load(MODEL_PATH)
    preds = routed_predictions(df, bundle)
    preds.to_csv(os.path.join(REP, "latest_predictions.csv"), index=False)

    print("4/5 tahmin defteri...")
    met_now = pd.read_csv(met_path).set_index("asset")
    edge_map = {a: int(met_now.loc[a, "auc"] >= 0.53) for a in met_now.index}
    led = L.load_ledger(LEDGER_PATH)
    led = L.resolve(led, panel)
    led = L.append_predictions(led, preds[["asset", "aclass", "date", "close", "p_up", "drivers"]],
                               edge_map=edge_map)
    led.to_csv(LEDGER_PATH, index=False)
    rh = L.rolling_hit(led)
    rh.to_csv(os.path.join(STORE, "rolling_hit.csv"), index=False)

    # --- CANLI BEKCI (18 gunluk otopsinin dersleri) ---
    import numpy as np
    dres = led[led["resolved"] == 1].copy()
    if len(dres):
        conf_side = dres["p_up"].where(dres["p_up"] >= 0.5, 1 - dres["p_up"])
        dd = dres[conf_side > 0.55]
        g = dd.groupby("asset")["correct"].agg(["size", "mean"])
        # 1) ASKI: canli sicili cokenler sinyal setinden cikar (backtest ne derse desin)
        suspended = set(g[(g["size"] >= 4) & (g["mean"] < 0.40)].index)
        # 2) SOGUMA: son 7 gunde buyuk zarar ettiren varliga yeniden girme (dusen bicak freni)
        up_res = dres[dres["direction"] == "up"].copy()
        up_res["net"] = np.exp(up_res["realized_ret"]) - 1
        recent = up_res[pd.to_datetime(up_res["resolve_date"]) >=
                        pd.Timestamp.today() - pd.Timedelta(days=7)]
        cooldown = set(recent[recent["net"] < -0.04]["asset"]) - suspended
    else:
        suspended, cooldown = set(), set()
    if suspended:
        print(f"   bekci ASKI: {sorted(suspended)}")
    if cooldown:
        print(f"   bekci SOGUMA: {sorted(cooldown)}")
    done = led[led["resolved"] == 1]
    if len(done):
        print(f"   defter: {len(led)} tahmin, {len(done)} cozuldu, "
              f"gercek isabet {done['correct'].mean():.1%}")
    else:
        print(f"   defter: {len(led)} tahmin, henuz cozulen yok")

    print("4.5/5 temettu radari...")
    income_df = pd.DataFrame()
    try:
        from income import build_income
        income_df = build_income(panel)
    except Exception as e:
        print(f"   ! temettu modulu atlandi: {type(e).__name__}")

    print("4.55/5 bilanco takvimi...")
    earnings_soon = set()
    try:
        from earnings import upcoming_earnings
        earnings_soon = upcoming_earnings(days=4)
        if earnings_soon:
            print(f"   bilancoya yakin (4 gun): {sorted(earnings_soon & assets)}")
    except Exception as e:
        print(f"   ! bilanco takvimi atlandi: {type(e).__name__}")

    print("4.6/5 paper trading...")
    paper = None
    try:
        from paper_trade import run_paper
        paper = run_paper(preds, pd.read_csv(met_path),
                          skip=suspended | cooldown | earnings_soon | bad_news)
    except Exception as e:
        print(f"   ! paper modulu atlandi: {type(e).__name__}")

    print("4.7/5 balina takibi...")
    whale_list = []
    try:
        from whales import build_whales
        whale_list = build_whales()
    except Exception as e:
        print(f"   ! balina modulu atlandi: {type(e).__name__}")

    print("5/5 pano...")
    import build_dashboard as BD
    met = pd.read_csv(met_path)
    oos = pd.read_parquet(os.path.join(REP, "oos_predictions.parquet"))
    imp = pd.read_csv(os.path.join(REP, "feature_importance.csv"),
                      names=["feat", "imp"], skiprows=1)
    # yorum katmani icin piyasa durumu
    last = df.sort_values("date").groupby("asset").tail(1)
    def _sv(col):
        s = last[col].dropna()
        return float(s.iloc[-1]) if len(s) else None
    yorum = {
        "vix_pct": _sv("vix_pct"), "vix_z": _sv("vix_z"),
        "risk_off": int(last["risk_off"].max()) if last["risk_off"].notna().any() else 0,
        "squeeze_assets": last[last["squeeze"] < 0.6]["asset"].tolist(),
    }
    usdtry_last = None
    ut = panel[panel["asset"] == "USDTRY"]
    if len(ut):
        usdtry_last = float(ut.sort_values("date")["close"].iloc[-1])
    # saatlik haber kosusu (run_news.py) panoyu yeniden kurabilsin diye durum kaydi
    try:
        json.dump({"yorum": yorum, "usdtry": usdtry_last,
                   "suspended": sorted(suspended), "cooldown": sorted(cooldown),
                   "heat": heat, "book": book, "earnings_soon": sorted(earnings_soon),
                   "whales": whale_list, "paper": paper, "asof": str(date.today())},
                  open(os.path.join(REP, "dash_state.json"), "w"), ensure_ascii=False, default=str)
    except Exception as e:
        print(f"   ! pano durumu kaydedilemedi: {type(e).__name__}")
    html = BD.build(met, preds, oos, imp, extra_names=ASSET_NAMES_LIVE, led=led,
                    yorum=yorum, income=income_df, usdtry=usdtry_last,
                    whales=whale_list, paper=paper,
                    suspended=suspended, cooldown=cooldown,
                    heat=heat, book=book, earnings_soon=earnings_soon,
                    news_latest=news_latest, bad_news=bad_news)
    for out in (os.path.join(DOCS, "index.html"),
                os.path.join(REP, "graphfinance_panosu.html")):
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"bitti — {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
