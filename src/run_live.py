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
from features import build_features
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


def main():
    t0 = time.time()
    print("1/5 veri katmani...")
    gh_panel, vix = load_panel()
    live_panel = load_live_panel()
    n_live = live_panel["asset"].nunique() if not live_panel.empty else 0
    print(f"   mirror: {gh_panel['asset'].nunique()} varlik | canli: {n_live} varlik")
    panel = merge_panels(gh_panel, live_panel)
    panel.to_parquet(os.path.join(STORE, "panel.parquet"))

    df, feat_cols = build_features(panel, vix)
    assets = set(df["asset"].unique())
    print(f"   toplam {len(assets)} varlik, {len(df)} satir")

    # --- aylik dogrulama / egitim karari ---
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    met_path = os.path.join(REP, "metrics.csv")
    need_full = (
        not os.path.exists(met_path)
        or not os.path.exists(MODEL_PATH)
        or meta.get("validated_month") != month_key()
        or not assets.issubset(set(pd.read_csv(met_path)["asset"])) )

    if need_full:
        print("2/5 AYLIK tam dogrulama (walk-forward) + egitim...")
        oos = walk_forward(df, feat_cols)
        oos.to_parquet(os.path.join(REP, "oos_predictions.parquet"))
        met = per_asset_metrics(oos)
        met.to_csv(met_path, index=False)
        calibration(oos).to_csv(os.path.join(REP, "calibration.csv"))
        m = train_final(df, feat_cols)
        joblib.dump({"model": m, "feat_cols": feat_cols}, MODEL_PATH)
        imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
        imp.to_csv(os.path.join(REP, "feature_importance.csv"))
        meta = {"validated_month": month_key(),
                "trained_at": str(date.today()),
                "n_assets": len(assets), "n_rows": len(df)}
        json.dump(meta, open(META_PATH, "w"))
    else:
        print("2/5 gunluk mod: kayitli model kullaniliyor "
              f"(dogrulama: {meta.get('validated_month')})")
        bundle = joblib.load(MODEL_PATH)
        m, feat_cols = bundle["model"], bundle["feat_cols"]

    print("3/5 guncel tahminler...")
    bundle = joblib.load(MODEL_PATH)
    m, model_feats = bundle["model"], bundle["feat_cols"]
    # ozellik kolonlari egitimdekiyle ayni sirada olmali
    preds = latest_predictions(df, model_feats, m)
    preds.to_csv(os.path.join(REP, "latest_predictions.csv"), index=False)

    print("4/5 tahmin defteri...")
    led = L.load_ledger(LEDGER_PATH)
    led = L.resolve(led, panel)
    led = L.append_predictions(led, preds[["asset", "aclass", "date", "close", "p_up"]])
    led.to_csv(LEDGER_PATH, index=False)
    rh = L.rolling_hit(led)
    rh.to_csv(os.path.join(STORE, "rolling_hit.csv"), index=False)
    done = led[led["resolved"] == 1]
    if len(done):
        print(f"   defter: {len(led)} tahmin, {len(done)} cozuldu, "
              f"gercek isabet {done['correct'].mean():.1%}")
    else:
        print(f"   defter: {len(led)} tahmin, henuz cozulen yok")

    print("5/5 pano...")
    import build_dashboard as BD
    met = pd.read_csv(met_path)
    oos = pd.read_parquet(os.path.join(REP, "oos_predictions.parquet"))
    imp = pd.read_csv(os.path.join(REP, "feature_importance.csv"),
                      names=["feat", "imp"], skiprows=1)
    html = BD.build(met, preds, oos, imp, extra_names=ASSET_NAMES_LIVE, led=led)
    for out in (os.path.join(DOCS, "index.html"),
                os.path.join(REP, "graphfinance_panosu.html")):
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"bitti — {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
