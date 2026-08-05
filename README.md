# GraphFinance

Çok varlıklı (kripto · hisse · döviz · emtia) 5 günlük yön tahmini otomasyonu.
Her sabah 06:00'da (TR) kendi kendine çalışır, tahmin panosunu yeniler ve her
tahminini deftere yazıp 5 gün sonra sonucuyla yüzleşir.

**Pano:** `docs/index.html` (GitHub Pages açıksa yayın linki) ·
**Plan:** [PLAN.md](PLAN.md)

## Nasıl çalışır

```
veri (Stooq/Yahoo/Binance/datahub, hepsi ücretsiz)
  → özellikler (momentum, oynaklık, VIX, çapraz-varlık)
  → LightGBM (aylık yeniden eğitim + walk-forward yeniden doğrulama)
  → tahminler → tahmin defteri (data_store/ledger.csv)
  → pano (docs/index.html)
```

- **Günlük:** kayıtlı modelle tahmin, defter çözümü, pano.
- **Aylık (otomatik):** tam walk-forward yeniden doğrulama — her varlığın
  "kenar" durumu güncellenir; kenarı kanıtlanamayan varlık panoda açıkça
  "kenar yok" görünür.

## Çalıştırma

```bash
pip install -r requirements.txt
python src/run_live.py          # canlı kaynaklar kapalıysa mirror kaynaklara düşer
```

## Dürüstlük sözleşmesi

Bu depo bir araştırma aracıdır, yatırım tavsiyesi değildir. Tüm isabet
rakamları out-of-sample'dır; tahmin defteri hiçbir satırı silmez.
