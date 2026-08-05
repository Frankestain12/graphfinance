# GraphFinance — Mimari ve Evrim Planı

*Hazırlayan: Claude · 4 Ağustos 2026 · Sahibi: Bedirhan*

**Ne bu:** Kripto, global hisseler, döviz ve emtiada 5 günlük yön tahmini üreten; her sabah kendi kendine çalışan; yaptığı her tahmini kaydedip ders çıkaran, kendini yenileyen bir otomasyon.

**Ne değil:** Garantili para makinesi. Sistem kenar (istatistiksel avantaj) bulur, ölçer ve sadece kanıtlananı söyler. "Kenar yok" da bir cevaptır ve açıkça gösterilir.

---

## 1. Dört katman

### Veri katmanı — hepsi ücretsiz
| Sinyal | Kaynak | Sıklık |
|---|---|---|
| ABD hisseleri, ETF'ler (SPY, QQQ...), altın, USD/TRY | Stooq + Yahoo (yedekli) | Günlük |
| Kripto (BTC, ETH, SOL, XRP...) | Binance/Bybit public API | Günlük |
| Makro: faiz, DXY, enflasyon | FRED | Günlük |
| Korku endeksi (VIX) | CBOE/datahub | Günlük |
| Haber akışı + ton skoru | GDELT (küresel, 15 dk'lık, Türk medyası dahil) | Faz 3 |
| Olay olasılıkları ("Trump'ın sağlığı" dahil siyasi/jeopolitik olaylar) | Polymarket/Kalshi tahmin piyasaları | Faz 3 |
| BlackRock/Vanguard/fon pozisyonları | SEC EDGAR 13F (çeyreklik) + iShares günlük ETF CSV | Faz 3 |

### Sinyal katmanı
- Havuzlanmış LightGBM sınıflandırıcı; hedef: 5 işlem günlük yön.
- 23+ özellik: momentum (1g–3ay), oynaklık, RSI, hareketli ortalama açıkları, 52 haftalık zirveye uzaklık, VIX z-skoru, çapraz-varlık ivmeleri (petrol, EUR, BTC), takvim.
- Doğrulama: **walk-forward** (aylık yeniden eğitim, 10 gün embargo, işlem maliyeti dahil). İlk doğrulama: 25.142 out-of-sample tahmin, Oca 2019 – Tem 2026.

### Karar katmanı
- Sadece **kanıtlanmış kenarlı** varlıklarda sinyal (ilk testte: USD/CNY AUC 0,542; Doğalgaz 0,537).
- Kalibrasyon bulgusu gereği düşüş çağrıları filtrelenir (ters çalışıyor); güven < %55 ise "işlem yok".
- Kill-switch: kayan 6 aylık isabet taban çizgisinin altına düşerse varlık otomatik yayından kalkar.

### Çıktı katmanı
- Her sabah kendini yenileyen pano (GitHub Pages linki — telefondan açılır).
- `predictions.csv` + tahmin defteri (aşağıda).
- Opsiyonel: Telegram'a sabah özeti.

---

## 2. Evrim döngüsü — "sürekli öğrenme"

1. **Tahmin defteri:** Her tahmin, özellik anlık görüntüsüyle kaydedilir; 5 gün sonra sonucu otomatik yazılır. Hiçbir tahmin unutulmaz.
2. **Aylık yeniden eğitim:** Model her ay son veriyle güncellenir; rejim değişimlerine uyum sağlar.
3. **Kenar bekçisi:** Varlık bazında kayan isabet izlenir → otomatik terfi/emeklilik. Kimseye sormaz, raporlar.
4. **Şampiyon/rakip yarışı:** Her ay yeni özellik setleri ve parametrelerle rakip modeller eğitilir; şampiyonu out-of-sample'da **anlamlı farkla** yenen tahta geçer. (Aşırı deneme tuzağına karşı sert eşik.)
5. **Hata otopsisi:** En büyük hataların ortak örüntüsü aranır (ör. Fed günleri) → yeni özellik adayı → 4. adımdaki yarışa girer.
6. **Aylık araştırmacı (Claude):** Zamanlanmış görev ayda bir Claude'u çalıştırır: defteri okur, iyileştirme dener, kanıtlananı yayına alır, tek sayfalık "bu ay ne öğrendik" raporu bırakır.

> Sürekli öğrenen ≠ sürekli kazanan. Sistemin en değerli dersi bazen "bu sinyal öldü, işlem yapma"dır.

---

## 3. Fazlar

| Faz | İçerik | Zaman |
|---|---|---|
| **1** | GitHub reposu + Actions: canlı veri hattı (hisse, altın, USDTRY, güncel kripto), mevcut model, günlük pano, tahmin defteri | Kurulum günü |
| **2** | Varlık evreni ~50'ye çıkar; kenar testleri otomatik raporlanır; kenar bekçisi devrede | 1. hafta |
| **3** | GDELT haber tonu + Polymarket olay olasılıkları + 13F balina takibi — her biri tek tek kanıtlanarak girer | 1. ay |
| **4** | Şampiyon/rakip + aylık Claude araştırmacısı tam otomatik | 1.–2. ay |
| **5** | Paper trading (Binance testnet / Alpaca paper) — sahte parayla canlı isabet kaydı | Kenarlar oturunca |
| **6** | Küçük gerçek sermaye — **yalnızca** aylarca paper trading kanıtından sonra; risk kuralları: pozisyon limiti, max drawdown kill-switch | Karar senin |

## 4. Altyapı ve maliyet

- **GitHub Actions:** günlük cron 06:00 TR; çalışma ~5 dk/gün → ücretsiz kotanın ~%8'i. Sunucu yok, kira yok. **Toplam: $0/ay.**
- Kod: Python (pandas, LightGBM). Tüm çıktılar repoda versiyonlu — geçmiş hiçbir zaman silinmez, "isabet oranımız neydi" sorusunun cevabı her zaman denetlenebilir.
- Türkiye notu: veri çekmek serbest; ileride canlı işlem yapılacaksa SPK lisanslı borsalar (Binance TR, OKX TR, Bybit) kullanılır.

## 5. Kurulum için gerekenler (5 dakika)

1. GitHub hesabı (yoksa github.com'dan aç).
2. Token: GitHub → Settings → Developer settings → Personal access tokens (classic) → Generate new token → izinler: `repo` + `workflow`, süre: 90 gün.
3. Token'ı Claude'a ver → repo kurulur, ilk çalıştırma yapılır, pano linki elinde.
4. İstediğin an token'ı iptal edebilirsin; sistem çalışmaya devam eder.

---
*Bu belge yatırım tavsiyesi değildir. Sistem araştırma ve karar destek aracıdır; para riske etme kararları ve sonuçları sahibine aittir.*
