# Raw DICOM Drop Folder

Bu klasör yerel kullanım içindir.

- Kendi ham `.dcm` / DICOM dosyalarınızı buraya koyun.
- Bu klasör Git'e eklenmez; sadece sizin bilgisayarınızda kalır.
- İşleme adımı `create_dataset.py` ile burada bulunan dosyaları okuyup `dataset/processed/` altına PNG olarak yazar.

## Önerilen akış

1. Ham DICOM dosyalarını buraya kopyalayın.
2. `python create_dataset.py` çalıştırın.
3. Oluşan PNG'leri `dataset/processed/` altında kontrol edin.
4. Eğitim/evaluasyon için `dataset/labels.csv` hazırlayın.

