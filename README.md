# GlycoAI

GlycoAI adalah web app prediksi risiko diabetes berbasis Flask dan Fuzzy Logic Mamdani.

## Struktur

- `app.py` - routing Flask dan API
- `fuzzy_logic.py` - implementasi fuzzy logic
- `templates/` - halaman HTML
- `static/css/` - styling
- `static/js/` - interaksi frontend
- `dataset/` - contoh dataset Pima Indians Diabetes
- `models/` - penyimpanan riwayat prediksi

## Letak Dataset

Jika Anda memiliki dataset Pima Indians Diabetes versi mentah, simpan file CSV-nya di folder `dataset/` dengan nama `diabetes.csv`.

Backend akan melakukan preprocessing otomatis saat membaca dataset:

- mendeteksi missing value seperti kosong, `NaN`, `?`, atau nilai `0` pada kolom numerik tertentu
- mengisi missing value numerik dengan median kolom
- menghitung ringkasan dataset dari data yang sudah dibersihkan

## Cara Menjalankan

1. Buat virtual environment Python.
2. Instal dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
python app.py
```

4. Buka browser ke:

```text
http://127.0.0.1:5000
```

## Catatan

- File `dataset/diabetes.csv` adalah dataset yang dipakai aplikasi dan mengikuti format Pima Indians Diabetes Dataset.
- Riwayat prediksi disimpan di `models/prediction_history.json` agar statistik dashboard tetap hidup saat demo.
- Aplikasi ini untuk edukasi dan tugas akademik, bukan diagnosis medis.
