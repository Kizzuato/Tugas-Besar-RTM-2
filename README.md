# Sistem Deteksi Karakter Plat Nomor untuk Klasifikasi Jenis Kendaraan

## Deskripsi
Proyek ini adalah sebuah aplikasi pemrosesan citra digital (PCD) yang dirancang untuk mendeteksi nomor registrasi (angka) pada plat nomor kendaraan. Sistem kemudian mengklasifikasikan jenis kendaraan dan kategori roda berdasarkan rentang angka tersebut. Aplikasi ini mendukung visualisasi berstruktur (GUI) interaktif.

**Perhatian**: Aplikasi ini **TIDAK** menggunakan modul _machine learning_ (seperti TensorFlow/PyTorch) maupun library OCR siap pakai (Tesseract/EasyOCR) untuk mempertahankan kemurnian metode Pemrosesan Citra Digital matematis di tingkat dasar.

## Metode yang Digunakan
1. **Preprocessing & Segmentasi Karakter**: Grayscale, Gaussian Blur, Thresholding (Otsu & Adaptif), Morphological Operations, serta pemotongan (crop) area kontur berdasarkan rasio spasial huruf.
2. **Ekstraksi Ciri (Manual LBP)**: Menggunakan algoritma *Local Binary Pattern* (LBP) mandiri berbasis `numpy` untuk membangun histogram *256-bin* dari tekstur karakter.
3. **Pengenalan Pola (Manual Template Matching)**: Mencocokkan input karakter terhadap *dataset dummy* angka dan huruf (di folder `templates/`) memanfaatkan formula jarak *Chi-Square Histogram*.
4. **Metode Canny**: Tersedia secara eksklusif hanya sebagai metode pembanding *(eksperimen visual)*, bukan *pipeline* klasifikasi.

## Dependencies (Persyaratan)
Aplikasi hanya membutuhkan *library* dasar dan dijamin bebas dari module AI *heavyweight*:
- `opencv-python` (cv2)
- `numpy`
- `pandas`
- `PyQt5`
- `matplotlib`

## Cara Instalasi
1. Pastikan Python 3.x telah ter-install di sistem operasi Anda.
2. *Clone* atau unduh *repository* proyek ini.
3. Buka direktori proyek (contoh: `cd Tugas-Besar-RTM-2`).
4. Install library yang dibutuhkan menggunakan *pip*:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan GUI (Aplikasi Utama)
1. Eksekusi skrip Python GUI:
   ```bash
   python app_gui.py
   ```
2. Klik tombol **Pilih Gambar** dan masukkan foto dari folder `data/`.
3. Klik tombol tahapan citra berurutan: **Normalisasi**, **Grayscale**, **Thresholding**, dan **Deteksi Plat (Crop)**.
4. Klik **Jalankan Deteksi Karakter & Klasifikasi** untuk melihat teks nomor plat yang terbaca beserta tipe kendaraannya. Tombol Canny Edge juga bisa ditekan sebagai pembanding eksperimen.

## Cara Menjalankan Batch Testing
Untuk pengujian masal secara otomatis pada semua citra di folder `data/`:
1. Pastikan ada file gambar berformat JPG/PNG di direktori `data/`.
2. Jalankan perintah:
   ```bash
   python batch_test.py
   ```
3. Hasil pemrosesan akan terangkum rapi di dalam file berformat CSV yang dapat diakses di `output/batch_results.csv`.
