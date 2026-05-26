# Laporan Perubahan Project (Revisi)

## Latar Belakang Revisi
Project ini awalnya menggunakan metode segmentasi warna pelat nomor (HSV) untuk melakukan klasifikasi jenis kendaraan. Namun, berdasarkan hasil asistensi, metode tersebut dinilai tidak reliabel. Warna pelat nomor dapat dengan mudah dimanipulasi (misal: dicat ulang) sehingga tidak valid digunakan sebagai acuan utama klasifikasi. 

Oleh karena itu, project ini **direvisi secara fundamental**. Fokus sistem saat ini adalah pada pengenalan karakter (terutama **angka nomor registrasi**) pada pelat nomor kendaraan.

## Konsep dan Metode Baru
1. **Dasar Klasifikasi**:
   Klasifikasi jenis kendaraan dan kategori roda kini ditentukan melalui **angka nomor registrasi pelat**, bukan warna.
   Aturan umum yang digunakan:
   - 1 - 1999: Mobil Penumpang (Roda 4)
   - 2000 - 6999: Sepeda Motor (Roda 2)
   - 7000 - 7999: Mobil Bus (Roda 4 atau lebih)
   - 8000 - 8999: Mobil Barang (Roda 4 atau lebih)
   - 9000 - 9999: Kendaraan Khusus (Roda 4 atau lebih)

2. **Metode Utama (Pengolahan Citra Murni Tanpa AI Siap Pakai)**:
   Sistem dilarang menggunakan _library_ OCR siap pakai (seperti EasyOCR atau Tesseract) dan model *deep learning* (TensorFlow/Keras/Torch). Metode yang diaplikasikan adalah:
   - **Segmentasi Karakter**: Dilakukan secara matematis mencari dan memfilter kontur (berdasarkan rasio, lebar, tinggi, luasan).
   - **Ekstraksi Ciri Tekstur**: Menggunakan Local Binary Pattern (LBP) yang dibangun secara **manual** dengan NumPy (tanpa `scikit-image`).
   - **Template Matching**: Karakter dicocokkan terhadap dataset *template* *font* menggunakan pendekatan jarak *Chi-Square Histogram*.

3. **Posisi Metode Canny Edge**:
   Metode **Canny edge** tetap disertakan di dalam aplikasi **hanya sebagai alat pembanding eksperimen**, dan **BUKAN** merupakan _pipeline_ metode klasifikasi utama. Metode utama sistem tetap berbasis *thresholding morphological* (Otsu) untuk deteksi karakter.

4. **Kondisi Antarmuka (GUI)**:
   Aplikasi tetap mempertahankan antarmuka **PyQt5** (`app_gui.py`) yang diadaptasi tanpa perombakan ekstrem. Teks lama yang menyebutkan "Warna" telah diganti dengan "Karakter". Web UI (_Streamlit_ / _Flask_) dapat menjadi opsi tahapan pengembangan lanjut ke depannya setelah model inti ini disempurnakan.

## Panduan Penggunaan
- **Menjalankan Aplikasi (GUI)**: 
  Jalankan `python app_gui.py`, buka citra, jalankan _Thresholding_ -> Deteksi Plat -> Deteksi Karakter & Klasifikasi.
- **Menjalankan Batch Testing**: 
  Jalankan `python batch_test.py` untuk memproses seluruh citra yang ada di folder `data/`. Hasilnya akan di-_export_ otomatis ke file CSV di `output/batch_results.csv`.
