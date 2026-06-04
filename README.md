# Sistem Deteksi Karakter Plat Nomor untuk Klasifikasi Jenis Kendaraan

Repository ini merupakan project akhir Praktikum Pengolahan Citra Digital yang berfokus pada pengolahan citra plat nomor kendaraan. Sistem dirancang untuk mendeteksi area plat, melakukan preprocessing, segmentasi karakter, pengenalan angka/huruf, serta mengklasifikasikan jenis kendaraan berdasarkan angka nomor registrasi pada plat.

Project ini dikembangkan sebagai bagian dari tugas akhir Praktikum Pengolahan Citra Digital, Program Studi Informatika.

---

## Anggota Kelompok

| No | Nama | NIM |
|---:|---|---|
| 1 | Zakhwa Aliya Maryam | 15-2024-032 |
| 2 | Zeta Mardhotillah Ronny | 15-2024-047 |
| 3 | Dzakiyya Puteri Aulia | 15-2024-127 |
| 4 | Zahratu Thohiroh Sunanto | 15-2024-198 |

---

## Deskripsi Project

Sistem ini dibuat untuk memproses citra plat nomor kendaraan menggunakan metode Pengolahan Citra Digital. Fokus utama project adalah mendeteksi karakter pada plat, khususnya angka nomor registrasi, kemudian menggunakan angka tersebut untuk menentukan jenis kendaraan dan kategori roda.

Pada rancangan awal, sistem sempat menggunakan pendekatan klasifikasi berdasarkan warna plat. Namun, setelah proses asistensi dan evaluasi, pendekatan tersebut dikembangkan agar klasifikasi utama tidak hanya bergantung pada warna. Hal ini karena warna plat dapat dipengaruhi oleh pencahayaan, kualitas kamera, bayangan, pantulan cahaya, maupun kemungkinan manipulasi visual.

Oleh karena itu, sistem diarahkan untuk menggunakan karakter plat, khususnya angka nomor registrasi, sebagai dasar utama klasifikasi kendaraan. Klasifikasi warna tetap dipertahankan sebagai informasi tambahan atau pendukung visual, tetapi bukan sebagai dasar utama penentuan jenis kendaraan.

Sistem ini tidak menggunakan OCR library siap pakai seperti EasyOCR atau Tesseract, serta tidak menggunakan model machine learning/deep learning seperti TensorFlow, PyTorch, atau Keras. Proses pengenalan karakter dilakukan menggunakan metode manual berbasis pengolahan citra, segmentasi karakter, template matching, dan post-processing pola plat.

---

## Tujuan Project

Tujuan dari project ini adalah:

1. Membangun sistem Pengolahan Citra Digital untuk mendeteksi area plat nomor kendaraan.
2. Melakukan preprocessing citra agar area plat dan karakter menjadi lebih jelas.
3. Melakukan segmentasi karakter angka dan huruf pada plat nomor.
4. Mengenali karakter plat tanpa menggunakan OCR library siap pakai.
5. Mengklasifikasikan jenis kendaraan berdasarkan angka nomor registrasi.
6. Menampilkan hasil berupa angka plat, huruf jika berhasil dikenali, jenis kendaraan, kategori roda, informasi warna tambahan, dan visualisasi tahapan proses.

---

## Ruang Lingkup Project

Project ini berfokus pada:

* Deteksi area plat nomor kendaraan.
* Cropping area plat.
* Preprocessing citra.
* Segmentasi karakter angka/huruf.
* Template matching manual.
* Post-processing hasil pembacaan karakter berdasarkan pola plat Indonesia.
* Klasifikasi kendaraan berdasarkan angka nomor registrasi.
* Klasifikasi warna sebagai informasi tambahan.
* Visualisasi tahapan proses melalui GUI PyQt5 dan web demo sederhana.

Project ini tidak menggunakan OCR library siap pakai seperti EasyOCR atau Tesseract, serta tidak menggunakan model deep learning siap pakai seperti CNN, TensorFlow, PyTorch, atau Keras.

---

## Alur Sistem

Alur utama sistem adalah sebagai berikut:

```text
Input Gambar
→ Deteksi Area Plat
→ Cropping Plat
→ Preprocessing
→ Segmentasi Karakter
→ Pengenalan Angka/Huruf
→ Klasifikasi Jenis Kendaraan
→ Output Hasil
```

Output yang dihasilkan meliputi:

* Gambar asli.
* Hasil normalisasi.
* Hasil grayscale.
* Hasil thresholding.
* Edge visual / Canny sebagai pembanding.
* Crop area plat.
* Hasil segmentasi karakter.
* Angka plat terdeteksi.
* Huruf plat jika berhasil dikenali.
* Jenis kendaraan.
* Kategori roda.
* Informasi warna plat sebagai data tambahan.
* Visualisasi tahapan proses.

---

## Metode yang Digunakan

### 1. Preprocessing

Preprocessing digunakan untuk memperbaiki kualitas citra sebelum diproses lebih lanjut. Tahapan yang digunakan meliputi:

* **Resize**
  Menyesuaikan ukuran citra agar proses komputasi lebih stabil.

* **Normalisasi citra**
  Membantu menstabilkan pencahayaan dan kontras pada citra.

* **Grayscale**
  Mengubah citra berwarna menjadi citra keabuan agar proses pemisahan karakter lebih mudah dilakukan.

* **Gaussian Blur / Smoothing**
  Mengurangi noise kecil pada citra.

* **Thresholding Otsu**
  Mengubah citra grayscale menjadi citra biner dengan nilai ambang otomatis.

* **Polarity Adjustment**
  Menyesuaikan kondisi foreground dan background agar karakter lebih konsisten untuk diproses.

* **Morphological Operation**
  Membersihkan noise, memperjelas bentuk karakter, dan membantu memperbaiki karakter yang terputus.

---

### 2. Deteksi dan Cropping Area Plat

Sistem mendeteksi kandidat area plat berdasarkan hasil preprocessing dan analisis kontur. Kandidat plat dipilih berdasarkan karakteristik bentuk plat, seperti:

* ukuran area,
* rasio lebar dan tinggi,
* bentuk persegi panjang,
* posisi kandidat pada citra,
* kontur yang menyerupai area plat.

Setelah area plat ditemukan, sistem melakukan cropping agar proses selanjutnya hanya berfokus pada bagian plat nomor.

---

### 3. Segmentasi Karakter

Segmentasi karakter dilakukan untuk memisahkan angka dan huruf pada plat nomor. Proses segmentasi meliputi:

1. Mengubah crop plat menjadi grayscale.
2. Menggunakan Gaussian Blur untuk mengurangi noise.
3. Menggunakan thresholding Otsu dengan penyesuaian polarity agar karakter menjadi foreground.
4. Melakukan morphological opening dan closing.
5. Mencari kontur karakter.
6. Memfilter kontur berdasarkan:

   * tinggi karakter,
   * lebar karakter,
   * luas area,
   * rasio lebar dan tinggi,
   * posisi karakter pada plat.
7. Mengurutkan karakter dari kiri ke kanan.
8. Melakukan crop pada setiap karakter.
9. Menormalisasi ukuran karakter menjadi ukuran standar agar dapat dibandingkan dengan template.

Hasil dari tahap ini adalah potongan karakter angka/huruf yang akan digunakan pada tahap pengenalan karakter.

---

### 4. Modul LBP Manual

Project ini memiliki modul **Local Binary Pattern (LBP)** manual pada file `texture_features.py`. Modul ini digunakan untuk mengekstraksi fitur tekstur dari citra dengan membandingkan piksel pusat terhadap piksel-piksel tetangganya, kemudian membentuk histogram 256-bin.

Fungsi LBP yang tersedia meliputi:

* `extract_lbp_image()`
* `extract_lbp_features()`
* `compute_chi_square_distance()`
* `compare_lbp_histogram()`

Pada versi project saat ini, modul LBP berperan sebagai modul ekstraksi fitur tekstur yang tersedia untuk eksperimen atau pengembangan lebih lanjut. Pipeline pengenalan karakter utama pada file `template_recognition.py` menggunakan template matching berbasis korelasi ternormalisasi.

---

### 5. Pengenalan Karakter dengan Template Matching

Pengenalan karakter dilakukan dengan membandingkan karakter hasil segmentasi terhadap template angka dan huruf.

Template yang digunakan berada pada folder:

```text
templates/digits/
templates/letters/
```

Template angka digunakan untuk mengenali karakter 0–9, sedangkan template huruf digunakan untuk mengenali karakter A–Z.

Pada implementasi saat ini, pengenalan karakter dilakukan menggunakan **template matching manual** dengan metode `cv.TM_CCOEFF_NORMED`. Setiap karakter hasil segmentasi dinormalisasi ukurannya, kemudian dibandingkan dengan template. Template dengan nilai korelasi tertinggi dipilih sebagai hasil pengenalan karakter.

Sistem juga memiliki post-processing untuk menyesuaikan hasil pembacaan dengan pola plat nomor Indonesia, yaitu:

```text
[Kode Wilayah] [Angka Registrasi] [Seri Huruf]
```

Post-processing membantu menangani kemungkinan karakter yang bentuknya mirip, misalnya:

* O dan 0,
* I dan 1,
* Z dan 2,
* S dan 5,
* B dan 8.

---

### 6. Canny Edge sebagai Pembanding

Canny edge digunakan sebagai pembanding visual, bukan metode utama klasifikasi.

Canny membantu menampilkan bentuk tepi pada citra sehingga dapat digunakan untuk membandingkan struktur tepi dengan hasil thresholding dan morfologi. Namun, Canny tidak dijadikan metode utama karena Canny menghasilkan tepi, bukan objek karakter yang utuh.

Untuk proses segmentasi dan template matching, sistem membutuhkan karakter dalam bentuk area yang solid. Oleh karena itu, pipeline utama tetap menggunakan thresholding, morfologi, segmentasi kontur, dan template matching manual.

---

### 7. Klasifikasi Warna sebagai Informasi Tambahan

Sistem tetap menyediakan klasifikasi warna sebagai informasi tambahan.

Klasifikasi warna dilakukan untuk mendeteksi warna dominan plat seperti:

* hitam,
* putih,
* kuning,
* merah,
* hijau.

Informasi warna dapat digunakan sebagai pendukung visual, misalnya:

| Warna Plat    | Informasi Umum           |
| ------------- | ------------------------ |
| Hitam / Putih | Kendaraan pribadi        |
| Kuning        | Kendaraan umum           |
| Merah         | Kendaraan pemerintah     |
| Hijau         | Kendaraan kawasan khusus |

Namun, warna tidak digunakan sebagai dasar utama klasifikasi kendaraan karena dapat dipengaruhi pencahayaan, kualitas kamera, atau manipulasi visual.

---

## Proses Klasifikasi Kendaraan

Klasifikasi utama dilakukan berdasarkan angka nomor registrasi pada plat kendaraan.

Setelah angka berhasil dikenali, sistem mencocokkan angka tersebut dengan rentang nomor registrasi kendaraan untuk menentukan jenis kendaraan dan kategori roda.

Aturan umum klasifikasi yang digunakan adalah:

| Rentang Nomor | Jenis Kendaraan  | Kategori Roda     |
| ------------- | ---------------- | ----------------- |
| 1–1999        | Mobil Penumpang  | Roda 4            |
| 2000–6999     | Sepeda Motor     | Roda 2            |
| 7000–7999     | Mobil Bus        | Roda 4 atau lebih |
| 8000–8999     | Mobil Barang     | Roda 4 atau lebih |
| 9000–9999     | Kendaraan Khusus | Roda 4 atau lebih |

Jika kode wilayah `B` terdeteksi, sistem dapat menggunakan aturan khusus wilayah Metro Jaya:

| Rentang Nomor | Jenis Kendaraan                 | Kategori Roda     |
| ------------- | ------------------------------- | ----------------- |
| 1–2999        | Mobil Penumpang                 | Roda 4            |
| 3000–6999     | Sepeda Motor                    | Roda 2            |
| 7000–7999     | Mobil Bus                       | Roda 4 atau lebih |
| 8000–8999     | Mobil Penumpang                 | Roda 4            |
| 9000–9999     | Mobil Barang / Kendaraan Khusus | Roda 4 atau lebih |

Contoh:

```text
Input plat       : D 2534 ABC
Angka terdeteksi : 2534
Rentang angka    : 2000–6999
Jenis kendaraan  : Sepeda Motor
Kategori roda    : Roda 2
```

---

## Fitur Sistem

Fitur utama sistem meliputi:

1. Membaca gambar kendaraan atau plat nomor.
2. Menampilkan citra input.
3. Melakukan normalisasi citra.
4. Mengubah citra ke grayscale.
5. Melakukan thresholding Otsu.
6. Menampilkan edge visual / Canny sebagai pembanding.
7. Mendeteksi dan melakukan crop area plat.
8. Melakukan segmentasi karakter.
9. Mengenali angka dan huruf plat menggunakan template matching manual.
10. Mengklasifikasikan jenis kendaraan berdasarkan angka nomor registrasi.
11. Menampilkan kategori roda kendaraan.
12. Menampilkan klasifikasi warna sebagai informasi tambahan.
13. Menampilkan visualisasi tahapan proses.
14. Menyediakan antarmuka GUI PyQt5.
15. Menyediakan web demo sederhana berbasis Python HTTP Server.

---

## Struktur Folder Project

Struktur project secara umum:

```text
Tugas-Besar-RTM-2/
├── app_gui.py
├── main_ui.ui
├── web_server.py
├── index.html
├── styles.css
├── script.js
├── plate_processor.py
├── character_segmentation.py
├── texture_features.py
├── template_recognition.py
├── vehicle_classifier.py
├── canny_comparison.py
├── batch_test.py
├── data/
├── templates/
│   ├── digits/
│   └── letters/
├── output/
├── requirements.txt
└── README.md
```

Keterangan file utama:

| File                        | Fungsi                                                |
| --------------------------- | ----------------------------------------------------- |
| `app_gui.py`                | Aplikasi GUI berbasis PyQt5                           |
| `main_ui.ui`                | Desain tampilan GUI                                   |
| `web_server.py`             | Server web demo sederhana berbasis Python HTTP Server |
| `index.html`                | Halaman utama web demo                                |
| `styles.css`                | Styling tampilan web                                  |
| `script.js`                 | Logic interaksi pada web                              |
| `plate_processor.py`        | Pipeline utama pemrosesan crop plat                   |
| `character_segmentation.py` | Segmentasi karakter dari area plat                    |
| `texture_features.py`       | Modul LBP manual untuk ekstraksi fitur tekstur        |
| `template_recognition.py`   | Pengenalan karakter berbasis template matching        |
| `vehicle_classifier.py`     | Klasifikasi kendaraan berdasarkan angka registrasi    |
| `canny_comparison.py`       | Canny edge sebagai pembanding visual                  |
| `batch_test.py`             | Pengujian batch pada dataset                          |
| `data/`                     | Folder dataset gambar                                 |
| `templates/`                | Folder template angka dan huruf                       |
| `output/`                   | Folder hasil pengujian atau output sistem             |

---

## Teknologi yang Digunakan

Project ini menggunakan:

* Python
* OpenCV
* NumPy
* Pandas
* PyQt5
* Matplotlib
* HTML
* CSS
* JavaScript
* Python HTTP Server

Library yang digunakan berfokus pada operasi dasar pengolahan citra. Project ini tidak menggunakan OCR library siap pakai atau model AI siap pakai.

---

## Instalasi

Pastikan Python sudah terpasang pada perangkat.

Clone repository:

```bash
git clone https://github.com/Kizzuato/Tugas-Besar-RTM-2.git
cd Tugas-Besar-RTM-2
```

Buat virtual environment:

```bash
python -m venv myenv
```

Aktifkan virtual environment.

Windows PowerShell:

```bash
myenv\Scripts\activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jika `requirements.txt` belum terbaca dengan baik, install manual:

```bash
pip install opencv-python numpy pandas PyQt5 matplotlib
```

---

## Cara Menjalankan GUI PyQt5

Jalankan:

```bash
python app_gui.py
```

Catatan untuk Windows:

Jika muncul error Qt platform plugin, jalankan:

```bash
$env:QT_QPA_PLATFORM="windows"
python app_gui.py
```

---

## Cara Menjalankan Web Demo

Jalankan server:

```bash
python web_server.py
```

Setelah server berjalan, buka browser dan akses:

```text
http://127.0.0.1:8000
```

Web demo digunakan untuk menampilkan pipeline visual pemrosesan citra, seperti:

* input gambar,
* normalisasi,
* grayscale,
* threshold Otsu,
* edge visual,
* crop plat,
* ekstraksi karakter,
* hasil angka/huruf,
* klasifikasi warna tambahan,
* klasifikasi nomor registrasi.

---

## Cara Menjalankan Batch Test

Jalankan:

```bash
python batch_test.py
```

Batch test digunakan untuk menguji beberapa gambar pada folder `data/` dan menyimpan hasil pengujian ke:

```text
output/batch_results.csv
```

---

## Dataset

Dataset yang digunakan merupakan kumpulan citra uji internal kelompok.

Contoh file dataset:

```text
mobil_sipil.png
mobil_sipil2.png
plathitam1.jpg
platkuning5.png
platmerah.jpg
plathijau.png
```

Dataset digunakan untuk menguji:

* deteksi area plat,
* cropping plat,
* preprocessing,
* segmentasi karakter,
* pengenalan angka/huruf,
* klasifikasi kendaraan,
* klasifikasi warna tambahan.

Dataset masih terbatas dan dapat dikembangkan dengan menambahkan variasi gambar lain, seperti variasi sudut, jarak kamera, pencahayaan, blur, dan warna plat.

---

## Rancangan Output

Output sistem meliputi:

```text
Teks Plat          : [hasil deteksi]
Angka Registrasi   : [angka]
Huruf Terdeteksi   : [huruf]
Jenis Kendaraan    : [jenis kendaraan]
Kategori Roda      : [roda]
Warna Dominan      : [warna plat]
Status             : [berhasil/gagal]
```

Contoh output:

```text
Teks Plat          : D2534ABC
Angka Registrasi   : 2534
Huruf Terdeteksi   : DABC
Jenis Kendaraan    : Sepeda Motor
Kategori Roda      : Roda 2
Warna Dominan      : Putih
Status             : Klasifikasi berhasil
```

---

## Evaluasi Sementara

Beberapa hal yang sudah tersedia:

1. Pipeline Pengolahan Citra Digital dari input hingga output klasifikasi.
2. Preprocessing citra.
3. Deteksi dan crop area plat.
4. Segmentasi karakter.
5. Template angka dan huruf.
6. Pengenalan karakter berbasis template matching manual.
7. Modul LBP manual sebagai fitur ekstraksi tekstur yang tersedia untuk eksperimen/pengembangan.
8. Klasifikasi kendaraan berdasarkan angka registrasi.
9. Klasifikasi warna sebagai informasi tambahan.
10. GUI PyQt5 dan web demo.
11. Canny edge sebagai pembanding visual.

Kendala yang masih dapat diperbaiki:

1. Deteksi area plat masih dipengaruhi posisi dan kualitas gambar.
2. Segmentasi karakter masih sensitif terhadap blur, noise, pencahayaan, dan kemiringan plat.
3. Template matching sensitif terhadap variasi font, ketebalan karakter, dan ukuran karakter.
4. Karakter yang bentuknya mirip masih berpotensi tertukar, seperti O dan 0, I dan 1, B dan 8.
5. Dataset uji masih terbatas.
6. Metode pengenalan karakter masih dapat dioptimalkan.

---

## Rencana Pengembangan

Pengembangan berikutnya yang dapat dilakukan:

1. Menambah jumlah dataset dengan variasi gambar yang lebih banyak.
2. Meningkatkan akurasi deteksi area plat.
3. Menyempurnakan segmentasi karakter.
4. Mengoptimalkan template matching.
5. Mengaktifkan atau membandingkan modul LBP secara lebih menyeluruh terhadap template matching.
6. Menambahkan evaluasi kuantitatif yang lebih lengkap.
7. Menyempurnakan tampilan web demo.
8. Menambahkan dokumentasi hasil pengujian.

---

## Catatan Project

Project ini masih berada dalam tahap pengembangan dan penyempurnaan. Fokus utama project adalah menerapkan tahapan Pengolahan Citra Digital secara manual untuk mendeteksi karakter plat dan mengklasifikasikan jenis kendaraan.

Klasifikasi utama dilakukan berdasarkan angka nomor registrasi. Klasifikasi warna tetap dipertahankan sebagai informasi tambahan, tetapi tidak dijadikan dasar utama klasifikasi kendaraan.

---

## Referensi

1. Peraturan Kepolisian Negara Republik Indonesia Nomor 7 Tahun 2021 tentang Registrasi dan Identifikasi Kendaraan Bermotor.
2. Dokumentasi OpenCV — Image Processing, Thresholding, Contour Detection, Morphological Transformations, dan Template Matching.
3. Dokumentasi NumPy — Array Processing dan Numerical Computation.
4. Dokumentasi PyQt5 — GUI Application Development.
5. Materi Praktikum Pengolahan Citra Digital.
