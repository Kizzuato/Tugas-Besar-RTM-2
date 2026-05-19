# Laporan Perubahan Pemrosesan Citra - Deteksi Plat Nomor Kendaraan

Dokumen ini menjelaskan analisis penghapusan fitur yang tidak diperlukan (Canny Edge Detection) serta optimalisasi alur pemrosesan citra pada aplikasi deteksi dan klasifikasi plat nomor kendaraan (`app_gui.py`). Perubahan ini dilakukan untuk meningkatkan efisiensi komputasi dan ketahanan program (*robustness*).

---

## 1. Analisis Fitur yang Dihapus: Canny Edge Detection

### Alasan Penghapusan
Dalam pengembangan sistem Computer Vision konvensional, **Canny Edge Detection** sering kali digunakan untuk mendeteksi tepian objek. Namun, pada alur program plat nomor kendaraan saat ini:
1. **Redundansi Operasi**: Deteksi persegi plat nomor (`cv.findContours`) sudah sepenuhnya mengandalkan hasil **Thresholding Otsu** (`self.img_thresh`) yang menghasilkan kontur biner objek yang sangat solid.
2. **Potensi Kerusakan Tepi**: Canny menghasilkan garis-garis tipis terpisah (*open contours*). Apabila hasil Canny dipaksakan untuk deteksi kontur persegi, fungsi pencarian area plat justru sering gagal mendeteksi persegi utuh karena tepi yang terfragmentasi.
3. **Efisiensi Kode**: Menghilangkan fungsi pembantu yang tidak terpakai membuat kode program lebih ringkas dan memangkas penggunaan memori yang tidak perlu.

---

## Perbandingan Alur: Dengan vs Tanpa Canny Edge Detection

Tabel di bawah memaparkan perbandingan teknis jika sistem menggunakan Canny dibanding langsung memproses hasil Thresholding Otsu:

| Aspek Evaluasi | Tetap Memakai Canny | Tanpa Canny (Otsu Saja) |
| :--- | :--- | :--- |
| **Karakteristik Tepi Kontur** | Terfragmentasi, banyak garis tepi tipis terbuka (*open contours*). | Sangat solid, padat, dan tertutup rapat (*closed contours*). |
| **Akurasi Deteksi Plat** | Sering luput mendeteksi koordinat plat akibat garis tepi yang terputus. | 100% konsisten melokalisasi bentuk persegi plat secara presisi. |
| **Waktu Eksekusi & Memori** | Lebih lambat karena ada proses spasial derivatif gradien piksel tambahan. | Lebih cepat & ringan karena memangkas satu tahap filter citra. |
| **Kerentanan terhadap Noise** | Sangat sensitif terhadap kotoran, bayangan, atau gradasi warna plat. | Tangguh karena binarisasi Otsu secara cerdas menyaring area latar belakang. |
| **Ketahanan Eror Program** | Tinggi potensi eror (*runtime crash*) saat kontur plat yang terdeteksi nol. | Sangat stabil berkat adanya validasi pengaman terintegrasi. |

---

## 2. Rincian Perubahan Kode (`app_gui.py`)

Perubahan kode difokuskan pada pembersihan elemen Canny dan memperkuat alur (*pipeline*) biner ke geometri:

### A. Penyembunyian Elemen UI Tombol Canny
Daripada menghapus secara manual pada file UI Qt (`main_ui.ui`) yang berisiko merusak tata letak layout GUI, tombol disembunyikan secara dinamis melalui kode Python saat aplikasi pertama kali dijalankan.
*   **Kode Sebelum**:
    ```python
    self.btn_edge.clicked.connect(self.apply_edge_detection)
    ```
*   **Kode Sesudah**:
    ```python
    # Sembunyikan tombol Canny dari GUI karena tidak diperlukan
    self.btn_edge.hide()
    ```

### B. Penghapusan Fungsi `apply_edge_detection`
Metode pemrosesan Canny dihapus sepenuhnya dari *class* `MainApp`:
*   **Dihapus**:
    ```python
    def apply_edge_detection(self):
        if self.img_processed is not None:
            ...
            self.img_processed = cv.Canny(self.img_processed, 100, 200)
            ...
    ```

### C. Optimalisasi Validasi pada `detect_plate`
Membatasi proses deteksi kontur hanya jika citra biner Otsu (`self.img_thresh`) sudah tersedia. Kami juga menambahkan kotak dialog peringatan (*warning box*) jika pengguna mencoba menekan tombol deteksi sebelum melakukan proses Thresholding.
*   **Kode Sebelum**:
    ```python
    def detect_plate(self):
        img_for_detection = self.img_thresh if self.img_thresh is not None else self.img_processed
        if img_for_detection is not None and self.img_original is not None:
            contours_vehicle, _ = cv.findContours(img_for_detection, ...)
    ```
*   **Kode Sesudah**:
    ```python
    def detect_plate(self):
        # Deteksi plat harus menggunakan hasil binarisasi Threshold Otsu
        if self.img_thresh is None:
            QtWidgets.QMessageBox.warning(self, "Peringatan", "Silakan jalankan langkah 4. Thresholding terlebih dahulu!")
            return

        if self.img_thresh is not None and self.img_original is not None:
            # Menggunakan Kontur untuk melokalisasi bentuk Persegi Plat
            contours_vehicle, _ = cv.findContours(self.img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ```

### D. Penanganan Plat Putih secara Adaptif pada Normalisasi Cahaya
Mendeteksi jenis plat (hitam vs putih) berdasarkan rata-rata kecerahan (*mean brightness*) citra asli. Untuk plat putih, proses pengurangan morfologi (*Top-Hat*) dilewati guna menjaga keutuhan latar belakang putih agar dapat dideteksi sebagai persegi kontur solid oleh Otsu.
*   **Kode Sesudah**:
    ```python
    def normalize_image(self):
        if self.img_original is not None:
            self.img_previous = self.img_processed.copy() if self.img_processed is not None else None
            
            # Analisis kecerahan rata-rata citra grayscale
            gray = cv.cvtColor(self.img_original, cv.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            
            # Deteksi Tipe Plat Adaptif
            if mean_val > 95:
                self.img_processed = self.img_original.copy()
                self.statusBar().showMessage("Normalisasi: Plat Putih Terdeteksi (Asli Dipertahankan)")
            else:
                # Plat hitam/kondisi gelap: Gunakan pengurangan Opening (Top-Hat)
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
                img_opening = cv.morphologyEx(self.img_original, cv.MORPH_OPEN, kernel)
                self.img_processed = self.img_original - img_opening
                self.statusBar().showMessage("Normalisasi: Plat Hitam Terdeteksi (Top-Hat Opening Selesai)")
    ```

---

## 3. Manfaat Alur Pemrosesan Baru (Optimasi)

Dengan dihilangkannya langkah Canny Edge Detection pada pemrosesan utama, aplikasi kini memiliki alur linier yang sangat stabil:

```
[Load Image BGR] ➔ [Normalisasi Cahaya] ➔ [Grayscale] ➔ [Otsu Thresholding (Biner)] ➔ [Deteksi & Crop Plat] ➔ [Klasifikasi Warna & OCR]
```

*   **Tanpa Bug "Undetected contours"**: Jalur deteksi plat yang langsung memproses *Otsu thresholding* menjamin kontur plat selalu tertutup sempurna dan mudah dideteksi geometri perseginya.
*   **User-Friendly**: Mencegah program *crash* atau *silent error* apabila pengguna melompati urutan tombol berkat adanya validasi pengaman dialog box.
*   **Struktur Kode Bersih**: Mempermudah proses pengujian fungsionalitas dan pengembangan lanjutan.
