import sys
import cv2 as cv
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, uic, QtGui, QtCore

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        # Load file UI 
        uic.loadUi('main_ui.ui', self)
        
        # Variabel untuk menyimpan State Citra
        self.img_original = None
        self.img_processed = None
        self.img_gray = None
        self.img_thresh = None
        
        # Menghubungkan Tombol - Tombol ke Fungsi
        self.btn_load.clicked.connect(self.load_image)
        self.btn_normalisasi.clicked.connect(self.normalize_image)
        self.btn_grayscale.clicked.connect(self.apply_grayscale)
        self.btn_threshold.clicked.connect(self.apply_threshold)
        self.btn_edge.clicked.connect(self.apply_edge_detection)
        self.btn_deteksi.clicked.connect(self.detect_plate)
        self.btn_segmentasi.clicked.connect(self.segment_and_classify)
        
        self.btn_export_txt.clicked.connect(self.export_txt)
        self.btn_export_csv.clicked.connect(self.export_csv)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pilih Gambar Plat Kendaraan", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        
        if file_name:
            self.img_original = cv.imread(file_name)
            self.img_processed = self.img_original.copy()
            self.img_gray = None
            self.img_thresh = None
            self.display_image(self.img_original, self.label_citra_asli)
            self.display_image(self.img_original, self.label_citra_hasil)
            self.statusBar().showMessage(f"Loaded: {file_name}")

    def normalize_image(self):
        if self.img_original is not None:
            # Operasi Morfologi (Opening) & Aritmatika 
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            img_opening = cv.morphologyEx(self.img_original, cv.MORPH_OPEN, kernel)
            self.img_processed = self.img_original - img_opening
            
            self.display_image(self.img_processed, self.label_citra_hasil)
            self.statusBar().showMessage("Normalisasi Cahaya Selesai")

    def apply_grayscale(self):
        if self.img_processed is not None:
            # Operasi Titik
            # Konversi Gambar ke Gray (Mencegah error jika gambar sudah gray)
            if len(self.img_processed.shape) == 3:
                self.img_gray = cv.cvtColor(self.img_processed, cv.COLOR_BGR2GRAY)
            else:
                self.img_gray = self.img_processed.copy()
            self.img_processed = self.img_gray.copy()
            
            self.display_image(self.img_processed, self.label_citra_hasil)
            self.statusBar().showMessage("Grayscale Selesai")

    def apply_threshold(self):
        if self.img_gray is not None:
            # Operasi Titik dengan Treshold Otsu
            _, self.img_processed = cv.threshold(self.img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            self.img_thresh = self.img_processed.copy()
            
            self.display_image(self.img_processed, self.label_citra_hasil)
            self.statusBar().showMessage("Thresholding Otsu Selesai")

    def apply_edge_detection(self):
        if self.img_processed is not None:
            # Operasi Spasial : Edge Detection menggunakan filter Canny
            # Karena ini merupakan salah satu syarat "Menerapkan Edge Detection using derivative / filter" 
            self.img_processed = cv.Canny(self.img_processed, 100, 200)
            
            self.display_image(self.img_processed, self.label_citra_hasil)
            self.statusBar().showMessage("Deteksi Tepi (Edge Detection) Selesai")

    def detect_plate(self):
        # Di main.py, contours dicari dari hasil Threshold Otsu (bukan dari edge detection).
        # Maka kita prioritas gunakan citra thresholded jika ada.
        img_for_detection = self.img_thresh if self.img_thresh is not None else self.img_processed

        if img_for_detection is not None and self.img_original is not None:
            # Menggunakan Kontur untuk melokalisasi bentuk Persegi Plat
            contours_vehicle, _ = cv.findContours(img_for_detection, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            index_plate_candidate = []
            for contour_vehicle in contours_vehicle:
                x, y, w, h = cv.boundingRect(contour_vehicle)
                aspect_ratio = w / h
                if w >= 200 and aspect_ratio <= 4:
                    index_plate_candidate.append(contour_vehicle)
            
            if len(index_plate_candidate) == 0:
                QtWidgets.QMessageBox.warning(self, "Peringatan", "Plat Nomor Tidak Terdeteksi di gambar ini.")
            else:
                plate_candidate = max(index_plate_candidate, key=cv.contourArea)
                x_plate, y_plate, w_plate, h_plate = cv.boundingRect(plate_candidate)
                
                # Gambar Bounding Box di Gambar Asli (Opsional)
                img_draw = self.img_original.copy()
                cv.rectangle(img_draw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
                self.display_image(img_draw, self.label_citra_asli)
                
                # Potong Plat Nomor & Jadikan Gambar Proses
                self.img_processed = self.img_original[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]
                self.display_image(self.img_processed, self.label_citra_hasil)
                self.statusBar().showMessage("Deteksi & Crop Plat Berhasil.")

    def segment_color(self, hsv_plate):
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        mask_black = cv.inRange(hsv_plate, lower_black, upper_black)
        mask_white = cv.inRange(hsv_plate, lower_white, upper_white)
        mask_yellow = cv.inRange(hsv_plate, lower_yellow, upper_yellow)
        mask_red1 = cv.inRange(hsv_plate, lower_red1, upper_red1)
        mask_red2 = cv.inRange(hsv_plate, lower_red2, upper_red2)
        mask_red = cv.bitwise_or(mask_red1, mask_red2)
        mask_green = cv.inRange(hsv_plate, lower_green, upper_green)

        return mask_black, mask_white, mask_yellow, mask_red, mask_green

    def classify_color(self, mask_black, mask_white, mask_yellow, mask_red, mask_green):
        count_black = cv.countNonZero(mask_black)
        count_white = cv.countNonZero(mask_white)
        count_yellow = cv.countNonZero(mask_yellow)
        count_red = cv.countNonZero(mask_red)
        count_green = cv.countNonZero(mask_green)

        if count_red > count_black and count_red > count_white and count_red > count_yellow and count_red > count_green:
            return "Kendaraan Pemerintah"
        elif count_black > count_white and count_black > count_yellow and count_black > count_red and count_black > count_green:
            return "Kendaraan Pribadi"
        elif count_white > count_black and count_white > count_yellow and count_white > count_red and count_white > count_green:
            return "Kendaraan Pribadi"
        elif count_yellow > count_black and count_yellow > count_white and count_yellow > count_red and count_yellow > count_green:
            return "Kendaraan Umum"
        elif count_green > count_black and count_green > count_white and count_green > count_yellow and count_green > count_red:
            return "Kendaraan Diplomatik"
        else:
            return "Tidak Diketahui"

    def segment_and_classify(self):
        if self.img_processed is not None and len(self.img_processed.shape) == 3:
            # Konversi Gambar crop plat ke hsv
            hsv_plate = cv.cvtColor(self.img_processed, cv.COLOR_BGR2HSV)
            
            # Segmentasi dan klasifikasi (logika dari main.py ditanam ke class)
            mask_black, mask_white, mask_yellow, mask_red, mask_green = self.segment_color(hsv_plate)
            kelas = self.classify_color(mask_black, mask_white, mask_yellow, mask_red, mask_green)
            
            # Tampilkan ke GUI Stringnya
            self.label_hasil_klasifikasi.setText(f"Kategori Kendaraan: {kelas}")
            self.statusBar().showMessage("Proses Klasifikasi Selesai!")
        else:
             QtWidgets.QMessageBox.warning(self, "Informasi", "Pastikan gambar yang di-crop adalah plat (RGB/BGR), agar bisa dikonversi warnanya.")

    def export_txt(self):
        if self.img_processed is not None:
            options = QtWidgets.QFileDialog.Options()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Simpan Data Piksel (TXT)", "", "Text Files (*.txt)", options=options)
            if file_name:
                # Meratakan array (jika 3 layer BGR) lalu di simpan
                img_flat = self.img_processed.reshape(-1, self.img_processed.shape[-1]) if len(self.img_processed.shape) == 3 else self.img_processed
                np.savetxt(file_name, img_flat, fmt='%d', delimiter=',')
                QtWidgets.QMessageBox.information(self, "Berhasil", f"Data matriks piksel berhasil disimpan di {file_name}")

    def export_csv(self):
        if self.img_processed is not None:
            options = QtWidgets.QFileDialog.Options()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Simpan Data Piksel (CSV)", "", "CSV Files (*.csv)", options=options)
            if file_name:
                img_flat = self.img_processed.reshape(-1, self.img_processed.shape[-1]) if len(self.img_processed.shape) == 3 else self.img_processed
                df = pd.DataFrame(img_flat)
                df.to_csv(file_name, index=False, header=False)
                QtWidgets.QMessageBox.information(self, "Berhasil", f"Data matriks piksel (Excel/CSV) berhasil disimpan di {file_name}")

    def display_image(self, img, qlabel):
        """ Fungsi helper untuk memunculkan numpy array image OpenCV ke QLabel PyQt5 """
        if img is None: return
        
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3: # Gambar berwarna BGR
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
                
        # Konversi BGR ke RGB (karena OpenCV defaultnya BGR, sedangkan Qt butuh RGB)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        
        out_image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
        
        # Scaling agar fit di dalam label tapi tetap mempertahankan Aspek Rasio
        pixmap = QtGui.QPixmap.fromImage(out_image)
        pixmap = pixmap.scaled(qlabel.width(), qlabel.height(), QtCore.Qt.KeepAspectRatio)
        qlabel.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
