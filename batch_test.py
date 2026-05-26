import os
import cv2 as cv
import pandas as pd
from plate_processor import process_plate_image

def run_batch_test(data_dir='data', output_file='output/batch_results.csv'):
    if not os.path.exists(data_dir):
        print(f"Error: Folder '{data_dir}' tidak ditemukan. Pastikan Anda memiliki gambar di dalam folder ini.")
        return

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not files:
        print(f"Tidak ada gambar di dalam folder '{data_dir}'.")
        return
        
    results = []
    
    print(f"Memulai batch testing untuk {len(files)} gambar...")
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        img = cv.imread(file_path)
        
        if img is None:
            print(f"[-] {filename}: Gagal membaca gambar.")
            results.append({
                "Nama File": filename,
                "Angka Terdeteksi": "",
                "Huruf Terdeteksi": "",
                "Nomor Registrasi": "",
                "Jenis Kendaraan": "",
                "Kategori Roda": "",
                "Status": "Gagal",
                "Pesan": "Gagal membaca gambar"
            })
            continue
            
        print(f"Memproses {filename}...")
        
        # Asumsi untuk batch testing, karena process_plate_image butuh crop plat,
        # kita lakukan crop sederhana secara otomatis jika belum di-crop.
        # Simulasi deteksi plat sederhana (berdasarkan kontur atau asumsi gambar sudah crop).
        # Agar robust, kita ubah ke grayscale, otsu, lalu cari kontur terbesar.
        # Jika gambar sudah berupa crop plat, dia akan memproses keseluruhan gambar.
        
        # Simulasikan crop sederhana
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        plate_crop = img
        if contours:
            # Filter yang mungkin berupa plat
            plate_candidate = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(plate_candidate)
            if 150 <= w <= (0.85 * img.shape[1]) and h <= (0.85 * img.shape[0]):
                plate_crop = img[y:y+h, x:x+w]
        
        result = process_plate_image(plate_crop)
        
        if result["success"]:
            results.append({
                "Nama File": filename,
                "Angka Terdeteksi": result.get("detected_digits", ""),
                "Huruf Terdeteksi": result.get("detected_letters", ""),
                "Nomor Registrasi": result.get("registration_number", ""),
                "Jenis Kendaraan": result.get("vehicle_type", ""),
                "Kategori Roda": result.get("wheel_category", ""),
                "Status": "Berhasil",
                "Pesan": result.get("message", "")
            })
            print(f"[+] {filename}: {result.get('registration_number', '-')} -> {result.get('vehicle_type', '-')}")
        else:
            results.append({
                "Nama File": filename,
                "Angka Terdeteksi": "",
                "Huruf Terdeteksi": "",
                "Nomor Registrasi": "",
                "Jenis Kendaraan": "",
                "Kategori Roda": "",
                "Status": "Gagal",
                "Pesan": result.get("message", "")
            })
            print(f"[-] {filename}: {result.get('message', 'Gagal memproses.')}")
            
    # Simpan hasil ke CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nBatch testing selesai. Hasil disimpan di '{output_file}'.")

if __name__ == "__main__":
    run_batch_test()
