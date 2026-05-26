import cv2 as cv
import numpy as np

# PENTING: Canny digunakan HANYA sebagai metode pembanding eksperimen, 
# bukan metode utama sistem. Pipeline utama tetap menggunakan 
# segmentasi karakter + LBP + template matching.

def process_canny(image):
    """
    Menghasilkan citra deteksi tepi menggunakan algoritma Canny.
    """
    if image is None:
        return None
        
    # Konversi ke grayscale jika belum
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Blur untuk mereduksi noise
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    # Thresholding bawah dan atas bisa disesuaikan, 
    # biasanya 50 dan 150 adalah titik awal yang baik
    edges = cv.Canny(blur, 50, 150)
    
    return edges

def compare_threshold_vs_canny(plate_img):
    """
    Membandingkan hasil deteksi tepi Canny dengan metode preprocessing 
    thresholding utama dari pipeline segmentasi karakter.
    """
    from character_segmentation import preprocess_plate_for_characters
    
    if plate_img is None:
        return {
            "threshold": None,
            "canny": None,
            "message": "Gambar kosong."
        }
        
    # Hasilkan threshold dari metode utama
    thresh_img = preprocess_plate_for_characters(plate_img)
    
    # Hasilkan canny edge
    canny_img = process_canny(plate_img)
    
    return {
        "threshold": thresh_img,
        "canny": canny_img,
        "message": "Pembandingan berhasil. Ingat, Canny hanya metode eksperimental."
    }

if __name__ == "__main__":
    # Test Canny
    dummy_img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    edges = process_canny(dummy_img)
    print("Shape gambar asli:", dummy_img.shape)
    print("Shape gambar Canny:", edges.shape)
    print("Canny edge berhasil dihasilkan. Metode ini murni sebagai pembanding eksperimen.")
