import cv2 as cv
import numpy as np
import os

def preprocess_plate_for_characters(plate_img):
    if plate_img is None:
        return None
    
    # Konversi ke grayscale
    gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    
    # Blur ringan untuk mengurangi noise (noise reduksi)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Analisis kecerahan untuk memastikan karakter selalu foreground putih
    mean_val = np.mean(gray)
    if mean_val > 127: # Plat warna terang (misal putih, teks hitam)
        # Gunakan Inverse agar teks hitam menjadi putih
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    else: # Plat warna gelap (misal hitam, teks putih)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    # Morphology untuk membersihkan noise kecil dan memperjelas karakter
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    # Tambahkan closing sedikit jika karakter terputus
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel, iterations=1)
    
    return morph

def filter_character_contours(contours, plate_shape):
    plate_h, plate_w = plate_shape[:2]
    filtered_contours = []
    
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv.contourArea(c)
        
        # Heuristik bentuk karakter (huruf/angka) pada plat:
        # 1. Tinggi karakter tidak mungkin selebar plat atau setinggi keseluruhan plat
        #    Tingginya biasanya berkisar 35% sampai 95% dari tinggi crop plat.
        # 2. Rasio aspek (lebar/tinggi). Karakter biasa (0.2 hingga 1.0).
        # 3. Luasan minimum menghindari noise titik.
        
        if h > plate_h * 0.35 and h < plate_h * 0.95:
            if 0.1 < aspect_ratio < 1.1: # Angka 1 sangat ramping (aspect_ratio kecil)
                if area > 40:
                    filtered_contours.append(c)
                    
    return filtered_contours

def sort_character_contours(contours):
    # Urutkan berdasarkan posisi X (kiri ke kanan)
    if not contours:
        return []
    bounding_boxes = [cv.boundingRect(c) for c in contours]
    contours_sorted = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]
    return contours_sorted

def normalize_character_image(image, size=(30, 50)):
    if image is None:
        return None
    
    # Resize langsung ke ukuran tetap (30x50 misalnya) sesuai standar
    resized = cv.resize(image, size, interpolation=cv.INTER_AREA)
    
    # Binarisasi ulang untuk memastikan citra tetap tajam dan bersih
    _, binarized = cv.threshold(resized, 127, 255, cv.THRESH_BINARY)
    return binarized

def segment_characters(plate_img):
    if plate_img is None:
        return [], None
        
    thresh = preprocess_plate_for_characters(plate_img)
    if thresh is None:
        return [], None
        
    # Cari kontur (menggunakan RETR_EXTERNAL agar hanya kontur luar)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    filtered = filter_character_contours(contours, plate_img.shape)
    sorted_contours = sort_character_contours(filtered)
    
    characters = []
    
    for c in sorted_contours:
        x, y, w, h = cv.boundingRect(c)
        
        # Pad area bounding box dengan sedikit margin agar utuh
        pad_y = 2
        pad_x = 2
        y1 = max(0, y - pad_y)
        y2 = min(thresh.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(thresh.shape[1], x + w + pad_x)
        
        char_img = thresh[y1:y2, x1:x2]
        norm_char = normalize_character_image(char_img)
        characters.append((norm_char, (x, y, w, h)))
        
    return characters, thresh

def save_segmented_characters(characters, output_dir="output/characters"):
    if not characters:
        return False
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Bersihkan file karakter sebelumnya
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
                
    for i, (char_img, _) in enumerate(characters):
        filename = os.path.join(output_dir, f"char_{i}.png")
        cv.imwrite(filename, char_img)
        
    return True
