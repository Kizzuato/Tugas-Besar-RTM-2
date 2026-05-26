import cv2 as cv
import numpy as np
import os
import string

from character_segmentation import normalize_character_image
from texture_features import extract_lbp_features, compare_lbp_histogram

def generate_default_templates(base_dir="templates", overwrite=False):
    """
    Membuat gambar template default untuk angka (0-9) dan huruf (A-Z)
    dan menyimpannya di folder templates/digits dan templates/letters.
    Hanya dijalankan jika gambar belum ada atau overwrite=True.
    """
    digits_dir = os.path.join(base_dir, "digits")
    letters_dir = os.path.join(base_dir, "letters")
    
    os.makedirs(digits_dir, exist_ok=True)
    os.makedirs(letters_dir, exist_ok=True)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    size = (30, 50) # Harus sama dengan ukuran normalisasi karakter
    
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    for char in characters:
        is_digit = char.isdigit()
        save_dir = digits_dir if is_digit else letters_dir
        file_path = os.path.join(save_dir, f"{char}.png")
        
        if not overwrite and os.path.exists(file_path):
            continue
            
        # Buat background hitam
        img = np.zeros((size[1], size[0]), dtype=np.uint8)
        
        # Dapatkan ukuran teks untuk menempatkannya di tengah
        text_size = cv.getTextSize(char, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        # Tulis teks putih
        cv.putText(img, char, (text_x, text_y), font, font_scale, 255, thickness)
        
        # Binarisasi untuk memastikan konsistensi
        _, img_thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        
        # Simpan template
        cv.imwrite(file_path, img_thresh)

def load_templates(base_dir="templates", include_letters=True):
    """
    Memuat template gambar karakter, menormalisasinya, 
    dan mengekstrak LBP histogram untuk setiap template.
    """
    # Pastikan template default ada
    generate_default_templates(base_dir, overwrite=False)
    
    digits_dir = os.path.join(base_dir, "digits")
    letters_dir = os.path.join(base_dir, "letters")
    
    templates = {}
    
    # Load Angka
    if os.path.exists(digits_dir):
        for f in os.listdir(digits_dir):
            if f.endswith('.png'):
                char = f.split('.')[0]
                path = os.path.join(digits_dir, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    norm_img = normalize_character_image(img)
                    if norm_img is not None:
                        hist = extract_lbp_features(norm_img)
                        templates[char] = hist
                        
    # Load Huruf (opsional, karena prioritas utama angka)
    if include_letters and os.path.exists(letters_dir):
        for f in os.listdir(letters_dir):
            if f.endswith('.png'):
                char = f.split('.')[0]
                path = os.path.join(letters_dir, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    norm_img = normalize_character_image(img)
                    if norm_img is not None:
                        hist = extract_lbp_features(norm_img)
                        templates[char] = hist
                        
    return templates

def recognize_character(char_img, templates):
    """
    Mengenali satu karakter dengan membandingkan LBP historgramnya 
    terhadap semua template yang ada.
    """
    if char_img is None or not templates:
        return {"character": "-", "score": 999.0, "success": False}
        
    try:
        norm_img = normalize_character_image(char_img)
        if norm_img is None:
            return {"character": "-", "score": 999.0, "success": False}
            
        hist = extract_lbp_features(norm_img)
        
        best_char = "-"
        best_score = float('inf')
        
        # Cari jarak terdekat (Chi-Square)
        for char, temp_hist in templates.items():
            dist = compare_lbp_histogram(hist, temp_hist)
            if dist < best_score:
                best_score = dist
                best_char = char
                
        # Toleransi skor jarak (bisa disesuaikan jika perlu)
        # Untuk implementasi awal kita selalu ambil yang terbaik
        success = best_score < 0.8 # contoh nilai rasional 0.8
        
        return {"character": best_char, "score": best_score, "success": success}
        
    except Exception:
        return {"character": "-", "score": 999.0, "success": False}

def recognize_plate_characters(character_images, include_letters=True):
    """
    Mengenali serangkaian citra karakter dari plat nomor yang telah disegmentasi.
    """
    if not character_images:
        return {
            "success": False,
            "detected_text": "",
            "detected_digits": "",
            "detected_letters": "",
            "details": [],
            "message": "Tidak ada karakter untuk dikenali."
        }
        
    templates = load_templates("templates", include_letters)
    if not templates:
        return {
            "success": False,
            "detected_text": "",
            "detected_digits": "",
            "detected_letters": "",
            "details": [],
            "message": "Gagal memuat template."
        }
        
    detected_text = ""
    detected_digits = ""
    detected_letters = ""
    details = []
    
    # character_images berupa list tuples (char_img, bbox)
    for item in character_images:
        char_img = item[0]
        bbox = item[1] if len(item) > 1 else None
        
        result = recognize_character(char_img, templates)
        char = result["character"]
        
        details.append({
            "bbox": bbox,
            "character": char,
            "score": result["score"],
            "success": result["success"]
        })
        
        if char != "-":
            detected_text += char
            if char.isdigit():
                detected_digits += char
            else:
                detected_letters += char
                
    success = len(detected_text) > 0
    message = "Pengenalan selesai." if success else "Tidak ada satupun karakter yang berhasil dikenali."
    
    return {
        "success": success,
        "detected_text": detected_text,
        "detected_digits": detected_digits,
        "detected_letters": detected_letters,
        "details": details,
        "message": message
    }

if __name__ == "__main__":
    print("Test generate templates...")
    generate_default_templates(overwrite=True)
    print("Templates (0-9, A-Z) generated in templates/digits and templates/letters")
    
    print("Mencoba me-load templates...")
    templates = load_templates()
    print(f"Berhasil load {len(templates)} templates. Contoh key: {list(templates.keys())[:5]}")
    
    # Buat dummy char image (misalnya pakai template "8")
    dummy_img = cv.imread(os.path.join("templates", "digits", "8.png"), cv.IMREAD_GRAYSCALE)
    if dummy_img is not None:
        print("\nMencoba mengenali citra angka 8 dengan metode LBP...")
        result = recognize_character(dummy_img, templates)
        print("Hasil:", result)
