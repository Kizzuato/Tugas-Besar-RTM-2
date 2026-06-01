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
    Memuat template gambar karakter dan menormalisasinya.
    Mengembalikan dictionary berisi citra template.
    """
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
                    templates[char] = normalize_character_image(img)
                        
    # Load Huruf
    if include_letters and os.path.exists(letters_dir):
        for f in os.listdir(letters_dir):
            if f.endswith('.png'):
                char = f.split('.')[0]
                path = os.path.join(letters_dir, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char] = normalize_character_image(img)
                        
    return templates

def recognize_character(char_img, templates):
    """
    Mengenali satu karakter dengan Template Matching (Correlation Coefficient).
    """
    if char_img is None or not templates:
        return {"character": "-", "score": 0.0, "success": False}
        
    try:
        norm_img = normalize_character_image(char_img)
        if norm_img is None:
            return {"character": "-", "score": 0.0, "success": False}
            
        best_char = "-"
        best_score = -1.0 # Semakin tinggi TM_CCOEFF_NORMED, semakin mirip (max 1.0)
        
        for char, temp_img in templates.items():
            # Gunakan TM_CCOEFF_NORMED untuk korelasi yang dinormalisasi
            res = cv.matchTemplate(norm_img, temp_img, cv.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_char = char
                
        # Ambang batas korelasi (0.0 - 1.0). 0.6 cukup moderat.
        success = best_score > 0.55
        
        return {"character": best_char, "score": float(best_score), "success": success}
        
    except Exception:
        return {"character": "-", "score": 0.0, "success": False}

# Mapping untuk koreksi karakter yang sering tertukar
LETTER_TO_DIGIT_MAP = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
}

DIGIT_TO_LETTER_MAP = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
}

CONFIDENCE_THRESHOLD = 0.55 # TM_CCOEFF_NORMED: makin besar makin mirip

def post_process_plate_format(details):
    """
    Memperbaiki hasil pembacaan berdasarkan pola plat nomor Indonesia yang lebih kuat.
    Format: [Huruf] [1-4 Angka] [1-3 Huruf]
    """
    if not details:
        return ""

    raw_chars = [item["character"] for item in details]
    scores = [item["score"] for item in details]
    
    # Identifikasi transisi dari angka ke huruf
    # Cari indeks pertama yang kemungkinan besar adalah angka
    first_digit_idx = -1
    last_digit_idx = -1
    
    for i, char in enumerate(raw_chars):
        if char.isdigit() or LETTER_TO_DIGIT_MAP.get(char, "").isdigit():
            if first_digit_idx == -1:
                first_digit_idx = i
            last_digit_idx = i

    fixed_chars = []
    for i, item in enumerate(details):
        char = item["character"]
        score = item["score"]
        
        if score < CONFIDENCE_THRESHOLD:
            fixed_chars.append("?")
            continue

        # Aturan Posisi
        if i == 0:
            # Karakter pertama HARUS huruf (Kode Wilayah)
            if char.isdigit():
                char = DIGIT_TO_LETTER_MAP.get(char, char)
        elif first_digit_idx != -1 and last_digit_idx != -1 and first_digit_idx <= i <= last_digit_idx:
            # Bagian tengah yang diidentifikasi sebagai angka
            if char.isalpha():
                char = LETTER_TO_DIGIT_MAP.get(char, char)
        elif i > last_digit_idx and last_digit_idx != -1:
            # Setelah bagian angka berakhir, harus kembali ke huruf
            if char.isdigit():
                char = DIGIT_TO_LETTER_MAP.get(char, char)
        
        fixed_chars.append(char)
        item["character"] = char # Update juga di detail
    
    return "".join(fixed_chars)


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
        
    details = []
    
    # character_images berupa list tuples (char_img, bbox)
    for item in character_images:
        char_img = item[0]
        bbox = item[1] if len(item) > 1 else None
        
        result = recognize_character(char_img, templates)
        
        details.append({
            "bbox": bbox,
            "character": result["character"],
            "score": result["score"],
            "success": result["success"]
        })
        
    # Jalankan Post-Processing Format
    detected_text = post_process_plate_format(details)
    
    detected_digits = "".join([c for c in detected_text if c.isdigit()])
    detected_letters = "".join([c for c in detected_text if c.isalpha()])
    
    success = len(detected_text.replace("?", "")) > 0
    message = "Pengenalan selesai." if success else "Tidak ada satupun karakter yang berhasil dikenali dengan yakin."
    
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
