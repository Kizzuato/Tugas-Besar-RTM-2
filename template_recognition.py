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
    
    # Gunakan beberapa font untuk averaging template yang lebih robust
    font_configs = [
        (cv.FONT_HERSHEY_DUPLEX, 1.3, 3),
        (cv.FONT_HERSHEY_SIMPLEX, 1.4, 4),
    ]
    size = (30, 50) # Harus sama dengan ukuran normalisasi karakter
    
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    for char in characters:
        is_digit = char.isdigit()
        save_dir = digits_dir if is_digit else letters_dir
        file_path = os.path.join(save_dir, f"{char}.png")
        
        if not overwrite and os.path.exists(file_path):
            continue
        
        # Buat canvas yang lebih besar lalu resize agar antialiasing lebih baik
        scale = 4
        big_size = (size[0] * scale, size[1] * scale)
        
        # Average dari beberapa font config
        combined = np.zeros((big_size[1], big_size[0]), dtype=np.float32)
        
        for font, font_scale, thickness in font_configs:
            img = np.zeros((big_size[1], big_size[0]), dtype=np.uint8)
            text_size = cv.getTextSize(char, font, font_scale * scale, thickness)[0]
            text_x = max(0, (img.shape[1] - text_size[0]) // 2)
            text_y = min(img.shape[0] - 2, (img.shape[0] + text_size[1]) // 2)
            cv.putText(img, char, (text_x, text_y), font, font_scale * scale, 255, thickness * scale)
            combined += img.astype(np.float32)
        
        # Normalisasi dan threshold
        combined = np.clip(combined / len(font_configs), 0, 255).astype(np.uint8)
        
        # Resize ke ukuran target (30x50)
        resized = cv.resize(combined, size, interpolation=cv.INTER_AREA)
        
        # Binarisasi
        _, img_thresh = cv.threshold(resized, 60, 255, cv.THRESH_BINARY)
        
        # Simpan template
        cv.imwrite(file_path, img_thresh)

def load_templates(base_dir="templates", include_letters=False):
    """
    Memuat template gambar karakter, fokus pada angka (0-9).
    """
    generate_default_templates(base_dir, overwrite=False)
    
    digits_dir = os.path.join(base_dir, "digits")
    letters_dir = os.path.join(base_dir, "letters")
    
    templates = {}
    
    # Load Angka (Prioritas)
    if os.path.exists(digits_dir):
        for f in os.listdir(digits_dir):
            if f.endswith('.png'):
                char = f.split('.')[0]
                path = os.path.join(digits_dir, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char] = normalize_character_image(img)
                        
    # Load Huruf (Hanya jika include_letters=True)
    if include_letters and os.path.exists(letters_dir):
        for f in os.listdir(letters_dir):
            if f.endswith('.png'):
                char = f.split('.')[0]
                path = os.path.join(letters_dir, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char] = normalize_character_image(img)
                        
    return templates

def extract_character_features(img):
    """
    Ekstraksi fitur struktural dari citra karakter biner 30x50.
    Fitur: density(1) + grid_3x3(9) + h_proj(10) + v_proj(6) + holes(1)
           + h_crossings(5) + v_crossings(3) = 35 dimensi.
    """
    if img is None:
        return np.zeros(35, dtype=np.float32)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(35, dtype=np.float32)

    feats = []

    # 1. Densitas piksel global
    total = h * w
    feats.append(cv.countNonZero(img) / total)

    # 2. Densitas per region 3x3
    for ry in range(3):
        for rx in range(3):
            y1 = ry * h // 3
            y2 = h if ry == 2 else (ry + 1) * h // 3
            x1 = rx * w // 3
            x2 = w if rx == 2 else (rx + 1) * w // 3
            region = img[y1:y2, x1:x2]
            rt = region.shape[0] * region.shape[1]
            feats.append(cv.countNonZero(region) / rt if rt > 0 else 0)

    # 3. Profil proyeksi horizontal (10 bin)
    h_proj = np.sum(img > 0, axis=1).astype(float) / w
    for b in np.array_split(h_proj, 10):
        feats.append(float(np.mean(b)))

    # 4. Profil proyeksi vertikal (6 bin)
    v_proj = np.sum(img > 0, axis=0).astype(float) / h
    for b in np.array_split(v_proj, 6):
        feats.append(float(np.mean(b)))

    # 5. Jumlah lubang (holes) via hierarki kontur
    contours, hierarchy = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    n_holes = 0
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] != -1:
                n_holes += 1
    feats.append(float(n_holes))

    # 6. Transisi horizontal (crossing) pada 5 ketinggian
    for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
        row = img[min(int(h * frac), h - 1), :]
        trans = np.sum(np.abs(np.diff(row.astype(np.int16))) > 127)
        feats.append(float(trans) / 10.0)

    # 7. Transisi vertikal pada 3 posisi lebar
    for frac in [0.3, 0.5, 0.7]:
        col = img[:, min(int(w * frac), w - 1)]
        trans = np.sum(np.abs(np.diff(col.astype(np.int16))) > 127)
        feats.append(float(trans) / 10.0)

    return np.array(feats, dtype=np.float32)


# Cache fitur template agar tidak dihitung ulang setiap karakter
_template_features_cache = {}

def _get_template_features(templates):
    global _template_features_cache
    key = tuple(sorted(templates.keys()))
    if key not in _template_features_cache:
        feats = {}
        for char, temp_img in templates.items():
            f = extract_character_features(temp_img)
            if f is not None:
                feats[char] = f
        _template_features_cache[key] = feats
    return _template_features_cache[key]


def recognize_character(char_img, templates):
    """
    Mengenali satu karakter menggunakan structural feature matching.
    Fitur struktural (density grid, projection profiles, hole count, crossings)
    menangkap TOPOLOGI karakter, bukan posisi piksel individual — sehingga
    robust terhadap perbedaan font antara template dan plat real.
    Skor: 70% cosine similarity fitur + 30% IoU piksel.
    """
    if char_img is None or not templates:
        return {"character": "-", "score": 0.0, "success": False}

    try:
        norm_img = normalize_character_image(char_img)
        if norm_img is None:
            return {"character": "-", "score": 0.0, "success": False}

        input_feats = extract_character_features(norm_img)
        template_feats = _get_template_features(templates)

        best_char = "-"
        best_score = -1.0

        norm_input = np.linalg.norm(input_feats)

        for char, t_feats in template_feats.items():
            # Cosine similarity pada fitur struktural
            norm_t = np.linalg.norm(t_feats)
            if norm_input > 0 and norm_t > 0:
                cosine = float(np.dot(input_feats, t_feats) / (norm_input * norm_t))
            else:
                cosine = 0.0

            # IoU piksel sebagai penguatan
            temp_img = templates[char]
            intersection = cv.countNonZero(cv.bitwise_and(norm_img, temp_img))
            union = cv.countNonZero(cv.bitwise_or(norm_img, temp_img))
            iou = float(intersection) / union if union > 0 else 0.0

            combined = 0.70 * cosine + 0.30 * iou

            if combined > best_score:
                best_score = combined
                best_char = char

        success = best_score > CONFIDENCE_THRESHOLD
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

CONFIDENCE_THRESHOLD = 0.65 # Disesuaikan sedikit lebih longgar untuk menangkap angka 'tipis'

def recognize_plate_characters(character_images, plate_bbox=None, include_letters=False):
    """
    Mengenali serangkaian citra karakter dari plat nomor dengan fokus maksimal pada ANGKA.
    Standar kemiripan ditingkatkan secara drastis (0.70).
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
        
    # Secara default hanya muat template angka (include_letters=False)
    templates = load_templates("templates", include_letters)
    
    details = []
    
    # character_images berupa list tuples (char_img, bbox)
    for item in character_images:
        char_img = item[0]
        bbox = item[1] if len(item) > 1 else None
        
        result = recognize_character(char_img, templates)
        char = result["character"]
        score = result["score"]
        
        # Validasi Skor: Minimal 0.70 korelasi struktural
        if score < CONFIDENCE_THRESHOLD:
            char = "?" 

        # Verifikasi Lubang (Topologi)
        if char != "?":
            norm_char = normalize_character_image(char_img)
            contours_h, hierarchy = cv.findContours(norm_char, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            n_holes = 0
            if hierarchy is not None:
                for i in range(len(hierarchy[0])):
                    if hierarchy[0][i][3] != -1: n_holes += 1
            
            # Angka 8 WAJIB 2 lubang
            if char == "8" and n_holes < 2: char = "?"
            # Angka 0, 6, 9 WAJIB minimal 1 lubang
            if char in ["0", "6", "9"] and n_holes == 0: char = "?"

        details.append({
            "bbox": bbox,
            "character": char,
            "score": score,
            "success": char != "?"
        })
        
    # --- STRATEGI CLUSTERING TENGAH (KHUSUS INDONESIA) ---
    # Kita hanya ingin cluster angka yang berada di tengah.
    digit_candidates = [d for d in details if d["character"].isdigit()]
    
    if len(digit_candidates) > 4 and plate_bbox and plate_bbox.get("w"):
        plate_width_ref = plate_bbox["w"]
        
        # Calculate overall plate center (relative to the cropped plate's width)
        plate_center_x_relative_to_crop = plate_width_ref / 2
        
        for d in digit_candidates:
            if d["bbox"]:
                # Character center relative to the cropped plate
                char_center_x_relative_to_crop = d["bbox"][0] + (d["bbox"][2] / 2)
                
                # Distance of character center to plate center (absolute pixels)
                dist_to_plate_center = abs(char_center_x_relative_to_crop - plate_center_x_relative_to_crop)
                
                # Normalize distance (0.0 - 1.0) relative to half plate width
                # Max distance from center is half plate width
                normalized_dist = dist_to_plate_center / (plate_width_ref / 2) 
                
                # Spatial score: closer to center, higher score.
                # Factor 5.0 penalizes heavily if far from center
                spatial_score = 1.0 / (1.0 + normalized_dist * 5.0) 
                
                # Combined score: Confidence (0.4) + Spatial Centrality (0.6)
                d["combined_priority"] = (d["score"] * 0.4) + (spatial_score * 0.6)
            else:
                # If no bbox, assign a lower priority based only on confidence
                d["combined_priority"] = d["score"] * 0.4 
        
        # Sort by combined priority and take the top 4
        digit_candidates.sort(key=lambda x: x.get("combined_priority", 0), reverse=True)
        top_4_digits = digit_candidates[:4]
    else:
        # If <= 4 candidates or no plate_bbox, use all digit candidates
        top_4_digits = digit_candidates

    # Urutkan kembali dari kiri ke kanan (Posisi X)
    top_4_digits.sort(key=lambda x: x["bbox"][0] if x["bbox"] else 0)
    
    detected_digits = "".join([d["character"] for d in top_4_digits])
    detected_text = detected_digits
    
    success = len(detected_digits) > 0
    message = f"Angka terdeteksi (Top 4): {detected_digits}" if success else "Tidak ada angka valid ditemukan."
    
    return {
        "success": success,
        "detected_text": detected_text,
        "detected_digits": detected_digits,
        "detected_letters": "",
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
