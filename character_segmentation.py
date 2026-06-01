import cv2 as cv
import numpy as np
import os

def preprocess_plate_for_characters(plate_img):
    if plate_img is None:
        return None, {}
    
    stages = {}
    
    # 1. Grayscale
    gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    stages["char_gray"] = gray.copy()
    
    # 2. Sharpening
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv.filter2D(gray, -1, kernel_sharpen)
    stages["char_sharpen"] = sharpened.copy()
    
    # 3. Blur
    blur = cv.GaussianBlur(sharpened, (3, 3), 0)
    
    # 4. Otsu Threshold
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    stages["char_thresh_raw"] = thresh.copy()
    
    # 5. Polarity Correction
    white_pixels = cv.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    if white_pixels > total_pixels / 2:
        thresh = cv.bitwise_not(thresh)
    stages["char_polarity"] = thresh.copy()
        
    # 6. Morphology (Gunakan kernel 3x3 agar karakter yang terputus menyambung)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel, iterations=1)
    stages["char_morph"] = morph.copy()
    
    return morph, stages

def filter_character_contours(contours, plate_shape):
    plate_h, plate_w = plate_shape[:2]
    filtered_contours = []
    
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv.contourArea(c)
        
        # HEURISTIK DIGIT (DIPERKETAT KEMBALI)
        # Tinggi minimal 30% untuk menghindari deteksi noise kecil
        if h > plate_h * 0.30 and h < plate_h * 0.95:
            # Aspect ratio fleksibel (0.1 - 0.9) - menghindari objek terlalu lebar/tipis
            if 0.1 <= aspect_ratio <= 0.9: 
                # Area minimal disesuaikan agar tidak menangkap noise kecil
                if area > 100 and w >= 5:
                    # Ambil lebar plat yang lebih relevan
                    x_rel = x / plate_w
                    if 0.05 <= x_rel <= 0.95: 
                        # Filter posisi Y (lebih ketat)
                        if y < plate_h * 0.90:
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
    
    target_w, target_h = size
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    # Binarisasi dulu di resolusi asli
    _, binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Cari bounding box konten (trim whitespace)
    coords = cv.findNonZero(binary)
    if coords is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    bx, by, bw, bh = cv.boundingRect(coords)
    cropped = binary[by:by+bh, bx:bx+bw]

    # Scale agar muat di canvas sambil menjaga rasio aspek (85% area)
    scale = min(target_w / max(bw, 1), target_h / max(bh, 1)) * 0.85
    new_w = max(1, int(bw * scale))
    new_h = max(1, int(bh * scale))
    resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Buat canvas dan center
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    _, result = cv.threshold(canvas, 127, 255, cv.THRESH_BINARY)
    return result

def segment_characters(plate_img):
    if plate_img is None:
        return [], None, {}
        
    thresh, stages = preprocess_plate_for_characters(plate_img)
    if thresh is None:
        return [], None, {}
        
    # Cari kontur (menggunakan RETR_EXTERNAL agar hanya kontur luar)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    filtered = filter_character_contours(contours, plate_img.shape)
    sorted_contours = sort_character_contours(filtered)
    
    characters = []
    
    for c in sorted_contours:
        x, y, w, h = cv.boundingRect(c)
        
        # Padding lebih besar (4px) agar karakter tidak terpotong ketat
        pad = 4
        y1 = max(0, y - pad)
        y2 = min(thresh.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(thresh.shape[1], x + w + pad)
        
        char_img = thresh[y1:y2, x1:x2]

        norm_char = normalize_character_image(char_img)
        characters.append((norm_char, (x, y, w, h)))
        
    return characters, thresh, stages

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
