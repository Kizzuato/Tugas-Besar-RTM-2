from character_segmentation import segment_characters, save_segmented_characters
from template_recognition import recognize_plate_characters
from vehicle_classifier import parse_plate_result

def process_plate_image(plate_crop):
    """
    Fungsi utama untuk memproses citra plat nomor yang sudah dicrop.
    Meliputi: segmentasi karakter, pengenalan karakter (template matching), 
    dan klasifikasi kendaraan.
    """
    if plate_crop is None:
        return {"success": False, "message": "Gambar plat kosong."}
        
    # 1. Segmentasi Karakter
    chars, char_thresh, stages = segment_characters(plate_crop)
    if not chars:
        return {"success": False, "message": "Karakter gagal disegmentasi dari plat. Pastikan gambar jelas."}
        
    # Simpan hasil segmentasi untuk debugging/log
    save_segmented_characters(chars, "output/characters")
    
    # 2. Pengenalan Karakter
    recognition = recognize_plate_characters(chars, include_letters=True)
    if not recognition["success"]:
        return {"success": False, "message": "Gagal mengenali karakter pada plat."}
        
    detected_text = recognition["detected_text"]
    detected_digits = recognition["detected_digits"]
    detected_letters = recognition["detected_letters"]
    
    # 3. Klasifikasi Kendaraan Berdasarkan Angka Registrasi
    classification = parse_plate_result(detected_text, detected_digits, detected_letters)
    
    return {
        "success": True,
        "original_crop": plate_crop,
        "threshold": char_thresh,
        "characters": chars,
        "detected_text": detected_text,
        "detected_digits": detected_digits,
        "detected_letters": detected_letters,
        "registration_number": classification["registration_number"],
        "vehicle_type": classification["vehicle_type"],
        "wheel_category": classification["wheel_category"],
        "region_code": classification["region_code"],
        "message": "Klasifikasi berhasil."
    }
