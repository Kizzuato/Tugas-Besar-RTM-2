import re

def extract_registration_number(detected_text="", detected_digits=""):
    """
    Mengambil angka nomor registrasi dari hasil karakter.
    Prioritas utama dari detected_digits. Jika kosong, cari digit dari detected_text.
    Mengembalikan angka sebagai string. Jika tidak ada angka, mengembalikan string kosong.
    """
    if detected_digits:
        return detected_digits
        
    if detected_text:
        # Hapus semua karakter yang bukan angka
        digits = re.sub(r'\D', '', detected_text)
        return digits
        
    return ""

def classify_vehicle_by_number(number, region_code=None):
    """
    Menerima angka (string/int) dan mengklasifikasikan jenis kendaraan dan kategori roda
    berdasarkan rentang angka nomor registrasi.
    """
    result = {
        "success": False,
        "registration_number": str(number),
        "vehicle_type": "Tidak Diketahui",
        "wheel_category": "Tidak Diketahui",
        "rule_used": "Tidak Diketahui",
        "message": ""
    }
    
    if not number:
        result["message"] = "Angka registrasi kosong."
        return result
        
    try:
        num_val = int(number)
    except ValueError:
        result["message"] = "Format angka registrasi tidak valid."
        return result
        
    if num_val < 1 or num_val > 9999:
        result["message"] = "Angka di luar rentang valid (1-9999)."
        return result
        
    result["success"] = True
    
    # Aturan Khusus Metro Jaya (B)
    if region_code == "B":
        result["rule_used"] = "Metro Jaya"
        if 1 <= num_val <= 2999:
            result["vehicle_type"] = "Mobil Penumpang"
            result["wheel_category"] = "Roda 4"
        elif 3000 <= num_val <= 6999:
            result["vehicle_type"] = "Sepeda Motor"
            result["wheel_category"] = "Roda 2"
        elif 7000 <= num_val <= 7999:
            result["vehicle_type"] = "Mobil Bus"
            result["wheel_category"] = "Roda 4 atau lebih"
        elif 8000 <= num_val <= 8999:
            result["vehicle_type"] = "Mobil Penumpang"
            result["wheel_category"] = "Roda 4"
        elif 9000 <= num_val <= 9999:
            result["vehicle_type"] = "Mobil Barang / Kendaraan Khusus"
            result["wheel_category"] = "Roda 4 atau lebih"
    else:
        # Aturan Umum
        result["rule_used"] = "Umum"
        if 1 <= num_val <= 1999:
            result["vehicle_type"] = "Mobil Penumpang"
            result["wheel_category"] = "Roda 4"
        elif 2000 <= num_val <= 6999:
            result["vehicle_type"] = "Sepeda Motor"
            result["wheel_category"] = "Roda 2"
        elif 7000 <= num_val <= 7999:
            result["vehicle_type"] = "Mobil Bus"
            result["wheel_category"] = "Roda 4 atau lebih"
        elif 8000 <= num_val <= 8999:
            result["vehicle_type"] = "Mobil Barang"
            result["wheel_category"] = "Roda 4 atau lebih"
        elif 9000 <= num_val <= 9999:
            result["vehicle_type"] = "Kendaraan Khusus"
            result["wheel_category"] = "Roda 4 atau lebih"
            
    result["message"] = "Klasifikasi berhasil."
    return result

def parse_plate_result(detected_text, detected_digits, detected_letters):
    """
    Identifikasi kode wilayah (opsional), ambil angka registrasi,
    dan panggil classify_vehicle_by_number.
    """
    region_code = None
    if detected_text:
        # Coba identifikasi 1 atau 2 huruf kapital pertama sebagai region code (misal B, D, AB)
        match = re.match(r'^([A-Z]{1,2})', detected_text)
        if match:
            region_code = match.group(1)
            
    reg_number = extract_registration_number(detected_text, detected_digits)
    
    classification_result = classify_vehicle_by_number(reg_number, region_code)
    classification_result["region_code"] = region_code
    classification_result["raw_text"] = detected_text
    classification_result["raw_digits"] = detected_digits
    classification_result["raw_letters"] = detected_letters
    
    return classification_result

if __name__ == "__main__":
    print("Testing Aturan Umum:")
    test_cases = [1234, 2534, 7500, 8500, 9500]
    
    for tc in test_cases:
        res = classify_vehicle_by_number(tc)
        print(f"{tc} -> {res['vehicle_type']}, {res['wheel_category']}")
        
    print("\nTesting Aturan Metro Jaya (Region 'B'):")
    for tc in test_cases:
        res = classify_vehicle_by_number(tc, region_code="B")
        print(f"{tc} -> {res['vehicle_type']}, {res['wheel_category']}")
        
    print("\nTesting parse_plate_result (Contoh plat: B 2534 ABC):")
    res = parse_plate_result("B2534ABC", "2534", "BABC")
    print(f"Text asli: {res['raw_text']}")
    print(f"Region   : {res['region_code']}")
    print(f"Reg Num  : {res['registration_number']}")
    print(f"Jenis    : {res['vehicle_type']}")
    print(f"Roda     : {res['wheel_category']}")
    print(f"Aturan   : {res['rule_used']}")
