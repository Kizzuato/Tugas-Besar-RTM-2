import cv2 as cv
import numpy as np

def ensure_grayscale(image):
    """
    Memastikan citra dalam format grayscale.
    """
    if image is None:
        raise ValueError("Citra kosong (None).")
        
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image

def extract_lbp_image(gray_image):
    """
    Mengekstrak fitur Local Binary Pattern (LBP) secara manual menggunakan NumPy.
    Setiap piksel diubah nilainya berdasarkan perbandingan dengan 8 tetangganya.
    """
    if gray_image is None:
        raise ValueError("Citra kosong (None).")
        
    # Gunakan padding untuk menangani piksel di tepi batas gambar
    padded = np.pad(gray_image, 1, mode='constant', constant_values=0)
    
    # Ambil matriks tengah (citra asli tanpa padding)
    center = padded[1:-1, 1:-1]
    
    # Matriks LBP yang akan diisi
    lbp = np.zeros_like(center, dtype=np.uint8)
    
    # Bandingkan tetangga (berlawanan/searah jarum jam) dengan piksel tengah
    # Urutan bobot: Top-Left (1), Top (2), Top-Right (4), Right (8), 
    # Bottom-Right (16), Bottom (32), Bottom-Left (64), Left (128)
    
    # Top-Left
    lbp += (padded[0:-2, 0:-2] >= center).astype(np.uint8) * 1
    # Top
    lbp += (padded[0:-2, 1:-1] >= center).astype(np.uint8) * 2
    # Top-Right
    lbp += (padded[0:-2, 2:] >= center).astype(np.uint8) * 4
    # Right
    lbp += (padded[1:-1, 2:] >= center).astype(np.uint8) * 8
    # Bottom-Right
    lbp += (padded[2:, 2:] >= center).astype(np.uint8) * 16
    # Bottom
    lbp += (padded[2:, 1:-1] >= center).astype(np.uint8) * 32
    # Bottom-Left
    lbp += (padded[2:, 0:-2] >= center).astype(np.uint8) * 64
    # Left
    lbp += (padded[1:-1, 0:-2] >= center).astype(np.uint8) * 128
    
    return lbp

def extract_lbp_features(image, bins=256):
    """
    Menghitung citra LBP, membuat histogram, dan menormalisasinya.
    """
    if image is None:
        raise ValueError("Citra tidak valid (None).")
        
    gray = ensure_grayscale(image)
    lbp_img = extract_lbp_image(gray)
    
    # Hitung histogram 256-bin
    hist, _ = np.histogram(lbp_img.ravel(), bins=bins, range=(0, 256))
    
    # Normalisasi histogram agar jumlah semuanya 1
    hist = hist.astype("float")
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= (hist_sum + 1e-10) # 1e-10 untuk mencegah pembagian dengan nol
        
    return hist

def compute_chi_square_distance(hist1, hist2):
    """
    Menghitung Chi-Square distance antara 2 histogram.
    Semakin kecil nilainya, semakin mirip dua histogram tersebut.
    """
    # Tambahkan epsilon kecil untuk mencegah division by zero
    eps = 1e-10
    
    # Rumus Chi-Square distance manual
    dist = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    return float(dist)

def compare_lbp_histogram(hist1, hist2):
    """
    Membandingkan histogram fitur LBP.
    """
    return compute_chi_square_distance(hist1, hist2)

if __name__ == "__main__":
    # Test Mandiri
    print("Mencoba membuat dummy image ukuran 50x50...")
    dummy_img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    print("Ekstraksi LBP image...")
    lbp_img = extract_lbp_image(dummy_img)
    print("Shape LBP Image:", lbp_img.shape)
    
    print("Ekstraksi LBP Histogram...")
    hist = extract_lbp_features(dummy_img)
    print("Shape Histogram:", hist.shape)
    print("Jumlah elemen Histogram (Normalized, seharusnya ~1.0):", np.sum(hist))
    
    # Test Chi-Square
    dummy_img2 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    hist2 = extract_lbp_features(dummy_img2)
    dist = compare_lbp_histogram(hist, hist2)
    print(f"Chi-Square distance antara 2 citra random: {dist:.4f}")
    
    # Identik
    dist_same = compare_lbp_histogram(hist, hist)
    print(f"Chi-Square distance citra dengan dirinya sendiri (harus 0.0): {dist_same:.4f}")
