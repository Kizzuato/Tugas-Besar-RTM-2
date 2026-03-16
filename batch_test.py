import cv2 as cv
import numpy as np
import os

def segment_color(hsv_plate):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask_black = cv.inRange(hsv_plate, lower_black, upper_black)
    mask_white = cv.inRange(hsv_plate, lower_white, upper_white)
    mask_yellow = cv.inRange(hsv_plate, lower_yellow, upper_yellow)
    mask_red1 = cv.inRange(hsv_plate, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_plate, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_green = cv.inRange(hsv_plate, lower_green, upper_green)

    return mask_black, mask_white, mask_yellow, mask_red, mask_green

def classify_color(mask_black, mask_white, mask_yellow, mask_red, mask_green):
    count_black = cv.countNonZero(mask_black)
    count_white = cv.countNonZero(mask_white)
    count_yellow = cv.countNonZero(mask_yellow)
    count_red = cv.countNonZero(mask_red)
    count_green = cv.countNonZero(mask_green)

    if count_red > count_black and count_red > count_white and count_red > count_yellow and count_red > count_green:
        return "Kendaraan Pemerintah"
    elif count_black > count_white and count_black > count_yellow and count_black > count_red and count_black > count_green:
        return "Kendaraan Pribadi"
    elif count_white > count_black and count_white > count_yellow and count_white > count_red and count_white > count_green:
        return "Kendaraan Pribadi"
    elif count_yellow > count_black and count_yellow > count_white and count_yellow > count_red and count_yellow > count_green:
        return "Kendaraan Umum"
    elif count_green > count_black and count_green > count_white and count_green > count_yellow and count_green > count_red:
        return "Kendaraan Diplomatik"
    else:
        return "Tidak Diketahui"

def test_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        return "Error: Image not found"

    # Pre-processing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_norm = img - img_opening
    img_gray = cv.cvtColor(img_norm, cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Detection
    contours_vehicle, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    index_plate_candidate = []
    for contour_vehicle in contours_vehicle:
        x, y, w, h = cv.boundingRect(contour_vehicle)
        aspect_ratio = w / h
        if w >= 200 and aspect_ratio <= 4:
            index_plate_candidate.append(contour_vehicle)

    if len(index_plate_candidate) == 0:
        return "Plat nomor tidak ditemukan"
    
    plate_candidate = max(index_plate_candidate, key=cv.contourArea)
    x_p, y_p, w_p, h_p = cv.boundingRect(plate_candidate)
    cropped_img = img[y_p:y_p + h_p, x_p:x_p + w_p]

    # Classification
    hsv_plate = cv.cvtColor(cropped_img, cv.COLOR_BGR2HSV)
    masks = segment_color(hsv_plate)
    result = classify_color(*masks)
    
    return result

data_dir = '/home/kizzu/Kuliah/PCD/Tubes/projectPCD/data'
files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
files.sort()

print(f"{'Filename':<20} | {'Result':<25}")
print("-" * 50)
for filename in files:
    path = os.path.join(data_dir, filename)
    res = test_image(path)
    print(f"{filename:<20} | {res:<25}")
