import base64
import json
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv
import numpy as np

from character_segmentation import segment_characters
from template_recognition import recognize_plate_characters
from vehicle_classifier import classify_vehicle_by_number, parse_plate_result

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ALLOWED_SAMPLE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def encode_image(image):
    if image is None:
        return None
    ok, buffer = cv.imencode(".png", image)
    if not ok:
        return None
    payload = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{payload}"


def resize_for_web(image, max_width=960, max_height=540):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image.copy()
    return cv.resize(image, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)


def normalize_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val > 95:
        return image.copy()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return cv.subtract(image, opening)


def segment_color(plate):
    hsv = cv.cvtColor(plate, cv.COLOR_BGR2HSV)
    masks = {
        "hitam": cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45])),
        "putih": cv.inRange(hsv, np.array([0, 0, 185]), np.array([180, 60, 255])),
        "kuning": cv.inRange(hsv, np.array([18, 80, 80]), np.array([42, 255, 255])),
        "hijau": cv.inRange(hsv, np.array([38, 35, 35]), np.array([88, 255, 255])),
    }
    red1 = cv.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255]))
    red2 = cv.inRange(hsv, np.array([160, 80, 70]), np.array([180, 255, 255]))
    masks["merah"] = cv.bitwise_or(red1, red2)
    counts = {name: int(cv.countNonZero(mask)) for name, mask in masks.items()}
    dominant = max(counts, key=counts.get)
    labels = {
        "hitam": "Kendaraan pribadi",
        "putih": "Kendaraan pribadi",
        "kuning": "Kendaraan umum",
        "merah": "Kendaraan pemerintah",
        "hijau": "Kendaraan kawasan khusus",
    }
    return dominant, labels.get(dominant, "Tidak diketahui"), counts


def detect_plate(original, threshold):
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_h, img_w = original.shape[:2]
    candidates = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h == 0:
            continue
        ratio = w / float(h)
        if 2.0 <= ratio <= 4.8 and 120 <= w <= 0.9 * img_w and 18 <= h <= 0.45 * img_h:
            area = cv.contourArea(contour)
            center_penalty = abs((x + w / 2) - img_w / 2) / img_w
            candidates.append((area * (1.2 - min(center_penalty, 1.0)), x, y, w, h))

    if candidates:
        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "fallback": False}

    h, w = original.shape[:2]
    return {"x": int(w * 0.18), "y": int(h * 0.38), "w": int(w * 0.64), "h": int(h * 0.19), "fallback": True}


def draw_plate_overlay(image, plate_box, label):
    output = image.copy()
    x, y, w, h = plate_box["x"], plate_box["y"], plate_box["w"], plate_box["h"]
    cv.rectangle(output, (x, y), (x + w, y + h), (25, 165, 111), max(2, image.shape[1] // 220))
    cv.rectangle(output, (x, max(0, y - 34)), (min(image.shape[1], x + 310), y), (7, 59, 42), -1)
    text = "Estimasi area plat" if plate_box.get("fallback") else label
    cv.putText(output, text, (x + 10, max(20, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def draw_character_overlay(image, plate_box, details):
    output = image.copy()
    x0, y0 = plate_box["x"], plate_box["y"]
    for item in details:
        bbox = item.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        char = item.get("character", "?")
        color = (13, 123, 85) if char.isdigit() else (0, 106, 154)
        cv.rectangle(output, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), color, 2)
        cv.rectangle(output, (x0 + x, max(0, y0 + y - 22)), (x0 + x + 26, y0 + y), (7, 59, 42), -1)
        cv.putText(output, char, (x0 + x + 5, max(14, y0 + y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def crop_by_box(image, box):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    return image[y : y + h, x : x + w]


def process_image(image, flow="classification"):
    source = resize_for_web(image)
    normalized = normalize_image(source)
    gray = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)
    otsu_value, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    edge = cv.Canny(gray, 80, 180)
    plate_box = detect_plate(source, threshold)
    plate_crop = crop_by_box(source, plate_box)
    dominant_color, plate_label, color_counts = segment_color(plate_crop)
    
    # Draw initial overlay
    main_overlay = draw_plate_overlay(source, plate_box, plate_label)

    # Perform character extraction
    characters, char_threshold = segment_characters(plate_crop)
    recognition = recognize_plate_characters(characters, include_letters=True)
    details = recognition.get("details", []) if recognition.get("success") else []
    
    # Overlay logic: only show boxes if extraction flow is active
    if flow == "extraction" and details:
        main_overlay = draw_character_overlay(main_overlay, plate_box, details)

    detected_text = recognition.get("detected_text", "")
    detected_digits = recognition.get("detected_digits", "")
    detected_letters = recognition.get("detected_letters", "")
    
    # Classification logic: Use recognized digits for better accuracy in flow 'classification'
    parsed = parse_plate_result(detected_text, detected_digits, detected_letters)

    char_images = []
    for index, item in enumerate(characters):
        char_img, bbox = item
        detail = details[index] if index < len(details) else {}
        char = detail.get("character", "?")
        char_images.append({
            "image": encode_image(char_img),
            "bbox": bbox,
            "character": char,
            "score": detail.get("score"),
            "type": "angka" if char.isdigit() else "huruf",
        })

    return {
        "success": True,
        "flow": flow,
        "images": {
            "main": encode_image(main_overlay),
            "original": encode_image(source),
            "normalized": encode_image(normalized),
            "gray": encode_image(gray),
            "threshold": encode_image(threshold),
            "edge": encode_image(edge),
            "crop": encode_image(plate_crop),
            "character_threshold": encode_image(char_threshold),
        },
        "analysis": {
            "plate_class": plate_label,
            "dominant_color": dominant_color.capitalize(),
            "color_counts": color_counts,
            "plate_area": f'{plate_box["w"]} x {plate_box["h"]}px' + (" (estimasi)" if plate_box.get("fallback") else ""),
            "otsu_value": int(otsu_value),
            "plate_box": plate_box,
        },
        "recognition": {
            "detected_text": detected_text,
            "detected_digits": detected_digits,
            "detected_letters": detected_letters,
            "character_count": len(characters),
            "characters": char_images,
            "message": recognition.get("message", ""),
        },
        "classification": {
            "vehicle_type": parsed.get("vehicle_type", "-"),
            "wheel_category": parsed.get("wheel_category", "-"),
            "rule_used": parsed.get("rule_used", "-"),
            "region_code": parsed.get("region_code"),
            "registration_number": parsed.get("registration_number"),
        },
    }


def parse_multipart(content_type, body):
    marker = "boundary="
    if marker not in content_type:
        return {}, {}
    boundary = ("--" + content_type.split(marker, 1)[1]).encode()
    fields = {}
    files = {}
    for part in body.split(boundary):
        if b"Content-Disposition" not in part:
            continue
        header, _, value = part.partition(b"\r\n\r\n")
        value = value.rstrip(b"\r\n--")
        disposition = header.decode("utf-8", errors="ignore")
        name = None
        filename = None
        for item in disposition.split(";"):
            item = item.strip()
            if item.startswith("name="):
                name = item.split("=", 1)[1].strip('"')
            if item.startswith("filename="):
                filename = item.split("=", 1)[1].strip('"')
        if not name:
            continue
        if filename:
            files[name] = value
        else:
            fields[name] = value.decode("utf-8", errors="ignore")
    return fields, files


def load_image_from_request(fields, files):
    if files.get("image"):
        payload = np.frombuffer(files["image"], dtype=np.uint8)
        image = cv.imdecode(payload, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("File gambar tidak valid.")
        return image

    sample = fields.get("sample", "")
    if sample:
        safe_name = os.path.basename(sample)
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in ALLOWED_SAMPLE_EXT:
            raise ValueError("Format contoh gambar tidak diizinkan.")
        path = os.path.abspath(os.path.join(DATA_DIR, safe_name))
        if not path.startswith(DATA_DIR) or not os.path.exists(path):
            raise ValueError("Contoh gambar tidak ditemukan.")
        image = cv.imread(path)
        if image is None:
            raise ValueError("Contoh gambar gagal dibaca.")
        return image

    raise ValueError("Tidak ada gambar yang dikirim.")


class WebHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        try:
            if self.path == "/api/process":
                self.handle_process()
            elif self.path == "/api/classify-registration":
                self.handle_registration()
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Endpoint tidak ditemukan.")
        except Exception as exc:
            self.send_json({"success": False, "message": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_process(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        fields, files = parse_multipart(self.headers.get("Content-Type", ""), body)
        image = load_image_from_request(fields, files)
        flow = fields.get("flow", "classification")
        self.send_json(process_image(image, flow))

    def handle_registration(self):
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        
        # Move sanitization logic to Python
        region = re.sub(r'[^A-Z]', '', data.get("region", "").upper())[:2]
        number = re.sub(r'\D', '', data.get("number", ""))[:4]
        
        result = classify_vehicle_by_number(number, region_code=region or None)
        self.send_json({
            "success": True, 
            "classification": result,
            "sanitized": {"region": region, "number": number}
        })


    def send_json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    os.chdir(ROOT_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", 8000), WebHandler)
    print("Web PCD aktif di http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
