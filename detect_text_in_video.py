import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import json
import argparse
import os
import sys
from pathlib import Path

# Configuration
TARGET_KEYWORDS = ["stoic", "wisdom", "quotes"]
MIN_CONF = 30
MIN_TEXT_LEN = 2
MERGE_DISTANCE = 60
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"} # Native OpenCV supported formats

# Preprocessing strategies
def preprocess_white_channel(img: np.ndarray) -> np.ndarray:
    white = np.min(img, axis=2)
    _, binary = cv2.threshold(white, 190, 255, cv2.THRESH_BINARY)
    return binary

def preprocess_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_inverted(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(gray)

def preprocess_alpha_channel(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    contrast = np.clip(gray - blur + 128, 0, 255).astype(np.uint8)
    return contrast

STRATEGIES = {
    "white_channel": preprocess_white_channel,
    "grayscale": preprocess_grayscale,
    "inverted": preprocess_inverted,
    "local_contrast": preprocess_alpha_channel,
}

# OCR helpers
def run_ocr(proc_img: np.ndarray) -> list[dict]:
    data = pytesseract.image_to_data(
        proc_img,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3"
    )
    boxes = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < MIN_CONF or len(text) < MIN_TEXT_LEN:
            continue
        boxes.append({
            "text": text,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "conf": conf,
        })
    return boxes

def is_watermark_text(text: str) -> bool:
    t = text.lower().replace(".", "").replace("@", "")
    return any(kw.replace("@", "") in t for kw in TARGET_KEYWORDS)

# Bounding-box merge
def boxes_close(a: dict, b: dict, max_dist: int) -> bool:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    
    horiz_gap = max(0, max(bx1 - ax2, ax1 - bx2))
    vert_gap = max(0, max(by1 - ay2, ay1 - by2))
    return horiz_gap <= max_dist and vert_gap <= max_dist

def merge_boxes(boxes: list[dict]) -> list[dict]:
    if not boxes:
        return []
    parent = list(range(len(boxes)))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    def union(i, j):
        parent[find(i)] = find(j)
        
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_close(boxes[i], boxes[j], MERGE_DISTANCE):
                union(i, j)
                
    groups = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
        
    merged = []
    for group in groups.values():
        x1 = min(b["x"] for b in group)
        y1 = min(b["y"] for b in group)
        x2 = max(b["x"] + b["w"] for b in group)
        y2 = max(b["y"] + b["h"] for b in group)
        text = " ".join(sorted((b["text"] for b in group)))
        conf = max(b["conf"] for b in group)
        
        merged.append({
            "text": text, "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1,
            "x2": x2, "y2": y2, "conf": conf,
        })
    return merged

# Main detection function
def detect_watermarks_in_frame(img: np.ndarray) -> list[dict]:
    h, w = img.shape[:2]
    all_hits = []
    for name, fn in STRATEGIES.items():
        proc = fn(img)
        boxes = run_ocr(proc)
        for b in boxes:
            if is_watermark_text(b["text"]):
                b["strategy"] = name
                all_hits.append(b)
                
    merged = merge_boxes(all_hits)
    results = []
    for m in merged:
        results.append({
            "text": m["text"],
            "x": m["x"], "y": m["y"], "w": m["w"], "h": m["h"],
            "x2": m["x2"], "y2": m["y2"], "conf": m["conf"],
            "x_pct": round(m["x"] / w * 100, 2),
            "y_pct": round(m["y"] / h * 100, 2),
        })
    return results

def process_video(video_path: str, interval_sec: float, fast_mode: bool) -> dict:
    """Extracts frames from a video based on time interval and scans them."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"file": video_path, "error": "Could not open video", "watermarks": []}
    
    # Get FPS to calculate how many frames equal 'interval_sec'
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 # Fallback if OpenCV fails to read the header properly
        
    frame_skip = max(1, int(fps * interval_sec))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_count = 0
    all_watermarks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame based on the calculated time interval
        if frame_count % frame_skip == 0:
            timestamp_sec = round(frame_count / fps, 2)
            hits = detect_watermarks_in_frame(frame)
            
            if hits:
                for hit in hits:
                    hit["timestamp_sec"] = timestamp_sec
                    hit["frame_detected"] = frame_count
                all_watermarks.extend(hits)
                
                # If fast mode is enabled, stop processing this video after the first detection
                if fast_mode:
                    break 
                    
        frame_count += 1
        
    cap.release()
    return {"file": video_path, "video_size": [width, height], "watermarks": all_watermarks}

# Batch processing
def process_directory(folder: str, interval_sec: float, fast_mode: bool, output_file: str) -> dict:
    results = {}
    files = [f for f in Path(folder).iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    
    print(f"Found {len(files)} video(s) in {folder}")
    for f in sorted(files):
        print(f"Processing {f.name} ...", end=" ", flush=True)
        r = process_video(str(f), interval_sec, fast_mode)
        watermarks = r.get("watermarks", [])
        n = len(watermarks)
        print(f"{n} watermark(s) found")
        results[f.name] = watermarks
        
        # Write results after each video to preserve progress
        with open(output_file, "w") as f_out:
            json.dump(results, f_out, indent=2)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Detect watermarks in video datasets.")
    parser.add_argument("directory", help="Directory containing video files")
    parser.add_argument("--output", default="results.json", help="Save JSON results to this file path (default: results.json)")
    parser.add_argument("--interval", type=float, default=2.0, help="Check a frame every N seconds (default: 2.0)")
    parser.add_argument("--fast", action="store_true", help="Stop processing a video after the first watermark is found")
    args = parser.parse_args()

    inp = args.directory
    if not os.path.isdir(inp):
        print(f"Error: '{inp}' is not a valid directory.")
        sys.exit(1)

    all_results = process_directory(inp, args.interval, args.fast, args.output)
    print(f"\nAll processing complete. Final results saved to {args.output}")

if __name__ == "__main__":
    main()