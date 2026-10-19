import cv2
import json
import argparse
import os
import sys
from pathlib import Path

# Configuration
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def match_template(frame, template, threshold):
    """Perform template matching and return location if above threshold."""
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return max_loc, max_val
    return None

def detect_images_in_frame(frame, sample_images, threshold):
    """Detect all sample images in a frame."""
    detections = []
    for img_name, template in sample_images.items():
        res = match_template(frame, template, threshold)
        if res:
            loc, score = res
            h, w = template.shape[:2]
            detections.append({
                "image": img_name,
                "x": loc[0],
                "y": loc[1],
                "w": w,
                "h": h,
                "similarity": round(score, 3),
            })
    return detections

def process_video(video_path, sample_images, interval_sec, threshold):
    """Process a video to detect sample images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"file": video_path, "error": "Could not open video", "detections": []}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_skip = max(1, int(fps * interval_sec))
    frame_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            timestamp_sec = round(frame_count / fps, 2)
            hits = detect_images_in_frame(frame, sample_images, threshold)

            if hits:
                for hit in hits:
                    hit["timestamp_sec"] = timestamp_sec
                    hit["frame_detected"] = frame_count
                all_detections.extend(hits)

        frame_count += 1

    cap.release()
    return {"file": video_path, "detections": all_detections}

def process_directory(video_folder, image_folder, interval_sec, threshold, output_file):
    """Process all videos in directory."""
    results = {}

    # Load sample images
    sample_images = {}
    image_files = [f for f in Path(image_folder).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    for f in image_files:
        img = cv2.imread(str(f))
        if img is not None:
            sample_images[f.name] = img
        else:
            print(f"Warning: Could not load image {f.name}")

    if not sample_images:
        print("Error: No valid sample images found.")
        return {}

    print(f"Loaded {len(sample_images)} sample image(s): {list(sample_images.keys())}")

    # Process videos
    video_files = [f for f in Path(video_folder).iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

    print(f"Found {len(video_files)} video(s) in {video_folder}")
    for f in sorted(video_files):
        print(f"Processing {f.name} ...", end=" ", flush=True)
        r = process_video(str(f), sample_images, interval_sec, threshold)
        detections = r.get("detections", [])
        n = len(detections)
        print(f"{n} detection(s) found")
        results[f.name] = detections

        # Save after each video
        with open(output_file, "w") as f_out:
            json.dump(results, f_out, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(description="Detect sample images in videos.")
    parser.add_argument("video_directory", help="Directory containing video files")
    parser.add_argument("--output", default="image_results.json", help="Save JSON results to this file (default: image_results.json)")
    parser.add_argument("--interval", type=float, default=2.0, help="Check a frame every N seconds (default: 2.0)")
    args = parser.parse_args()

    # Hardcoded values
    image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")  # Change this path to your sample images folder
    threshold = 0.8  # Adjust similarity threshold (0.0-1.0)

    if not os.path.isdir(args.video_directory):
        print(f"Error: Video directory '{args.video_directory}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(image_directory):
        print(f"Error: Image directory '{image_directory}' does not exist.")
        sys.exit(1)

    if not (0.0 <= threshold <= 1.0):
        print("Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)

    all_results = process_directory(args.video_directory, image_directory, args.interval, threshold, args.output)
    print(f"\nAll processing complete. Final results saved to {args.output}")

if __name__ == "__main__":
    main()