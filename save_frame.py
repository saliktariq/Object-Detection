#!/usr/bin/env python3
"""
Save a single frame from a video as an image.

Usage:
    python save_frame.py input.mp4 150
    python save_frame.py input.mp4 150 -o frame_150.png

Frame numbers are 0-based, so frame 0 is the very first frame.
Requires ffmpeg to be installed and available in PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg was not found in PATH.", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a single frame from a video as an image."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "frame",
        type=int,
        help="0-based frame number to save.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to <video_name>_frame_<n>.png",
    )
    return parser.parse_args()


def save_frame(video_path: Path, frame_number: int, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_number})",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    video_path = args.video
    frame_number = args.frame

    if not video_path.exists() or not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if frame_number < 0:
        print("Error: frame number must be 0 or greater.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or video_path.with_name(
        f"{video_path.stem}_frame_{frame_number}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        save_frame(video_path, frame_number, output_path)
        print(f"Saved frame {frame_number} to: {output_path}")
    except subprocess.CalledProcessError:
        print("Error: ffmpeg failed while saving the frame.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
