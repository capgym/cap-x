"""Take the first frame of every video in a folder and assemble them into a new video.

Usage:
    python scripts/first_frames_video.py <input_folder> [--output output.mp4] [--fps 2] [--sort name]
"""

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Assemble first frames of videos into a new video")
    parser.add_argument("input_folder", type=str, help="Folder containing input videos")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video path (default: <input_folder>/first_frames.mp4)")
    parser.add_argument("--fps", type=float, default=1, help="FPS of the output video (default: 1)")
    parser.add_argument("--sort", choices=["name", "modified"], default="name", help="Sort order for videos (default: name)")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a directory")
        sys.exit(1)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    videos = [f for f in input_folder.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions]

    # Filter to video_combined.mp4 if present, otherwise take all
    combined = [f for f in videos if f.stem == "video_combined"]
    if combined:
        videos = combined

    if args.sort == "name":
        videos.sort(key=lambda f: f.parent.name)
    else:
        videos.sort(key=lambda f: f.stat().st_mtime)

    if not videos:
        print(f"No videos found in {input_folder}")
        sys.exit(1)

    print(f"Found {len(videos)} videos")

    frames = []
    trial_names = []
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frames.append(frame)
            trial_names.append(video_path.parent.name)
            print(f"  {video_path.relative_to(input_folder)}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  {video_path.name}: FAILED to read first frame, skipping")

    if not frames:
        print("No frames extracted")
        sys.exit(1)

    # Resize all frames to match the first frame's dimensions
    h, w = frames[0].shape[:2]
    for i in range(1, len(frames)):
        if frames[i].shape[:2] != (h, w):
            frames[i] = cv2.resize(frames[i], (w, h))

    output_path = args.output or str(input_folder / "first_frames.mp4")

    with imageio.get_writer(output_path, fps=args.fps, format="FFMPEG", codec="libx264") as writer:
        for i, (frame, name) in enumerate(zip(frames, trial_names), start=1):
            label = f"Trial {i}"
            # Draw text with black outline + white fill for readability
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2 reads BGR, convert to RGB for imageio
            writer.append_data(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    print(f"\nWrote {len(frames)} frames to {output_path} at {args.fps} fps")


if __name__ == "__main__":
    main()
