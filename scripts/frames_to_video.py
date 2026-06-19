from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a folder of image frames into an MP4 video."
    )

    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Folder containing video frames, for example data/raw/visdrone-vid/val/sequences/uav0000013_00000_v",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples/visdrone_test_video.mp4"),
        help="Output MP4 video path.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to convert.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))

    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {args.frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    written_frames = 0

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        written_frames += 1

    writer.release()

    print("Video created successfully.")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Frames written: {written_frames}")
    print(f"Output video: {args.output}")


if __name__ == "__main__":
    main()