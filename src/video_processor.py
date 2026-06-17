from pathlib import Path

import cv2

from detector import Detector

class VideoProcessor:
    def __init__(self, detector: Detector):
        self.detector = detector

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs/pc/videos"
    ) -> Path:
        video_file = Path(video_path)

        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_file}")

        output_folder = Path(output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_file))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_file}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0:
            fps = 25

        output_path = output_folder / f"detected_{video_file.stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            annotated_frame = self.detector.detect_frame(frame)
            writer.write(annotated_frame)

        capture.release()
        writer.release()

        return output_path