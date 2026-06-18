from pathlib import Path

import cv2
import time
from detector import Detector

class VideoProcessor:
    def __init__(self, detector: Detector):
        self.detector = detector

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs/pc/videos"
    ) -> tuple[Path, dict[str, int], dict[str,float]]:
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
        
        processed_frames = 0
        total_inference_time_ms = 0.0
        processing_start_time = time.perf_counter()
        
        max_object_counts: dict[str, int] = {}

        while True:
            success, frame = capture.read()

            if not success:
                break
            
            annotated_frame, frame_counts, total_inference_time_ms = self.detector.detect_frame(frame)
            
            processed_frames += 1
            total_inference_time_ms += total_inference_time_ms
            
            for class_name, count in frame_counts.items():
                previous_max = max_object_counts.get(class_name, 0)

                if count > previous_max:
                    max_object_counts[class_name] = count

            writer.write(annotated_frame)

        capture.release()
        writer.release()
        
        processing_end_time = time.perf_counter()
        total_processing_time = processing_end_time - processing_start_time
        
        average_fps = 0.0
        average_inference_time_ms = 0.0
        
        if processed_frames > 0:
            average_fps = processed_frames / total_processing_time
            average_inference_time_ms = total_inference_time_ms / processed_frames

        performance_metrics = {
            "processed_frames": processed_frames,
            "total_processing_time_sec": total_processing_time,
            "average_fps": average_fps,
            "average_inference_time_ms": average_inference_time_ms
        }

        return output_path, max_object_counts, performance_metrics