from pathlib import Path

import cv2
from ultralytics import YOLO

from visualizer import draw_counts_overlay

class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.model_path = model_path
        self.confidence = confidence
        self.model = YOLO(model_path)
        
        self.target_class_ids = [0, 1, 2, 3, 5, 7]

    def detect_image(self, image_path: str, output_dir: str = "outputs/pc/images") -> tuple[Path, dict[str, int]]:
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        output_folder = Path(output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(image_file),
            conf=self.confidence,
            classes=self.target_class_ids,
            save=False,
            verbose=False
        )
        result = results[0]

        object_counts = self._count_objects(result)
        
        
        annotated_image = result.plot()
        annotated_image = draw_counts_overlay(annotated_image, object_counts)

        output_path = output_folder / f"detected_{image_file.stem}.jpg"
        cv2.imwrite(str(output_path), annotated_image)

        

        return output_path, object_counts
    
    
    def detect_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.target_class_ids,
            save=False,
            verbose=False
        )

        result = results[0]
        
        object_counts = self._count_objects(result)
        
        annotated_frame = result.plot()
        annotated_frame = draw_counts_overlay(annotated_frame, object_counts)
        

        return annotated_frame, object_counts
    
    def _count_objects(self, result) -> dict[str, int]:
        counts: dict[str, int] = {}

        if result.boxes is None:
            return counts

        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for class_id in class_ids:
            class_name = result.names[class_id]

            if class_name not in counts:
                counts[class_name] = 0

            counts[class_name] += 1

        return counts