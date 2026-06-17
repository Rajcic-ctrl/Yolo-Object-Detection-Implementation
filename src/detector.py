from pathlib import Path

import cv2
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.model_path = model_path
        self.confidence = confidence
        self.model = YOLO(model_path)
        
        self.target_class_ids = [0, 1, 2, 3, 5, 7]

    def detect_image(self, image_path: str, output_dir: str = "outputs/pc/images") -> Path:
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
        annotated_image = result.plot()

        output_path = output_folder / f"detected_{image_file.stem}.jpg"
        cv2.imwrite(str(output_path), annotated_image)

        return output_path
    
    def detect_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.target_class_ids,
            save=False,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        return annotated_frame