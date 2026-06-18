import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_to_onnx(
    model_path: str = "yolov8n.pt",
    output_path: str = "models/jetson/yolov8n.onnx",
    image_size: int = 416
) -> Path:
    model_file = Path(model_path)
    output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_file))

    exported_path = model.export(
        format="onnx",
        imgsz=image_size
    )

    exported_file = Path(exported_path)

    if not exported_file.exists():
        raise FileNotFoundError(f"Exported ONNX file was not found: {exported_file}")

    if exported_file.resolve() != output_file.resolve():
        shutil.move(str(exported_file), str(output_file))

    print(f"ONNX export completed.")
    print(f"Output model: {output_file}")

    return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to input YOLO .pt model"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/jetson/yolov8n.onnx",
        help="Path where ONNX model will be saved"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size for ONNX export"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        image_size=args.imgsz
    )