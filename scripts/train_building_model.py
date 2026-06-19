from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the building detection dataset."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/building_dataset/data.yaml"),
        help="Path to YOLO data.yaml file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model to fine-tune.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device. Use 'cpu' or '0' for GPU.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n_building_e1_640",
        help="Training run name.",
    )

    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/pc/yolov8n_building.pt"),
        help="Path where the best trained model will be copied.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = resolve_project_path(args.data)
    output_model_path = resolve_project_path(args.output_model)
    project_dir = resolve_project_path(Path("runs/detect"))

    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": True,
    }

    print("Starting YOLOv8 building model fine-tuning...")
    print(f"Base model: {args.model}")
    print(f"Dataset: {data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Output directory: {project_dir / args.name}")
    print()

    model.train(**train_kwargs)

    if model.trainer is None:
        raise RuntimeError("Training finished, but trainer information is not available.")

    save_dir = Path(model.trainer.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Training completed, but best model was not found: {best_model_path}"
        )

    shutil.copy2(best_model_path, output_model_path)

    print()
    print("Training completed.")
    print(f"Best model: {best_model_path}")
    print(f"Copied to: {output_model_path}")


if __name__ == "__main__":
    main()