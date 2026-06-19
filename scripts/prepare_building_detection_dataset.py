from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert building polygon/segmentation labels to YOLO detection bbox labels."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/building_dataset"),
        help="Source building dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/building_detection_yolo"),
        help="Output YOLO detection dataset directory.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before conversion.",
    )

    return parser.parse_args()


def create_output_structure(output_dir: Path) -> None:
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float] | None:
    if len(values) < 4 or len(values) % 2 != 0:
        return None

    xs = values[0::2]
    ys = values[1::2]

    x_min = clamp(min(xs))
    y_min = clamp(min(ys))
    x_max = clamp(max(xs))
    y_max = clamp(max(ys))

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return x_center, y_center, width, height


def convert_label_line(line: str) -> str | None:
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    # We use one final class only: 0 = building
    output_class_id = 0

    numbers = [float(value) for value in parts[1:]]

    # Already YOLO detection format: class x_center y_center width height
    if len(parts) == 5:
        x_center, y_center, width, height = numbers

        if width <= 0 or height <= 0:
            return None

        return (
            f"{output_class_id} "
            f"{clamp(x_center):.6f} "
            f"{clamp(y_center):.6f} "
            f"{clamp(width):.6f} "
            f"{clamp(height):.6f}"
        )

    # YOLO segmentation/polygon format: class x1 y1 x2 y2 ...
    bbox = polygon_to_bbox(numbers)

    if bbox is None:
        return None

    x_center, y_center, width, height = bbox

    return (
        f"{output_class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


def convert_label_file(source_label_path: Path, output_label_path: Path) -> int:
    converted_lines: list[str] = []

    if source_label_path.exists():
        with source_label_path.open("r", encoding="utf-8") as file:
            for line in file:
                converted_line = convert_label_line(line)

                if converted_line is not None:
                    converted_lines.append(converted_line)

    output_label_path.write_text("\n".join(converted_lines), encoding="utf-8")

    return len(converted_lines)


def convert_split(source_dir: Path, output_dir: Path, split: str) -> dict[str, int]:
    source_images_dir = source_dir / "images" / split
    source_labels_dir = source_dir / "labels" / split

    output_images_dir = output_dir / "images" / split
    output_labels_dir = output_dir / "labels" / split

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing source images directory: {source_images_dir}")

    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Missing source labels directory: {source_labels_dir}")

    image_paths = sorted(
        image_path
        for image_path in source_images_dir.iterdir()
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    copied_images = 0
    created_labels = 0
    converted_objects = 0

    for image_path in image_paths:
        source_label_path = source_labels_dir / f"{image_path.stem}.txt"

        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, output_image_path)

        object_count = convert_label_file(
            source_label_path=source_label_path,
            output_label_path=output_label_path,
        )

        copied_images += 1
        created_labels += 1
        converted_objects += object_count

    return {
        "copied_images": copied_images,
        "created_labels": created_labels,
        "converted_objects": converted_objects,
    }


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
  0: building
"""

    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    create_output_structure(args.output_dir)

    train_stats = convert_split(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split="train",
    )

    val_stats = convert_split(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split="val",
    )

    write_data_yaml(args.output_dir)

    print("Building detection dataset conversion completed.")
    print()
    print("Train:")
    print(f"- Images copied: {train_stats['copied_images']}")
    print(f"- Labels created: {train_stats['created_labels']}")
    print(f"- Objects converted: {train_stats['converted_objects']}")
    print()
    print("Validation:")
    print(f"- Images copied: {val_stats['copied_images']}")
    print(f"- Labels created: {val_stats['created_labels']}")
    print(f"- Objects converted: {val_stats['converted_objects']}")
    print()
    print(f"YOLO detection dataset created at: {args.output_dir}")
    print(f"Data config: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()