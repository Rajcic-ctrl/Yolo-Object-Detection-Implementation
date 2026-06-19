from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


# VisDrone category IDs:
# 0 = ignored regions
# 1 = pedestrian
# 2 = people
# 3 = bicycle
# 4 = car
# 5 = van
# 6 = truck
# 7 = tricycle
# 8 = awning-tricycle
# 9 = bus
# 10 = motor
# 11 = others
#
# Our YOLO classes:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 4 = bus
# 5 = truck
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 1,   # bicycle -> bicycle
    4: 2,   # car -> car
    10: 3,  # motor -> motorcycle
    9: 4,   # bus -> bus
    5: 5,   # van -> truck
    6: 5,   # truck -> truck
}

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone DET dataset annotations to YOLO format."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/visdrone-det"),
        help="Path to raw VisDrone DET dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visdrone_yolo"),
        help="Path where YOLO dataset will be created.",
    )

    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Optional limit for number of train images.",
    )

    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Optional limit for number of validation images.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before conversion.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when limiting number of images.",
    )

    return parser.parse_args()


def create_output_structure(output_dir: Path) -> None:
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    return width, height


def convert_bbox_to_yolo(
    bbox_left: float,
    bbox_top: float,
    bbox_width: float,
    bbox_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_center = (bbox_left + bbox_width / 2) / image_width
    y_center = (bbox_top + bbox_height / 2) / image_height
    normalized_width = bbox_width / image_width
    normalized_height = bbox_height / image_height

    return x_center, y_center, normalized_width, normalized_height


def is_valid_yolo_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> bool:
    return (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    )


def convert_annotation_file(
    annotation_path: Path,
    image_path: Path,
    output_label_path: Path,
) -> int:
    image_width, image_height = get_image_size(image_path)

    converted_lines: list[str] = []

    with annotation_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 6:
                continue

            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_width = float(parts[2])
            bbox_height = float(parts[3])
            object_category = int(parts[5])

            if object_category not in VISDRONE_TO_YOLO:
                continue

            if bbox_width <= 0 or bbox_height <= 0:
                continue

            yolo_class_id = VISDRONE_TO_YOLO[object_category]

            x_center, y_center, normalized_width, normalized_height = convert_bbox_to_yolo(
                bbox_left=bbox_left,
                bbox_top=bbox_top,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                image_width=image_width,
                image_height=image_height,
            )

            if not is_valid_yolo_bbox(
                x_center=x_center,
                y_center=y_center,
                width=normalized_width,
                height=normalized_height,
            ):
                continue

            converted_lines.append(
                f"{yolo_class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

    output_label_path.write_text("\n".join(converted_lines), encoding="utf-8")

    return len(converted_lines)


def select_images(image_paths: list[Path], max_images: int | None, seed: int) -> list[Path]:
    if max_images is None or max_images >= len(image_paths):
        return image_paths

    random.seed(seed)
    return sorted(random.sample(image_paths, max_images))


def convert_split(
    raw_dir: Path,
    output_dir: Path,
    split: str,
    max_images: int | None,
    seed: int,
) -> dict[str, int]:
    source_images_dir = raw_dir / split / "images"
    source_annotations_dir = raw_dir / split / "annotations"

    output_images_dir = output_dir / "images" / split
    output_labels_dir = output_dir / "labels" / split

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {source_images_dir}")

    if not source_annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {source_annotations_dir}")

    image_paths = sorted(source_images_dir.glob("*.jpg"))
    image_paths = select_images(image_paths, max_images=max_images, seed=seed)

    copied_images = 0
    created_labels = 0
    total_objects = 0
    skipped_without_annotation = 0

    for image_path in image_paths:
        annotation_path = source_annotations_dir / f"{image_path.stem}.txt"

        if not annotation_path.exists():
            skipped_without_annotation += 1
            continue

        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, output_image_path)

        object_count = convert_annotation_file(
            annotation_path=annotation_path,
            image_path=image_path,
            output_label_path=output_label_path,
        )

        copied_images += 1
        created_labels += 1
        total_objects += object_count

    return {
        "copied_images": copied_images,
        "created_labels": created_labels,
        "total_objects": total_objects,
        "skipped_without_annotation": skipped_without_annotation,
    }


def write_data_yaml(output_dir: Path) -> None:
    names_yaml = "\n".join(
        f"  {index}: {class_name}"
        for index, class_name in enumerate(CLASS_NAMES)
    )

    data_yaml = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
{names_yaml}
"""

    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    create_output_structure(args.output_dir)

    train_stats = convert_split(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split="train",
        max_images=args.max_train_images,
        seed=args.seed,
    )

    val_stats = convert_split(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split="val",
        max_images=args.max_val_images,
        seed=args.seed,
    )

    write_data_yaml(args.output_dir)

    print("VisDrone conversion completed.")
    print()
    print("Train:")
    print(f"- Images copied: {train_stats['copied_images']}")
    print(f"- Labels created: {train_stats['created_labels']}")
    print(f"- Objects converted: {train_stats['total_objects']}")
    print(f"- Images without annotation: {train_stats['skipped_without_annotation']}")
    print()
    print("Validation:")
    print(f"- Images copied: {val_stats['copied_images']}")
    print(f"- Labels created: {val_stats['created_labels']}")
    print(f"- Objects converted: {val_stats['total_objects']}")
    print(f"- Images without annotation: {val_stats['skipped_without_annotation']}")
    print()
    print(f"YOLO dataset created at: {args.output_dir}")
    print(f"Data config: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()