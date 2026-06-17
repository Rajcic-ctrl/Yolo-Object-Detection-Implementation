import argparse

from detector import Detector


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Drone Object Detection")

    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/test_image.jpg",
        help="Path to input image"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image"],
        help="Detection mode"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    detector = Detector(
        model_path=args.model,
        confidence=args.conf
    )

    if args.mode == "image":
        output_path = detector.detect_image(args.source)
        print(f"Detection completed. Output saved to: {output_path}")


if __name__ == "__main__":
    main()