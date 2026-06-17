import argparse

from detector import Detector

from video_processor import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Drone Object Detection")

    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/test_image.jpg",
        help="Path to input image or video"
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
        choices=["image", "video"],
        type=str,
        default="image",
        help="Detection mode"
    )

    return parser.parse_args()

def print_object_counts(title: str, object_counts: dict[str, int]):
    print(title)

    if not object_counts:
        print("- No objects detected")
        return

    for class_name, count in object_counts.items():
        print(f"- {class_name}: {count}")


def main():
    args = parse_args()

    detector = Detector(
        model_path=args.model,
        confidence=args.conf
    )

    if args.mode == "image":
        output_path, object_counts = detector.detect_image(args.source)
        print(f"Image detection completed. Output saved to: {output_path}")
        print_object_counts("Detected objects:", object_counts)
        
    elif args.mode == "video":
        video_processor = VideoProcessor(detector)
        output_path, max_object_counts = video_processor.process_video(args.source)
        print(f"Video detection completed. Output saved to: {output_path}")
        print_object_counts("Maximum objects detected in a single frame:", max_object_counts)


if __name__ == "__main__":
    main()