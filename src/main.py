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


def main():
    args = parse_args()

    detector = Detector(
        model_path=args.model,
        confidence=args.conf
    )

    if args.mode == "image":
        output_path = detector.detect_image(args.source)
        print(f"Detection completed. Output saved to: {output_path}")
        
    elif args.mode == "video":
        video_processor = VideoProcessor(detector)
        output_path = video_processor.process_video(args.source)
        print(f"Video detection completed. Output saved to: {output_path}")


if __name__ == "__main__":
    main()