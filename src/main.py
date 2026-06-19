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
    
    parser.add_argument(
        "--no-class-filter",
        action="store_true",
        help="Disable COCO filtering for custom models"
    )

    return parser.parse_args()

def print_object_counts(title: str, object_counts: dict[str, int]):
    print(title)

    if not object_counts:
        print("No objects detected")
        return

    for class_name, count in object_counts.items():
        print(f"{class_name}: {count}")

def print_performance_metrics(metrics: dict):
    print("Performance:")

    print(f"Processed frames: {metrics['processed_frames']}")
    print(f"Total processing time: {metrics['total_processing_time_sec']:.2f} s")
    print(f"Average FPS: {metrics['average_fps']:.2f}")
    print(f"Average inference time: {metrics['average_inference_time_ms']:.2f} ms")


def main():
    args = parse_args()

    detector = Detector(
        model_path=args.model,
        confidence=args.conf,
        use_coco_filter=not args.no_class_filter
    )

    if args.mode == "image":
        output_path, object_counts, inference_time_ms = detector.detect_image(args.source)
        print(f"Image detection completed. Output saved to: {output_path}")
        print_object_counts("Detected objects:", object_counts)
        print(f"Inference time: {inference_time_ms:.2f} ms")
        
    elif args.mode == "video":
        video_processor = VideoProcessor(detector)
        output_path, max_object_counts, performance_metrics = video_processor.process_video(args.source)
        print(f"Video detection completed. Output saved to: {output_path}")
        print_object_counts("Maximum objects detected in a single frame:", max_object_counts)
        print_performance_metrics(performance_metrics)



if __name__ == "__main__":
    main()