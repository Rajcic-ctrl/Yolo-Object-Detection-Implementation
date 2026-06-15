from ultralytics import YOLO


def export_to_onnx(model_path="models/pc/yolov8n.pt"):
    model = YOLO(model_path)
    model.export(format="onnx")


if __name__ == "__main__":
    export_to_onnx()
