# YOLO detector module

class Detector:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load_model(self):
        raise NotImplementedError("Model loading will be implemented later.")

    def detect(self, frame):
        raise NotImplementedError("Detection will be implemented later.")
