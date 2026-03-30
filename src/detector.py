import logging
from ultralytics import YOLO

class UrbanDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLO object detection model for urban scenes.
        """
        logging.info(f"Loading weights from {model_path}")
        self.model = YOLO(model_path)
        self.class_names = [
            "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
        ]
        self.urban_classes = list(range(10)) 

    def detect(self, frame, conf_threshold=0.3):
        """
        Applies YOLO object detection on a provided image frame.
        """
        results = self.model(frame, conf=conf_threshold, classes=self.urban_classes, verbose=False)
        annotated_frame = results[0].plot()
        return results, annotated_frame
