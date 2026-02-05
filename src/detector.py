from ultralytics import YOLO
import cv2
import numpy as np

class UrbanDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model weights. Defaults to 'yolov8n.pt' (nano version).
        """
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        
        # BDD100K Classes
        self.class_names = [
            "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
        ]
        # We don't filter by IDs anymore because the model trained on BDD will use these IDs directly 0-9
        self.urban_classes = list(range(10)) 

    def detect(self, frame, conf_threshold=0.3):
        """
        Perform object detection on a single frame.
        
        Args:
            frame (numpy.ndarray): The input image frame.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            results: The raw results object from YOLOv8.
            annotated_frame: The frame with bounding boxes drawn.
        """
        # Run inference
        results = self.model(frame, conf=conf_threshold, classes=self.urban_classes, verbose=False)
        
        # Visualize results on the frame
        annotated_frame = results[0].plot()
        
        return results, annotated_frame
