import argparse
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on webcam or video source")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0, 1) or video path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    logging.info(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    logging.info(f"Starting inference on source: {args.source}")
    model.predict(source=args.source, show=True, conf=args.conf)

if __name__ == "__main__":
    main()
