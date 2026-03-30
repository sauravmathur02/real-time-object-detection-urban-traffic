import cv2
import argparse
import time
import logging
from detector import UrbanDetector
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_video_writer(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection in Urban Traffic Scenes Using YOLO-Based Architectures")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model weight file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video (optional)")
    parser.add_argument("--save", action="store_true", help="Flag to save the output video")
    
    args = parser.parse_args()

    detector = UrbanDetector(model_path=args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logging.error(f"Could not open video source: {source}")
        return

    out = get_video_writer(cap, args.output) if args.save else None
    if out:
        logging.info(f"Recording output to {args.output}")

    logging.info("Starting YOLOv8 detection. Press 'q' to exit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Video stream ended.")
            break

        results, annotated_frame = detector.detect(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Urban Traffic Object Detection", annotated_frame)
        
        if out:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logging.info("Application closed.")

if __name__ == "__main__":
    main()
