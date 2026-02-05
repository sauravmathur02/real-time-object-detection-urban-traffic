import cv2
import argparse
import time
from detector import UrbanDetector
import os

def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection in Urban Scenes")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model weight file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video (optional)")
    parser.add_argument("--save", action="store_true", help="Flag to save the output video")
    
    args = parser.parse_args()

    # Initialize Detector
    detector = UrbanDetector(model_path=args.model)

    # Initialize Video Capture
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Video Writer setup
    out = None
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0 # Fallback
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Recording to {args.output}...")

    print("Starting detection using YOLOv8... Press 'q' to quit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Run Detection
        results, annotated_frame = detector.detect(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Urban Object Detection", annotated_frame)
        
        # Save output
        if out:
            out.write(annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Cleanup done.")

if __name__ == "__main__":
    main()
