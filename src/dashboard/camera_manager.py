import cv2
import threading
import time
import logging
from src.detector import UrbanDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CameraManager:
    def __init__(self, source="output.mp4", model_path="yolov8n.pt"):
        self.source = int(source) if str(source).isdigit() else source
        self.model_path = model_path
        self.detector = UrbanDetector(model_path=self.model_path)
        
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"Failed to open source {self.source}")
        
        self._frame = None
        self._annotated_frame = None
        self.is_running = False
        self.thread = None
        
        self.conf_threshold = 0.3
        
        # Stats
        self.current_fps = 0.0
        self.current_detections = {}
        
        self.lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info("Camera Manager background thread started.")

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        logging.info("Camera Manager stopped.")

    def set_conf_threshold(self, conf):
        with self.lock:
            self.conf_threshold = float(conf)

    def get_stats(self):
        with self.lock:
            return {
                "fps": round(self.current_fps, 2),
                "detections": self.current_detections,
                "conf_threshold": self.conf_threshold
            }

    def change_source(self, new_source):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.source = int(new_source) if str(new_source).isdigit() else new_source
            self.cap = cv2.VideoCapture(self.source)

    def _capture_loop(self):
        prev_time = time.time()
        
        while self.is_running:
            if not self.cap.isOpened():
                time.sleep(1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # If video ends, loop it.
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            with self.lock:
                conf = self.conf_threshold
                
            results, annotated_frame = self.detector.detect(frame, conf_threshold=conf)
            
            # Calculate stats
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            # Count detections
            detections = {}
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for cls_id in boxes.cls:
                    cls_name = self.detector.class_names[int(cls_id.item())]
                    detections[cls_name] = detections.get(cls_name, 0) + 1
            
            with self.lock:
                self.current_fps = fps
                self.current_detections = detections
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    self._annotated_frame = buffer.tobytes()
                    
            # Small sleep to prevent maxing out CPU if frame reading is too fast
            time.sleep(0.01)

    def get_frame(self):
        while True:
            with self.lock:
                frame = self._annotated_frame
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + b'NO VIDEO\r\n')
            time.sleep(0.03)  # Approx 30 FPS pushing
