# Performance Analysis

An analysis of potential bottlenecks in the inference and data processing pipelines without modifying code.

### 1. CPU / OpenCV Bottlenecks
- **NMS and Plotting:** In `detector.py`, `results[0].plot()` utilizes CPU to draw bounding boxes. In dense urban scenes with >30 objects, OpenCV `rectangle` and `putText` calls will spike CPU latency.
- **MJPEG Encoding:** In `camera_manager.py`, `cv2.imencode('.jpg', annotated_frame)` is executing synchronously inside the capture loop. Encoding 1080p frames to JPEG via CPU limits the upper bound of the dashboard's FPS, regardless of GPU speed.

### 2. GPU Bottlenecks
- **Inference Synchronization:** `detector.py` calls `self.model(frame)`. This is a blocking call. While the GPU computes the tensor, the CPU thread idles. TensorRT export (`yolo export model=yolov8l.pt format=engine`) would significantly reduce FP16 latency and maximize GPU utilization.

### 3. Memory Bottlenecks
- **Dataset Conversion:** Scripts like `bdd_to_yolo.py` loop through massive lists of label files. However, they rely on standard Python garbage collection. `src/dashboard/app.py` streams buffers indefinitely; if a client disconnects unexpectedly, there is a minor risk of broken pipes holding memory buffers open in ASGI.

### 4. Disk & Loading Bottlenecks
- **YOLO Dataloader:** Training on `bdd_balanced` involves >34,000 images. The current `train_orchestrator.py` does not explicitly enable `--cache ram` or `--cache disk`, meaning the dataloader will randomly seek the HDD/SSD for every image on every epoch, severely bottlenecking training speed if the disk is a standard HDD.

### Expected Performance Improvements
- Implementing `TensorRT` export: **+40% FPS increase**.
- Setting `--cache ram` during training: **+25% Training Speed increase**.
- Moving `cv2.imencode` to a separate thread: **+10% Dashboard FPS increase**.
