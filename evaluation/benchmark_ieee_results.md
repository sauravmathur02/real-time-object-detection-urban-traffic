# IEEE Research Paper YOLO Inference Benchmark Results

- **GPU Hardware:** NVIDIA GeForce RTX 4070
- **Image Resolution:** 640x640
- **Batch Size:** 1
- **Precision:** FP32
- **Sample Size:** 200 validation images evaluated per model

## Performance Breakdown

| Model | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) | FPS | Inference-only FPS |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5s | 10.28 | 14.4 | 2.97 | 27.65 | 36.16 | 69.42 |
| YOLOv5m | 8.48 | 16.14 | 3.78 | 28.41 | 35.2 | 61.94 |
| YOLOv7 | 3.27 | 6.19 | 2.03 | 11.49 | 87.0 | 161.43 |
| YOLOv8n | 0.63 | 4.16 | 0.79 | 5.58 | 179.21 | 240.28 |
| YOLOv8s | 0.65 | 4.54 | 0.9 | 6.08 | 164.44 | 220.37 |
| YOLOv8m | 0.65 | 5.82 | 0.85 | 7.33 | 136.45 | 171.76 |
| YOLOv8l | 1.05 | 14.22 | 1.57 | 16.84 | 59.4 | 70.33 |

## Trained Weight File Mappings

| Model | Framework | Weight Path |
| --- | --- | --- |
| YOLOv5s | YOLOV5 | experiments\yolov5\train_yolov5s_v1\weights\best.pt |
| YOLOv5m | YOLOV5 | experiments\yolov5\train_yolov5m\weights\best.pt |
| YOLOv7 | YOLOV7 | experiments\yolov7\train_yolov7\weights\best.pt |
| YOLOv8n | Ultralytics YOLOv8 | runs\detect\train_yolov8n\weights\best.pt |
| YOLOv8s | Ultralytics YOLOv8 | runs\detect\train_yolov8s2\weights\best.pt |
| YOLOv8m | Ultralytics YOLOv8 | runs\detect\train_yolov8m\weights\best.pt |
| YOLOv8l | Ultralytics YOLOv8 | runs\detect\train_fast2\weights\best.pt |
