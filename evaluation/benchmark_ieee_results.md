# IEEE Research Paper YOLO Inference Benchmark Results

- **GPU Hardware:** NVIDIA GeForce RTX 4070
- **Image Resolution:** 640x640
- **Batch Size:** 1
- **Precision:** FP32
- **Sample Size:** 200 validation images evaluated per model

## Performance Breakdown

| Model | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) | FPS | Inference-only FPS |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5s | 3.76 | 4.34 | 1.15 | 9.25 | 108.11 | 230.41 |
| YOLOv5m | 3.89 | 5.94 | 1.14 | 10.97 | 91.16 | 168.35 |
| YOLOv7 | 3.29 | 7.01 | 1.48 | 11.78 | 84.89 | 142.65 |
| YOLOv8n | 0.64 | 1.89 | 2.22 | 4.75 | 210.53 | 529.10 |
| YOLOv8s | 0.64 | 3.21 | 2.37 | 6.22 | 160.77 | 311.53 |
| YOLOv8m | 0.64 | 5.64 | 2.22 | 8.50 | 117.65 | 177.30 |
| YOLOv8l | 0.64 | 8.23 | 2.20 | 11.07 | 90.33 | 121.51 |

## Trained Weight File Mappings

| Model | Framework | Weight Path |
| --- | --- | --- |
| YOLOv5s | YOLOv5 | experiments\yolov5\train_yolov5s_v1\weights\best.pt |
| YOLOv5m | YOLOv5 | experiments\yolov5\train_yolov5m\weights\best.pt |
| YOLOv7 | YOLOv7 | experiments\yolov7\train_yolov7\weights\best.pt |
| YOLOv8n | Ultralytics YOLOv8 | runs\detect\train_yolov8n\weights\best.pt |
| YOLOv8s | Ultralytics YOLOv8 | runs\detect\train_yolov8s2\weights\best.pt |
| YOLOv8m | Ultralytics YOLOv8 | runs\detect\train_yolov8m\weights\best.pt |
| YOLOv8l | Ultralytics YOLOv8 | runs\detect\train_fast2\weights\best.pt |
