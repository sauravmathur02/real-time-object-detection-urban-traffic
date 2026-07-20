# IEEE Research Paper YOLO Inference Benchmark Results

- **GPU Hardware:** NVIDIA GeForce RTX 4070
- **Image Resolution:** 640x640
- **Batch Size:** 1
- **Precision:** FP32
- **Sample Size:** 200 validation images evaluated per model

## Performance Breakdown

| Model | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) | FPS | Inference-only FPS |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5s | 3.83 | 4.24 | 1.0 | 9.08 | 110.18 | 235.92 |
| YOLOv7 | 3.11 | 7.62 | 1.24 | 11.97 | 83.54 | 131.28 |
| YOLOv8n | 0.66 | 5.74 | 1.09 | 7.49 | 133.6 | 174.26 |
| YOLOv8s | 0.64 | 4.3 | 0.82 | 5.75 | 173.91 | 232.81 |
| YOLOv8m | 0.65 | 5.46 | 0.82 | 6.92 | 144.46 | 183.3 |
| YOLOv8l | 0.62 | 8.56 | 0.86 | 10.05 | 99.54 | 116.81 |

## Trained Weight File Mappings

| Model | Framework | Weight Path |
| --- | --- | --- |
| YOLOv5s | YOLOV5 | experiments\yolov5\train_yolov5s_v1\weights\best.pt |
| YOLOv7 | YOLOV7 | experiments\yolov7\train_yolov7\weights\best.pt |
| YOLOv8n | Ultralytics YOLOv8 | runs\detect\train_yolov8n\weights\best.pt |
| YOLOv8s | Ultralytics YOLOv8 | runs\detect\train_yolov8s2\weights\best.pt |
| YOLOv8m | Ultralytics YOLOv8 | runs\detect\train_yolov8m\weights\best.pt |
| YOLOv8l | Ultralytics YOLOv8 | runs\detect\train_fast2\weights\best.pt |
