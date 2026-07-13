# Model Inventory

| Model Name | YOLO Version | Dataset | Epochs | Optimizer | Batch | Image Size | Precision | Recall | mAP50 | mAP50-95 | FPS | Params | GFLOPs | Checkpoint Path | Training Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `object_stage1` | YOLOv8l | `bdd100k.yaml` | 30 | auto | 8 | 832 | 0.6443 | 0.4012 | 0.4106 | 0.2256 | N/A | 43.6M* | 165.2* | `runs/detect/object_stage1/weights/best.pt` | Completed |
| `object_stage2` | YOLOv8l | `bdd100k.yaml` | 70 | auto | 8 | 832 | 0.5058 | 0.3834 | 0.3901 | 0.2121 | N/A | 43.6M* | 165.2* | `runs/detect/object_stage2/weights/best.pt` | Completed |
| `object_stage3_strong2` | YOLOv8l | `bdd_no_train.yaml`| 50 | auto | 6 | 960 | 0.6116 | 0.4538 | 0.4732 | 0.2552 | N/A | 43.6M* | 165.2* | `runs/detect/object_stage3_strong2/weights/best.pt` | Completed |
| `object_stage4_riderfix4`| YOLOv8l | `bdd_balanced.yaml`| 50 | auto | 6 | 960 | 0.5998 | 0.4463 | 0.4618 | 0.2572 | N/A | 43.6M* | 165.2* | `runs/detect/object_stage4_riderfix4/weights/best.pt`| Completed |
| `train_fast2` | YOLOv8l | `bdd_balanced.yaml`| 30 | auto | 32 | 512 | 0.7147 | 0.5065 | 0.5524 | 0.3480 | N/A | 43.6M* | 165.2* | `runs/detect/train_fast2/weights/best.pt` | Completed |
| `train_auto_v1` | YOLOv8l | `bdd_balanced.yaml`| 50 | auto | 4 | 640 | 0.5554 | 0.4742 | 0.4828 | 0.2952 | N/A | 43.6M* | 165.2* | `runs/detect/train_auto_v1/weights/best.pt` | Completed |
| `train3` | YOLOv8n | `bdd100k.yaml` | 10 | auto | 16 | 640 | 0.4049 | 0.0019 | 0.2034 | 0.1220 | N/A | 3.2M* | 8.7* | `runs/detect/train3/weights/best.pt` | Completed |
| `validation_yolov5s` | YOLOv5s | `bdd_balanced.yaml`| 1 | SGD | 16 | 640 | 0.5072 | 0.2769 | 0.2726 | 0.1494 | N/A | 7.2M* | 16.5* | `experiments/yolov5/validation_yolov5s/weights/best.pt`| Completed |
| `train_yolov5s_v1` | YOLOv5s | `bdd_balanced.yaml`| *50*| SGD | 16 | 640 | *Pending*| *Pending*| *Pending*| *Pending*| N/A | 7.2M* | 16.5* | `experiments/yolov5/train_yolov5s_v1/weights/last.pt` | **In Progress (Epoch 10)** |

*(Note: FPS, exact Parameter Count, and exact GFLOPs are standard architecture estimates based on Ultralytics releases, as explicit benchmark tools/scripts do not exist in this repository)*

### Missing Models (Planned but not found)
- **YOLOv5m:** Not Implemented
- **YOLOv7x:** Not Implemented
- **YOLOv8s:** Directory exists (`train_yolov8s`), but no weights/logs found.
- **YOLOv8m:** Not Implemented
