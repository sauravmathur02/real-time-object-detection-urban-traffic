# Training History

The following table reconstructs the training history logs found in the repository metrics (`results.csv`, `args.yaml`, `opt.yaml`). 

| Run Identifier | Duration (s) | GPU/CPU | Start / End Date | Source | Training Command (Example) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`object_stage1`** | 4,078 | GPU 0 | Feb 28, 2026 | `results.csv` | `yolo detect train data=data1/bdd100k.yaml model=yolov8l.pt epochs=30 imgsz=832 batch=8 device=0` |
| **`object_stage2`** | 9,219 | GPU 0 | Feb 28, 2026 | `results.csv` | `yolo detect train data=data1/bdd100k.yaml model=runs/detect/object_stage1/weights/best.pt epochs=70 imgsz=832 batch=8 device=0` |
| **`object_stage3_strong2`**| 8,354 | GPU 0 | Mar 02, 2026 | `results.csv` | `yolo detect train data=data1/bdd_no_train.yaml model=runs/detect/object_stage2/weights/best.pt epochs=50 imgsz=960 batch=6 device=0` |
| **`object_stage4_riderfix4`**| 8,672 | GPU 0 | Mar 05, 2026 | `results.csv` | `yolo detect train data=data1/bdd_balanced.yaml model=runs/detect/object_stage3_strong2/weights/best.pt epochs=50 imgsz=960 batch=6 device=0` |
| **`train_fast2`** | 48,078 | GPU 0 | Mar 10, 2026 | `results.csv` | `yolo detect train data=data1/bdd_balanced.yaml model=yolov8l.pt epochs=30 imgsz=512 batch=32 device=0` |
| **`train_auto_v1`** | 5,205 | CPU | Mar 06, 2026 | `results.csv` | `yolo detect train data=data1/bdd_balanced.yaml model=yolov8l.pt epochs=50 imgsz=640 batch=4` |
| **`train6`** | 6,856 | GPU 0 | (Archived) | `results.csv` | `yolo detect train data=data1/bdd100k.yaml model=yolov8l.pt epochs=50 imgsz=832 batch=8 device=0` |
| **`train3`** | 144,025 | CPU | Feb 15, 2026 | `results.csv` | `yolo detect train data=data1/bdd100k.yaml model=yolov8n.pt epochs=10 imgsz=640 batch=16` |
| **`validation_yolov5s`** | Fast | GPU 0 | Jul 10, 2026 | `opt.yaml` | `python train_yolov5s.py --data data1/bdd_balanced.yaml --weights yolov5s.pt --epochs 1 --batch 16` |
| **`train_yolov5s_v1`** | **Active** | GPU 0 | Jul 10, 2026 | In progress | `python train_yolov5s.py --data data1/bdd_balanced.yaml --weights yolov5s.pt --epochs 50 --batch 16` |

*Note: Training execution for YOLOv5 is orchestrated via Python scripts (`scripts/train_orchestrator.py`), while YOLOv8 execution uses the standard Ultralytics `yolo detect` CLI.*
