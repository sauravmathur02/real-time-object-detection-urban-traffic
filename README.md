# Real-Time Urban Object Detection with YOLOv8

This repository contains a real-time urban object detection workflow built around YOLOv8. It includes model inference scripts, BDD100K conversion utilities, class-balancing helpers, hard-example mining, and extra dataset fusion steps for Indian traffic scenarios such as IDD remapping and auto-rickshaw integration.

## Overview

- Real-time webcam and video inference with OpenCV + YOLOv8
- BDD100K JSON-to-YOLO conversion utilities
- Dataset integrity verification and rider oversampling
- Hard-example mining for iterative retraining
- IDD label remapping into the BDD label space
- Auto-rickshaw VOC/XML conversion and merge workflow

## Core label spaces

### BDD100K base model

`pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`

### Balanced auto dataset

The optional balanced dataset swaps the `train` class out of the working label space and adds `auto` instead:

`pedestrian`, `rider`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `auto`

## Repository structure

```text
object-Detection/
|-- data1/                     # Local dataset workspace (YAML manifests are tracked, data stays ignored)
|-- paper/                     # Training notes and research writeups
|-- src/
|   |-- main.py                # Real-time inference entry point
|   |-- detector.py            # YOLO model wrapper
|   |-- predict_webcam.py      # Minimal webcam/video inference helper
|   |-- bdd_to_yolo.py         # Basic BDD JSON -> YOLO labels
|   |-- bdd_to_yolo_prod.py    # BDD JSON -> YOLO labels + image copy workflow
|   |-- verify_dataset.py      # Dataset integrity checks
|   |-- oversample_rider.py    # Rider class balancing utility
|   |-- collect_hard_examples.py
|   |-- merge_idd.py           # IDD to BDD label remap + merge
|   |-- convert_auto_to_yolo.py
|   |-- merge_auto_yolo.py
|-- IMPLEMENTATION_GUIDE.md
|-- requirements.txt
`-- README.md
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Keep datasets, training runs, videos, and `.pt` weights local. They are intentionally excluded from git.

3. Review the tracked dataset manifests:

- `data1/bdd100k.yaml`
- `data1/bdd_balanced.yaml`

## Dataset preparation

### Convert BDD100K into YOLO format

Use the production conversion script when you want both labels and matched images copied into a YOLO folder structure:

```bash
python src/bdd_to_yolo_prod.py \
  --json_train path/to/bdd100k_labels_images_train.json \
  --json_val path/to/bdd100k_labels_images_val.json \
  --images_train path/to/bdd100k/images/100k/train \
  --images_val path/to/bdd100k/images/100k/val \
  --output_dir data1/bdd_yolo
```

### Verify dataset integrity

```bash
python src/verify_dataset.py --data_dir data1/bdd_yolo
```

### Oversample the rider class

```bash
python src/oversample_rider.py --src_dir data1/bdd_yolo --dst_dir data1/bdd_balanced --duplication 2
```

### Merge IDD into the BDD label space

```bash
python src/merge_idd.py --idd-dir data1/IDDDetectionsYOLODataset --bdd-dir data1/bdd_balanced
```

### Convert and merge auto-rickshaw data

```bash
python src/convert_auto_to_yolo.py \
  --images-dir data1/auto/auto \
  --fallback-images-dir data1/auto \
  --xml-dir data1/Annotations/Annotations \
  --output-dir data1/auto_yolo

python src/merge_auto_yolo.py --src-dir data1/auto_yolo --dst-dir data1/bdd_balanced
```

### Mine hard examples

```bash
python src/collect_hard_examples.py \
  --model runs/detect/train/weights/best.pt \
  --val data1/bdd_yolo/images/val \
  --output data1/hard_examples
```

## Training

Train on the base BDD dataset:

```bash
yolo detect train data=data1/bdd100k.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Train on the balanced dataset with auto-rickshaw integration:

```bash
yolo detect train data=data1/bdd_balanced.yaml model=yolov8l.pt epochs=50 imgsz=640 batch=4 name=train_auto_v1
```

## Inference

Run the main application:

```bash
python src/main.py --source 0 --model runs/detect/train/weights/best.pt
python src/main.py --source path/to/video.mp4 --model runs/detect/train/weights/best.pt --save
```

Quick webcam test:

```bash
python src/predict_webcam.py --source 0 --model runs/detect/train/weights/best.pt
```

## Notes

- The default YOLO base weights are useful for smoke tests, but the full urban label space depends on your custom trained weights.
- `data1/*.yaml` is committed so the repository still contains the dataset schema without checking large data into git.
- `IMPLEMENTATION_GUIDE.md` and the notes under `paper/` document the step-by-step training workflow.

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [IDD Dataset](https://idd.insaan.iiit.ac.in/)
