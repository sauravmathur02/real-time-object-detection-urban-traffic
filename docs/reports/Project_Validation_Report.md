# Project Validation Report: Real-Time Object Detection in Urban Traffic Scenes

**Date:** July 10, 2026  
**Auditor:** Independent Research Auditor (AI System Instance)  
**Status:** Provisional (Training in Progress)  
**Target Project:** [object-Detection](file:///c:/Repo/object-Detection)  

---

## SECTION 1: PROJECT SUMMARY

### Project Objective
The objective of this project is to develop and validate a robust, real-time urban traffic object detection system suitable for autonomous driving and smart-city applications. It aims to deliver high-accuracy detections across multiple categories of road users and vehicles, operating efficiently under varying lighting conditions and dense city environments.

### Research Problem
Urban traffic scenes present severe challenges for standard computer vision models:
1. **Vulnerability & Scales:** Small, vulnerable road users (e.g., pedestrians, riders, cyclists) are easily obscured and exhibit high visual variance.
2. **Regional Variance:** Standard driving datasets (e.g., COCO, BDD100K) lack common regional transport vehicles such as three-wheeled auto-rickshaws, limiting their deployment in developing economies like India.
3. **Class Imbalance:** Extreme class distribution differences (e.g., a high volume of cars vs. very few riders) lead to model bias and high false-negative rates for minority classes.

### Novel Contribution
The repository addresses these problems through a unified workflow consisting of:
- **Dataset Integration & Fusion:** Merging the Berkeley BDD100K base dataset with the India Driving Dataset (IDD) and a custom Auto-rickshaw dataset, mapping them into a unified 10-class label space.
- **Class Balancing:** Automated oversampling of underrepresented classes (specifically the `rider` class).
- **Hard-Example Mining:** An automated mining pipeline that isolates low-confidence inferences and false negatives from validation runs, allowing for targeted model fine-tuning.
- **FastAPI Visualizer:** An interactive, glassmorphic live stream dashboard for real-time visualization and user control of detection thresholds.

### Detection Pipeline
1. Input frame from webcam, uploaded video, or stream.
2. Frame preprocessed (resize, normalize) and passed to the custom YOLO model wrapper [UrbanDetector](file:///c:/Repo/object-Detection/src/detector.py#L4).
3. The model outputs class bounding boxes and confidence scores.
4. Non-Maximum Suppression (NMS) removes duplicate detections.
5. Detections are overlaid on the video stream and rendered inside cv2 GUI or streamed to the FastAPI frontend.

### Dataset Pipeline
1. Raw BDD100K labels converted to YOLO format via [bdd_to_yolo_prod.py](file:///c:/Repo/object-Detection/src/bdd_to_yolo_prod.py).
2. Converted label-image mappings verified using [verify_dataset.py](file:///c:/Repo/object-Detection/src/verify_dataset.py).
3. XML auto-rickshaw annotations converted via [convert_auto_to_yolo.py](file:///c:/Repo/object-Detection/src/convert_auto_to_yolo.py).
4. IDD labels remapped and merged via [merge_idd.py](file:///c:/Repo/object-Detection/src/merge_idd.py).
5. Auto-rickshaw labels merged via [merge_auto_yolo.py](file:///c:/Repo/object-Detection/src/merge_auto_yolo.py).
6. Minority class oversampling applied via [oversample_rider.py](file:///c:/Repo/object-Detection/src/oversample_rider.py).

### Training Pipeline
A multi-stage transfer learning strategy is employed using YOLOv8l:
1. **Stage 1:** Base model initialized on COCO weights and pre-trained on BDD100K base classes for 30 epochs.
2. **Stage 2:** Fine-tuning continued on BDD100K for 70 additional epochs (total 100).
3. **Stage 3:** Model trained on mined hard-examples (BDD without train class) with stronger augmentations.
4. **Stage 4:** Final tuning on BDD Balanced (merged with IDD and Auto-rickshaw, plus rider oversampling) for 50 epochs.

### Evaluation Pipeline
Runs validation on target datasets after training, computing standard metrics (Precision, Recall, mAP@50, mAP@50-95, box/cls/dfl losses), and automatically plotting the Confusion Matrix, PR Curves, and Per-Class AP lists.

### Deployment & Dashboard Pipeline
FastAPI web app (`src/dashboard/app.py`) starts a background thread running [CameraManager](file:///c:/Repo/object-Detection/src/dashboard/camera_manager.py#L9). The manager reads video frames, feeds them to [UrbanDetector](file:///c:/Repo/object-Detection/src/detector.py#L4), caches predictions and performance stats, encodes frames as JPEG, and streams them to the glassmorphic HTML/JS interface via a StreamingResponse.

---

## SECTION 2: DATASET INFORMATION

### Dataset Details
- **Dataset Names:** 
  1. BDD100K (Berkeley DeepDrive) Subset
  2. IDD (India Driving Dataset) Subset
  3. Datacluster Labs Auto-Rickshaw Dataset
- **Dataset Versions:** Standard release variants remapped to YOLO formats.
- **Number of Datasets:** 3 distinct sources.
- **Training Images:**
  - Base BDD subset (`bdd_yolo`): **1,153** images
  - Merged Balanced subset (`bdd_balanced`): **34,536** images
- **Validation Images:**
  - Base BDD subset (`bdd_yolo`): **9,999** images
  - Merged Balanced subset (`bdd_balanced`): **14,150** images
- **Test Images:** `NOT FOUND` (The local workspace does not contain test splits).
- **Number of Classes:** 10 classes in both active schemas.
- **Class Names:**
  - **BDD100K Base:** `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`
  - **BDD Balanced:** `pedestrian`, `rider`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `auto`

### Dataset Merge Workflow
- **IDD Integration:** Remapped 13 original IDD class IDs to BDD equivalents using [merge_idd.py](file:///c:/Repo/object-Detection/src/merge_idd.py). Remapped index `1` (autorickshaw) -> `9` (auto), caravan and trailer to `truck`, and person to `pedestrian`.
- **Auto-rickshaw Integration:** Converted Pascal VOC XML bounding boxes (`autorickshaw`/`auto`) to normalized YOLO coordinates with class ID `9` via [convert_auto_to_yolo.py](file:///c:/Repo/object-Detection/src/convert_auto_to_yolo.py). Integrated them using [merge_auto_yolo.py](file:///c:/Repo/object-Detection/src/merge_auto_yolo.py) with an 80/20 train/val split.
- **Oversampling Strategy:** Duplicated training samples containing the rare `rider` class (class ID 1) by a factor of 2 via [oversample_rider.py](file:///c:/Repo/object-Detection/src/oversample_rider.py) to reduce class imbalance.

### Folder Structure
```text
object-Detection/data1/
|-- bdd100k/                       # Raw BDD100K subset images & labels
|-- IDDDetectionsYOLODataset/      # Raw IDD subset files (YOLO format)
|-- auto/                          # Auto-rickshaw images
|-- Annotations/                   # Auto-rickshaw XML annotations
|-- auto_yolo/                     # Converted auto-rickshaw YOLO files
|-- bdd_yolo/                      # BDD100K base dataset (YOLO format)
|-- bdd_balanced/                  # Unified final merged dataset
|   |-- images/
|   |   |-- train                  # 34,536 images (.jpg + .npy cache)
|   |   `-- val                    # 14,150 images (.jpg + .npy cache)
|   `-- labels/
|       |-- train                  # 34,536 text labels
|       `-- val                    # 14,150 text labels
|-- hard_examples/                 # Missed and low-confidence mined frames
```

### Dataset YAML Configurations

#### 1. [bdd100k.yaml](file:///c:/Repo/object-Detection/data1/bdd100k.yaml)
```yaml
path: ./bdd_yolo
train: images/train
val: images/val

nc: 10

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

#### 2. [bdd_balanced.yaml](file:///c:/Repo/object-Detection/data1/bdd_balanced.yaml)
```yaml
path: c:/Repo/object-Detection/data1/bdd_balanced

train: images/train
val: images/val

nc: 10

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: motorcycle
  6: bicycle
  7: traffic light
  8: traffic sign
  9: auto
```

---

## SECTION 3: MODEL INVENTORY

A comprehensive search of the repository reveals the following models:

### 1. YOLOv8l (object_stage1)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/object_stage1/weights/best.pt)
- **Dataset Used:** `data1/bdd100k.yaml`
- **Epochs:** 30
- **Batch Size:** 8
- **Image Size:** 832
- **Optimizer:** auto
- **Device:** GPU (device 0)
- **Date Trained:** Feb 28, 2026
- **Checkpoints:** Best & Last

### 2. YOLOv8l (object_stage2)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/object_stage2/weights/best.pt)
- **Dataset Used:** `data1/bdd100k.yaml` (from Stage 1 weights)
- **Epochs:** 70
- **Batch Size:** 8
- **Image Size:** 832
- **Optimizer:** auto
- **Device:** GPU (device 0)
- **Date Trained:** Feb 28, 2026
- **Checkpoints:** Best & Last

### 3. YOLOv8l (object_stage3_strong2)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/object_stage3_strong2/weights/best.pt)
- **Dataset Used:** `data1/bdd_no_train.yaml` (from Stage 2 weights)
- **Epochs:** 50
- **Batch Size:** 6
- **Image Size:** 960
- **Optimizer:** auto
- **Device:** GPU (device 0)
- **Date Trained:** Mar 02, 2026
- **Checkpoints:** Best & Last

### 4. YOLOv8l (object_stage4_riderfix4)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/object_stage4_riderfix4/weights/best.pt)
- **Dataset Used:** `data1/bdd_balanced.yaml` (from Stage 3 weights)
- **Epochs:** 50
- **Batch Size:** 6
- **Image Size:** 960
- **Optimizer:** auto
- **Device:** GPU (device 0)
- **Date Trained:** Mar 05, 2026
- **Checkpoints:** Best & Last

### 5. YOLOv8l (train_fast2)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/train_fast2/weights/best.pt)
- **Dataset Used:** `data1/bdd_balanced.yaml` (from Stage 3 weights)
- **Epochs:** 30
- **Batch Size:** 32
- **Image Size:** 512
- **Optimizer:** auto
- **Device:** GPU (device 0)
- **Date Trained:** Mar 10, 2026
- **Checkpoints:** Best & Last

### 6. YOLOv8l (train_auto_v1)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/train_auto_v1/weights/best.pt)
- **Dataset Used:** `data1/bdd_balanced.yaml`
- **Epochs:** 50
- **Batch Size:** 4
- **Image Size:** 640
- **Optimizer:** auto
- **Device:** CPU (None)
- **Date Trained:** Mar 06, 2026
- **Checkpoints:** Best & Last

### 7. YOLOv8n (train3)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/runs/detect/train3/weights/best.pt)
- **Dataset Used:** `data/bdd100k.yaml`
- **Epochs:** 10
- **Batch Size:** 16
- **Image Size:** 640
- **Optimizer:** auto
- **Device:** CPU
- **Date Trained:** Feb 15, 2026
- **Checkpoints:** Best & Last

### 8. YOLOv5s (validation_yolov5s)
- **Exists:** YES
- **Trained:** YES (Completed)
- **Training Status:** Completed (Sanity check)
- **Weight Path:** [best.pt](file:///c:/Repo/object-Detection/experiments/yolov5/validation_yolov5s/weights/best.pt)
- **Dataset Used:** `data1/bdd_balanced.yaml`
- **Epochs:** 1
- **Batch Size:** 16
- **Image Size:** 640
- **Optimizer:** SGD
- **Device:** GPU (device 0)
- **Date Trained:** Jul 10, 2026
- **Checkpoints:** Best & Last

### 9. YOLOv5s (train_yolov5s_v1)
- **Exists:** YES
- **Trained:** NO (Training in progress)
- **Training Status:** **TRAINING IN PROGRESS – FINAL RESULT PENDING**
- **Weight Path:** None (active checkpoint written to [weights](file:///c:/Repo/object-Detection/experiments/yolov5/train_yolov5s_v1/weights))
- **Dataset Used:** `data1/bdd_balanced.yaml`
- **Epochs:** 50 (Target)
- **Batch Size:** 16
- **Image Size:** 640
- **Optimizer:** SGD
- **Device:** GPU (device 0)
- **Date Trained:** Active (Started Jul 10, 2026 15:35)
- **Checkpoints:** Active (`last.pt` / `best.pt` changing dynamically)

### Other Models Search
- **YOLOv5m:** `NOT IMPLEMENTED` (No configs, weights, or logs exist in the repository).
- **YOLOv7x:** `NOT IMPLEMENTED` (No configs, weights, or logs exist in the repository).
- **YOLOv8s:** `NOT IMPLEMENTED` (The directory `train_yolov8s` exists but contains no logs or weights).
- **YOLOv8m:** `NOT IMPLEMENTED`.

---

## SECTION 4: TRAINING RESULTS

The metrics below are extracted from local training run records (`results.csv`, `opt.yaml`, `args.yaml`) in `runs/detect` and `experiments/yolov5`:

| Run Identifier | Model | Epochs | Batch | Imgsz | Precision | Recall | mAP@50 | mAP@50-95 | Train/Val Loss | Time (s) | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **object_stage1** | YOLOv8l | 30 | 8 | 832 | 0.6443 | 0.4012 | 0.4106 | 0.2256 | 1.169 / 1.373 | 4,078 | `best.pt`, `last.pt` |
| **object_stage2** | YOLOv8l | 70 | 8 | 832 | 0.5058 | 0.3834 | 0.3901 | 0.2121 | 0.858 / 1.443 | 9,219 | `best.pt`, `last.pt` |
| **object_stage3_strong2** | YOLOv8l | 50 | 6 | 960 | 0.6116 | 0.4538 | 0.4732 | 0.2552 | 0.988 / 1.493 | 8,354 | `best.pt`, `last.pt` |
| **object_stage4_riderfix4** | YOLOv8l | 50 | 6 | 960 | 0.5998 | 0.4463 | 0.4618 | 0.2572 | 0.655 / 1.483 | 8,672 | `best.pt`, `last.pt` |
| **train_fast2** | YOLOv8l | 30 | 32 | 512 | 0.7147 | 0.5065 | 0.5524 | 0.3480 | 0.973 / 1.227 | 48,078 | `best.pt`, `last.pt` |
| **train_auto_v1** | YOLOv8l | 50 | 4 | 640 | 0.5554 | 0.4742 | 0.4828 | 0.2952 | 1.049 / 1.365 | 5,205 | `best.pt`, `last.pt` |
| **train6** | YOLOv8l | 50 | 8 | 832 | 0.6526 | 0.3983 | 0.4086 | 0.2244 | 1.045 / 1.404 | 6,856 | `best.pt`, `last.pt` |
| **train3** | YOLOv8n | 10 | 16 | 640 | 0.4049 | 0.0019 | 0.2034 | 0.1220 | 0.470 / 1.862 | 144,025 | `best.pt`, `last.pt` |
| **validation_yolov5s** | YOLOv5s | 1 | 16 | 640 | 0.5072 | 0.2769 | 0.2726 | 0.1494 | 0.086 / 0.071 | Fast | `best.pt`, `last.pt` |
| **train_yolov5s_v1** | YOLOv5s | 50 | 16 | 640 | **PENDING** | **PENDING** | **PENDING** | **PENDING** | **PENDING** | **PENDING** | **PENDING** |

> [!NOTE]
> **Active YOLOv5s Training Status:** The training run `train_yolov5s_v1` is currently active. The last logged metrics at **Epoch 10** are:
> - **Precision:** 0.65861
> - **Recall:** 0.42751
> - **mAP@50:** 0.47126
> - **mAP@50-95:** 0.27244
> - **Train Box/Obj/Cls Loss:** 0.067867 / 0.050062 / 0.024563
> - **Val Box/Obj/Cls Loss:** 0.062933 / 0.071744 / 0.014115
> - Final metrics and checkpoints are: **TRAINING IN PROGRESS – FINAL RESULT PENDING**

### Visual Artifacts
- **Confusion Matrix:** Automatically generated for completed models and stored in the respective run folder (e.g. `runs/detect/object_stage4_riderfix4/confusion_matrix.png`).
- **PR Curves:** Automatically generated and saved in run folders (e.g. `validation_yolov5s/PR_curve.png`).
- **Per-Class AP:** Calculated during validation steps and saved in run logs.

---

## SECTION 5: PROJECT FEATURES

Below is the verification status of core project pipeline components:

| Feature / Pipeline | Status | Evidence | Notes |
| :--- | :--- | :--- | :--- |
| **Dataset Preprocessing** | **IMPLEMENTED** | `src/bdd_to_yolo.py` | Basic conversion scripts |
| **Dataset Verification** | **IMPLEMENTED** | [verify_dataset.py](file:///c:/Repo/object-Detection/src/verify_dataset.py) | Checks folder alignment |
| **Dataset Merging** | **IMPLEMENTED** | [merge_idd.py](file:///c:/Repo/object-Detection/src/merge_idd.py), [merge_auto_yolo.py](file:///c:/Repo/object-Detection/src/merge_auto_yolo.py) | Remaps and fuses IDD/Auto |
| **Oversampling** | **IMPLEMENTED** | [oversample_rider.py](file:///c:/Repo/object-Detection/src/oversample_rider.py) | Rider class duplicator |
| **Auto-Rickshaw Integration** | **IMPLEMENTED** | [convert_auto_to_yolo.py](file:///c:/Repo/object-Detection/src/convert_auto_to_yolo.py) | XML parser to YOLO format |
| **IDD Integration** | **IMPLEMENTED** | [merge_idd.py](file:///c:/Repo/object-Detection/src/merge_idd.py) | Class mapping script |
| **BDD100K Conversion** | **IMPLEMENTED** | `src/bdd_to_yolo_prod.py` | Image copy workflow |
| **YOLOv5 Support** | **PARTIAL** | `scripts/train_yolov5s.py` | Submodule train code, active YOLOv5s training |
| **YOLOv7 Support** | **NOT IMPLEMENTED** | None | No YOLOv7 model code or scripts found |
| **YOLOv8 Support** | **IMPLEMENTED** | `src/detector.py`, `src/predict_webcam.py` | Native Ultralytics implementation |
| **Training Pipeline** | **IMPLEMENTED** | `scripts/train_orchestrator.py` | Orchestration scripts for multi-stage runs |
| **Inference Pipeline** | **IMPLEMENTED** | [main.py](file:///c:/Repo/object-Detection/src/main.py), `src/detector.py` | OpenCV frame rendering |
| **Evaluation Pipeline** | **IMPLEMENTED** | YOLO `val` commands | Generates logs, CSV metrics, and curves |
| **Benchmark Pipeline** | **NOT IMPLEMENTED** | None | No speed/accuracy comparison suite found |
| **FastAPI Dashboard** | **IMPLEMENTED** | `src/dashboard/app.py` | Live preview web API |
| **Camera Manager** | **IMPLEMENTED** | `src/dashboard/camera_manager.py` | Threaded frame streamer |
| **Detector API** | **IMPLEMENTED** | `src/detector.py` | Wrapper class Around YOLO |
| **Web Interface** | **IMPLEMENTED** | `src/dashboard/static/index.html` | Glassmorphic frontend |
| **Model Comparison** | **NOT IMPLEMENTED** | None | No qualitative / quantitative comparison scripts |
| **Qualitative Comparison** | **NOT IMPLEMENTED** | None | No side-by-side verification figures |
| **Deployment Configuration**| **NOT IMPLEMENTED** | None | No Docker or container setups |

---

## SECTION 6: PAPER REPRODUCIBILITY

This section checks whether claims in the paper's outline can be reproduced using the current repository state:

| Paper Claim | Repository Evidence | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Dataset Configuration** | [bdd_balanced.yaml](file:///c:/Repo/object-Detection/data1/bdd_balanced.yaml) | **✓ MATCH** | Contains 10 final classes matching target schema |
| **Class Mapping** | [merge_idd.py](file:///c:/Repo/object-Detection/src/merge_idd.py#L16-L30) | **✓ MATCH** | Implements the IDD-to-BDD mapping correctly |
| **YOLOv5s** | `train_yolov5s.py`, `train_yolov5s_v1/` | **⚠ PARTIAL** | Training is actively running; results pending |
| **YOLOv5m** | None | **✗ NOT FOUND** | No YOLOv5m configs, weights, or references found |
| **YOLOv7x** | None | **✗ NOT FOUND** | No YOLOv7 architecture support found in repo |
| **YOLOv8n** | `runs/detect/train3` weights | **✓ MATCH** | Trained successfully for 10 epochs on CPU |
| **YOLOv8s** | `runs/detect/train_yolov8s` (empty) | **✗ NOT FOUND** | Folder exists but contains no training runs |
| **YOLOv8m** | None | **✗ NOT FOUND** | No YOLOv8m configs, weights, or references found |
| **YOLOv8l** | `object_stage4_riderfix4/weights/best.pt` | **✓ MATCH** | Fully trained through all 4 stages |
| **Training Pipeline** | `scripts/train_orchestrator.py` | **✓ MATCH** | Orchestration workflow runs Stage 1-4 training |
| **Evaluation Metrics** | `results.csv` files | **✓ MATCH** | Standard YOLO verification CSVs present |
| **Benchmark Metrics** | None | **✗ NOT FOUND** | No comparison or benchmarking scripts |
| **Inference Scripts** | [main.py](file:///c:/Repo/object-Detection/src/main.py) | **✓ MATCH** | Standard webcam/video inference wrapper |
| **FastAPI Dashboard** | `src/dashboard/app.py` | **✓ MATCH** | Glassmorphic active dashboard interface |
| **Deployment Setup** | None | **✗ NOT FOUND** | No production cloud or container config files |
| **Qualitative Fig** | None | **✗ NOT FOUND** | No visual detection comparisons between model versions |
| **Confusion Matrix** | `confusion_matrix.png` files | **✓ MATCH** | Automatically output by Ultralytics val runs |
| **PR Curve** | `PR_curve.png` files | **✓ MATCH** | Automatically output by Ultralytics val runs |
| **Per-Class AP** | Val logs | **✓ MATCH** | Per-class AP available in log archives |

### Explanations of Mismatches
1. **YOLOv5m, YOLOv7x, YOLOv8m:** Not implemented. No training code, weights, or configuration parameters are defined for these architectures in the repository.
2. **YOLOv8s:** A directory was created but no training was ever executed (empty folders, no checkpoints or logs).
3. **YOLOv5s:** Training is actively running. Final validation metrics are pending.
4. **Benchmark & Qualitative Fig:** No comparison script comparing cross-architecture inference speeds (FPS) or side-by-side detection images exists.

---

## SECTION 7: METRIC COMPARISON

> [!WARNING]
> **Research Paper Draft Missing:** The conference paper draft was **NOT FOUND** in the workspace. No draft metrics tables or baseline figures are available in the project directories to compile a direct validation comparison.
> 
> Furthermore, the YOLOv5s training run is currently active, which blocks final metric logging for the comparison models.

| Metric | Paper Value | Repository Value | Difference | % Var | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv5s mAP@50** | Unknown | **PENDING** | N/A | N/A | **TRAINING IN PROGRESS – FINAL RESULT PENDING** |
| **YOLOv5s Precision** | Unknown | **PENDING** | N/A | N/A | **TRAINING IN PROGRESS – FINAL RESULT PENDING** |
| **YOLOv5s Recall** | Unknown | **PENDING** | N/A | N/A | **TRAINING IN PROGRESS – FINAL RESULT PENDING** |
| **YOLOv8l mAP@50** | Unknown | 0.4618 (stage4) | N/A | N/A | Final Baseline Logged (0.46175) |
| **YOLOv8l Precision** | Unknown | 0.5998 (stage4) | N/A | N/A | Final Baseline Logged (0.59976) |
| **YOLOv8l Recall** | Unknown | 0.4463 (stage4) | N/A | N/A | Final Baseline Logged (0.44633) |

---

## SECTION 8: QUALITATIVE VALIDATION

### Reviewer Requirement
The reviewer requested a qualitative detection comparison displaying side-by-side inference outputs of the:
1. **Original Image** (Ground Truth / Unannotated)
2. **YOLOv5s Detections**
3. **YOLOv8l Detections**

### Implementation Status
This comparison has **NOT BEEN IMPLEMENTED**.

### Remaining Tasks to Resolve Reviewer Requirements
1. **Inference Script:** A Python comparison script (e.g. `scripts/generate_qualitative_figs.py`) needs to be created. This script should load `yolov5s` and `yolov8l` weights, run inference on the same subset of validation images, and save combined comparison figures.
2. **Comparison Image Generation:** Export images showing side-by-side bounding boxes for road scenes containing minority classes (`rider`, `auto`) during day, night, and heavy traffic conditions.
3. **Documentation:** Insert these comparison figures directly into the results folder and include a description in the paper.

---

## SECTION 9: REPOSITORY HEALTH

An evaluation of the repository health is detailed below:

| Audit Category | Score (1-10) | Auditor Rationale & Explanations |
| :--- | :---: | :--- |
| **Folder Structure** | **8 / 10** | Well-separated datasets, runs, src, and script dirs. However, old duplicate scripts in the root directory (`oversample_rider.py`, `downsample_bdd.py`, `remove_train_class.py`) conflict with `src/` files and clutter the root. |
| **Architecture** | **8.5 / 10**| Clear modular separation of pipeline stages: conversion scripts, training orchestrator, and FastAPI dashboard camera threads are well isolated. |
| **Documentation** | **7.5 / 10**| Clean setup guides (`README.md`, `IMPLEMENTATION_GUIDE.md`) exist, but there is no paper draft or explanation of training run histories/configs. |
| **Code Quality** | **8.5 / 10**| Uses clean standard libraries, structured arguments (`argparse`), progress bars (`tqdm`), and correct thread safety locks inside `camera_manager.py`. |
| **Maintainability** | **8 / 10** | Clear script naming, but duplicate code in the root directory risks configuration drift and increases maintenance cost. |
| **Scalability** | **8 / 10** | Pipeline handles large directories easily. Could be improved by moving hardcoded folders in scripts to central environment configuration files. |
| **Reproducibility** | **7 / 10** | The YOLOv5 orchestrator records `environment.json` and `experiment.yaml`. However, custom YOLOv8 stages lack documented seeds and parameters. |
| **Deployment Readiness**| **6.5 / 10**| Local FastAPI server works cleanly. There are no production configuration setups, environment variables, or Docker containerizations. |
| **Research Repro.** | **6 / 10** | Baseline paper metrics tables are missing from the repo, preventing direct verification of model performance claims. |

---

## SECTION 10: MISSING ITEMS

Below is a checklist of tasks still missing from the repository:
1. **YOLOv5m Model:** No training scripts, configurations, or weights exist.
2. **YOLOv7x Model:** Completely missing from the repository files.
3. **YOLOv8s Model:** Training folder `train_yolov8s` is empty; model was never trained.
4. **YOLOv8m Model:** Completely missing from the repository files.
5. **Qualitative Figure Generator:** Visual side-by-side comparison figures comparing Original vs. YOLOv5s vs. YOLOv8l are not generated.
6. **Benchmark Pipeline:** A tool to compare inference latency, speed (FPS), and mAP across trained versions is missing.
7. **Production Deployment Configs:** Missing Docker container configurations (`Dockerfile` / `docker-compose.yml`) or production server settings.
8. **Conference Paper Draft:** The actual LaTeX or Word draft of the conference paper is missing.

---

## SECTION 11: ITEMS PENDING COMPLETION

The following items are blocked and cannot be finalized because the YOLOv5s training run (`train_yolov5s_v1`) is actively running:
1. **Current Epoch:** Training is at **Epoch 10 / 50**.
2. **Final mAP:** Final mAP@50 and mAP@50-95 values are pending.
3. **Final Precision:** Final precision score is pending.
4. **Final Recall:** Final recall score is pending.
5. **Final Checkpoint:** The final optimized weight file (`best.pt`) is pending.
6. **Final validation report logs:** Post-training validation evaluations are pending.
7. **PR Curve & Confusion Matrix plots:** Final curve charts for YOLOv5s are pending.
8. **Qualitative Figures:** Generation of YOLOv5s detections for the reviewer comparison requires the final checkpoint.

---

## SECTION 12: SUPERVISOR SUMMARY

**To:** Research Supervisor  
**From:** Independent Research Auditor  
**Date:** July 10, 2026  
**Subject:** Technical Audit of YOLO Urban Object Detection Repository  

### Project Status Overview
This audit evaluated the `object-Detection` repository to compare current implementation states against the conference research claims. The pipeline is **PARTIALLY COMPLETED**. Core modules for dataset preprocessing, 1:1 integrity verification, custom IDD/Auto remapping, and class balancing have been successfully implemented. 

However, a critical YOLOv5s training run is **actively running** on the system, blocking final metric validation. Furthermore, several models (YOLOv5m, YOLOv7x, YOLOv8s, YOLOv8m) and reviewer requirements (qualitative comparison figures) remain **NOT IMPLEMENTED**.

### Status Breakdown

#### 1. COMPLETED
- **Dataset Preprocessing:** Raw BDD100K JSON conversion scripts are functional.
- **Dataset Merging:** remaps and integrates IDD and Auto-rickshaw datasets into BDD.
- **Class Balancing:** Oversampling minority classes (duplicating `rider` images) is completed.
- **Hard-Example Mining:** Isolated missed and low-confidence detections from validation runs.
- **YOLOv8l & YOLOv8n Models:** Multi-stage training (Stage 1 to 4) of YOLOv8l is fully completed.
- **FastAPI Dashboard:** Live streaming dashboard with conf slider controls, uploader, and threaded CameraManager is fully functional.

#### 2. IN PROGRESS
- **YOLOv5s Training:** The training orchestrator is actively running `train_yolov5s_v1` on GPU device 0 (Epoch 10 of 50 completed).
- **YOLOv5s Metrics & Plots:** results.csv log updating in real-time. Final metrics are pending.

#### 3. NOT IMPLEMENTED
- **Reviewer Qualitative Figures:** Side-by-side comparison images of Original vs. YOLOv5s vs. YOLOv8l have not been generated.
- **Benchmarking Suite:** A consolidated benchmarking tool for measuring FPS and speed differences is missing.
- **Other Models:** YOLOv5m, YOLOv7x, YOLOv8s, YOLOv8m training has not been executed.
- **Deployment configs:** Docker container configs are missing.
- **Paper Draft Document:** The draft of the conference paper is not stored in the repository.

---

## SECTION 13: CURRENT STATUS

Based on the actual state of the repository, the project is classified as:

### **PARTIALLY READY** (Provisional Assessment)

#### Rationale
The repository contains functional dataset conversion and fusion pipelines, completed multi-stage training runs for the YOLOv8l architecture, and a working FastAPI dashboard. However, since the YOLOv5s training job is actively running, final metrics cannot yet be logged. The reviewer's qualitative comparison request and benchmarking suite remain unimplemented.

*This assessment is **PROVISIONAL** and subject to change after the active training finishes and the reviewer requirements are resolved.*
