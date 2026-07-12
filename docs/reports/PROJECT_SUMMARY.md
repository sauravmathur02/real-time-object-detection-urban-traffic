# Project Summary

## Project Purpose
The objective of this repository is to develop and validate a robust, real-time urban traffic object detection system suitable for autonomous driving and smart-city applications. It aims to deliver high-accuracy detections across multiple categories of road users and vehicles, operating efficiently under varying lighting conditions and dense city environments, with a specific focus on developing economies by targeting vulnerable road users and regional vehicles (e.g., auto-rickshaws).

## Complete Pipeline
1. **Dataset Pipeline:** Converts raw BDD100K JSON labels to YOLO format, merges IDD and Auto-rickshaw datasets, and oversamples minority classes (rider).
2. **Training Pipeline:** Multi-stage transfer learning pipeline using YOLOv8l architecture.
3. **Hard-Example Mining Pipeline:** Isolates low-confidence inferences and false negatives from validation runs for targeted model fine-tuning.
4. **Inference Pipeline:** Non-Maximum Suppression (NMS) handling, bounding box overlaying via OpenCV for webcam and video feeds.
5. **Dashboard Pipeline:** Live visualization via an interactive FastAPI frontend.

## Datasets
- **BDD100K (Berkeley DeepDrive) Subset:** Base urban traffic dataset.
- **IDD (India Driving Dataset) Subset:** Mapped and integrated to account for Indian road scenes.
- **Datacluster Labs Auto-Rickshaw Dataset:** Custom dataset integrated to detect 3-wheel auto-rickshaws.

## Training Workflow
A multi-stage transfer learning strategy utilizing YOLO architectures (mainly YOLOv8l):
- **Stage 1:** Base model initialized on COCO weights and pre-trained on BDD100K base classes for 30 epochs.
- **Stage 2:** Fine-tuning continued on BDD100K for 70 additional epochs (total 100).
- **Stage 3:** Model trained on mined hard-examples with stronger augmentations.
- **Stage 4:** Final tuning on BDD Balanced (merged with IDD and Auto-rickshaw, plus rider oversampling).

## Inference Workflow
Inference is executed using `src/main.py` and `src/detector.py`, employing a custom YOLO wrapper (`UrbanDetector`). It captures video or webcam streams, performs standard preprocessing (resizing, normalization), outputs bounding boxes/confidence scores, and overlays them onto the frame in real-time.

## Deployment Workflow
**NOT IMPLEMENTED.** The repository currently lacks production deployment configurations such as Docker containerization (`Dockerfile`, `docker-compose.yml`), CI/CD workflows, or cloud platform deployment scripts.

## Dashboard Workflow
A fully functional FastAPI web app (`src/dashboard/app.py`). It employs a background thread using `CameraManager` to read frames, pass them through the YOLO detector, cache predictions, encode frames as JPEGs, and stream them via `StreamingResponse` to a glassmorphic HTML/JS frontend that allows for real-time visualization and confidence threshold controls.

## Evaluation Workflow
Utilizes YOLO's native validation commands to evaluate trained models against target validation datasets. It calculates standard metrics (Precision, Recall, mAP@50, mAP@50-95, Box/Cls/Dfl losses) and auto-generates visual validation artifacts like Confusion Matrices, PR Curves, and Per-Class AP lists.

## Research Contributions
- **Unified Regional Label Space:** Fusing standard western datasets (BDD) with regional datasets (IDD, Auto) into a consolidated 10-class label schema.
- **Class Imbalance Mitigation:** Automated oversampling to recover minority road users (riders).
- **Hard-Example Mining:** Systematic identification and retraining on missed detections.
- **Interactive Visualizer:** A low-latency web dashboard explicitly designed to adjust inference confidence dynamically.
