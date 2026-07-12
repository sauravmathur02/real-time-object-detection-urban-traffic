# Reviewer Requirements

The following items are missing from the repository and are required to make it fully support the claims and deliverables outlined in the research paper draft. 

They are prioritized by impact on the paper's validity.

### 🔴 CRITICAL
1. **Wait for YOLOv5s Training Completion:** The `train_yolov5s_v1` model is currently active (Epoch 10). It must finish to extract final mAP, Precision, and Recall scores for baseline comparisons.
2. **Qualitative Visual Comparisons:** The reviewer explicitly asked for side-by-side inference outputs showing: `Original Image | YOLOv5s Detections | YOLOv8l Detections`. A script must be created to load both weights, run inference on the same subset of edge-case images (night, heavy traffic, riders), and output combined figures.
3. **Compile Final Paper Tables:** The research paper's results tables cannot be verified because a master CSV or Markdown table comparing all final models is missing.

### 🟠 HIGH
4. **Implement Missing YOLO Architectures:** The paper claims to test or mentions YOLOv5m, YOLOv7x, YOLOv8s, and YOLOv8m. These architectures have absolutely zero training history, configs, or weights in the repository. They must be trained or removed from the paper claims.
5. **Develop Benchmark Suite (`benchmark.py`):** The paper references real-time capabilities, but there is no formalized benchmarking script to test average FPS, GPU Latency, and NMS Latency consistently across models on the same hardware.

### 🟡 MEDIUM
6. **Clean Up Duplicate Scripts:** The root directory is cluttered with old preprocessing scripts (`oversample_rider.py`, `fix_labels.py`) that also exist in `src/`. Delete the root copies to prevent execution of outdated logic.
7. **Production Deployment Configurations:** If this is designed for "smart cities", a `Dockerfile` and `docker-compose.yml` should exist to containerize the FastAPI dashboard and dependencies.

### 🟢 LOW
8. **Include Paper Draft:** The actual `.tex` (LaTeX) or Word draft of the paper should be tracked in the `paper/` directory to allow reviewers to cross-reference claims directly within the repo.
9. **Remove Unused Large Data:** The `bdd100k_seg` directory contains segmentation masks unused by this bounding box YOLO pipeline, wasting space.
