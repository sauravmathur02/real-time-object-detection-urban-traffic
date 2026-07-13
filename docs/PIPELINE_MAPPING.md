# Pipeline Mapping

This document maps the theoretical claims of the research paper to the actual code implementation.

| Paper Section | Status | Implementation Trace | Notes / Explanation |
| :--- | :---: | :--- | :--- |
| **Data Preprocessing** (BDD100k JSON to YOLO) | **MATCH** | `src/bdd_to_yolo_prod.py`, `src/verify_dataset.py` | Accurately extracts JSON polygons to normalized bounding boxes. |
| **Class Mapping** (IDD standard to BDD schema) | **MATCH** | `src/merge_idd.py` | Maps 13 IDD classes down to the targeted 10 BDD classes flawlessly. |
| **Augmentation** | **PARTIAL** | `scripts/train_orchestrator.py`, Ultralytics core | Uses built-in YOLO mosaic/mixup. No custom explicit augmentation script exists outside the YOLO framework itself. |
| **Dataset Balancing** (Rider Oversampling) | **MATCH** | `src/oversample_rider.py` | Finds `class 1` and physically duplicates the image and label files by a factor of 2. |
| **Hard Example Mining** (Mining false negatives) | **MATCH** | `src/collect_hard_examples.py` | Loads a validation set, runs inference, and extracts images with 0 boxes or confidence < 0.3. |
| **Training** (Multi-Stage Transfer Learning) | **MATCH** | `runs_inspection_report.json` logs, `scripts/train_orchestrator.py` | Stages 1 through 4 are visible in the `runs/` folder using YOLOv8l. YOLOv5s is orchestrated via `train_orchestrator.py`. |
| **Evaluation** (Precision/Recall metrics) | **MATCH** | Ultralytics core evaluation | `results.csv`, `PR_curve.png`, `confusion_matrix.png` exist in run folders. |
| **Inference** (Real-Time NMS processing) | **MATCH** | `src/detector.py`, `src/main.py` | Wrapper successfully utilizes GPU for inference and OpenCV for rendering. |
| **Visualization** (Dashboard) | **MATCH** | `src/dashboard/app.py`, `src/dashboard/camera_manager.py` | Fully threaded FastAPI backend delivering MJPEG streams. |
| **Deployment** (Cloud/Docker) | **MISSING** | None | No containerization logic exists. The paper claims real-world smart-city deployment applicability, but DevOps configurations are completely absent. |
