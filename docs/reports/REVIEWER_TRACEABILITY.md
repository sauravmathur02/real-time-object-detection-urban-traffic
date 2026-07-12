# Reviewer Traceability

Based on the `Project_Validation_Report.md`, peer reviewers requested specific additions to the repository to validate the research paper. This maps those comments to the exact repository state.

### Reviewer Comment 1: "Require qualitative visual comparison of edge cases."
- **Status:** **Missing**
- **Trace:** No script exists in `src/` or `scripts/` that loads multiple weights (e.g. YOLOv5s vs YOLOv8l), runs them on the same image, and outputs a combined side-by-side plot. `src/main.py` only tests a single model at a time.

### Reviewer Comment 2: "Address massive class imbalance for the 'rider' class."
- **Status:** **Implemented**
- **Trace:** `src/oversample_rider.py` line 65 implements a physical duplication loop (`duplication_factor=2`) for any image containing class 1 (rider). The results are used in `bdd_balanced.yaml`.

### Reviewer Comment 3: "Include comparative baseline tables for all claimed YOLO variants."
- **Status:** **Missing**
- **Trace:** The paper mentions YOLOv5m, YOLOv7x, YOLOv8s, and YOLOv8m. However, the `runs/detect/` folder only contains YOLOv8l and YOLOv8n, and `experiments/yolov5` only contains YOLOv5s. No master CSV table exists.

### Reviewer Comment 4: "Clarify real-time FPS benchmarking."
- **Status:** **Partially Implemented**
- **Trace:** `src/main.py` lines 53-57 calculate real-time FPS and draw it on the frame. However, there is no formalized benchmarking script (e.g., `benchmark.py`) that calculates average hardware latency and NMS latency over a standardized 1000-image test set.
