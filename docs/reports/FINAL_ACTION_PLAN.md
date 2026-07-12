# Final Action Plan

To transition this repository from an incomplete experimental state into a fully compliant academic standard that satisfies all reviewer constraints, the following tasks must be completed in order.

### 🔴 CRITICAL
1. **Wait for YOLOv5s Training to Finish**
   - **Difficulty:** Easy
   - **Time:** Automatic (Wait for process to exit)
   - **Affected Files:** `experiments/yolov5/train_yolov5s_v1/`
   - **Expected Improvement:** Provides the required baseline metrics for the paper's primary comparison.

2. **Generate Qualitative Visual Comparisons**
   - **Difficulty:** Medium
   - **Time:** 2 Hours
   - **Affected Files:** `scripts/generate_qualitative_figs.py` (NEW)
   - **Expected Improvement:** Satisfies the reviewer's explicit request for side-by-side inference images (Original vs YOLOv5 vs YOLOv8).

3. **Develop Benchmark Suite**
   - **Difficulty:** Medium
   - **Time:** 3 Hours
   - **Affected Files:** `scripts/benchmark.py` (NEW)
   - **Expected Improvement:** Generates statistically significant FPS and Latency metrics required for the paper's "Real-Time" claims.

### 🟠 HIGH
4. **Implement Missing YOLO Architectures**
   - **Difficulty:** High
   - **Time:** 3-5 Days (Training time)
   - **Affected Files:** `scripts/train_orchestrator.py` (Modify to sweep models)
   - **Expected Improvement:** Ensures the repository actually contains the YOLOv5m, YOLOv7x, YOLOv8s, and YOLOv8m weights claimed in the paper.

5. **Compile Final Results Table**
   - **Difficulty:** Easy
   - **Time:** 1 Hour
   - **Affected Files:** `evaluation/master_results.csv` (NEW)
   - **Expected Improvement:** Creates the central source of truth for all tables in the research paper.

### 🟡 MEDIUM
6. **Dockerize the Dashboard**
   - **Difficulty:** Low
   - **Time:** 1 Hour
   - **Affected Files:** `Dockerfile`, `docker-compose.yml` (NEW)
   - **Expected Improvement:** Validates the paper's claim regarding "smart city deployment readiness" by providing a reproducible environment.

7. **Clean Root Duplicates**
   - **Difficulty:** Very Easy
   - **Time:** 5 Minutes
   - **Affected Files:** `oversample_rider.py`, `analyze_bdd.py`, `downsample_bdd.py`, etc.
   - **Expected Improvement:** Eliminates code drift and prevents researchers from running broken, obsolete scripts.

### 🟢 LOW
8. **Export Models to TensorRT**
   - **Difficulty:** Medium
   - **Time:** 2 Hours
   - **Affected Files:** `scripts/export_models.py` (NEW)
   - **Expected Improvement:** Substantially boosts FPS during inference, backing up real-time claims.

9. **Add Standard Academic Files**
   - **Difficulty:** Easy
   - **Time:** 15 Minutes
   - **Affected Files:** `CITATION.cff`, `LICENSE` (NEW)
   - **Expected Improvement:** Makes the repository legally open-source and properly citable.
