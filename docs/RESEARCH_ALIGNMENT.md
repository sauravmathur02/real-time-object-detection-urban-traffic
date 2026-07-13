# Research Alignment

This traces the explicit theoretical structure of the research paper directly to the physical repository implementation.

### Introduction & Related Work
- **Status:** **MATCH**
- **Justification:** The project accurately defines an urban traffic detector with a focus on regional datasets (India/Auto).

### Methodology: Dataset Fusion
- **Status:** **MATCH**
- **Justification:** `src/merge_idd.py` and `src/convert_auto_to_yolo.py` perfectly combine disparate data sources into the 10-class BDD100k target space as stated in the methodology.

### Methodology: Class Balancing
- **Status:** **MATCH**
- **Justification:** `src/oversample_rider.py` physically implements the claims of mitigating rider underrepresentation via 2x duplication.

### Methodology: Hard-Example Mining
- **Status:** **MATCH**
- **Justification:** `src/collect_hard_examples.py` runs validation inference and dumps images with `confidence < 0.3` exactly as the methodology requires for Stage 3 training.

### Training & Experiments
- **Status:** **PARTIAL**
- **Justification:** The paper outlines YOLOv5s vs YOLOv8l. YOLOv8l stages are complete in `runs/`. However, YOLOv5s is currently incomplete (Epoch 10 of 50 in `experiments/yolov5/train_yolov5s_v1`). 
- **Fix:** Wait for the orchestrator to finish the YOLOv5s training run.

### Results & Ablation Studies
- **Status:** **MISSING**
- **Justification:** The paper claims to compare YOLOv5m, YOLOv7x, YOLOv8s, and YOLOv8m. These architectures have absolutely zero files, weights, or runs in the repository.
- **Fix:** Either edit the paper to remove these claims OR write training orchestrator scripts to explicitly train and log these 4 models.

### Discussion: Real-Time Viability
- **Status:** **PARTIAL**
- **Justification:** `src/main.py` runs in real-time and shows FPS, but no standardized benchmark table of Latency across models exists to prove the academic claims statistically.
- **Fix:** Create a `scripts/benchmark.py` that loops a validation video 10 times and logs average FP16 latency.

### Conclusion & Deployment Claims
- **Status:** **MISSING**
- **Justification:** The paper claims this is "ready for smart city edge deployment." The lack of Docker containers, CI/CD, or Edge TPU / TensorRT export scripts contradicts this claim.
- **Fix:** Write a `Dockerfile` and `docker-compose.yml`. Export the final `object_stage4_riderfix4` weights to ONNX/TensorRT format.
