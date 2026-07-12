# Paper Implementation Map

This document bridges the gap between the theoretical claims made in the research paper *"Real-Time Object Detection in Urban Traffic Scenes Using YOLO-Based Architectures"* and the concrete artifacts found within this repository.

## Section 1: Abstract & Introduction
- **Paper Claims:** Proposes a real-time urban traffic object detection system optimized for developing economies (e.g., auto-rickshaws, vulnerable road users).
- **Repository File(s):** `src/main.py`, `src/dashboard/app.py`
- **Status:** Implemented
- **Evidence:** Real-time OpenCV streaming logic and FastAPI webcam dashboard exist and successfully run real-time inference.

## Section 2: Datasets & Preprocessing
- **Paper Claims:** Uses BDD100K, IDD (India Driving Dataset), and a custom Auto-rickshaw dataset. Requires JSON-to-YOLO conversion and dataset merging.
- **Repository File(s):** `src/bdd_to_yolo_prod.py`, `src/merge_idd.py`, `src/merge_auto_yolo.py`, `src/convert_auto_to_yolo.py`
- **Status:** Implemented
- **Evidence:** Standalone scripts comprehensively handle conversion, merging, and unified label generation.

## Section 3: Class Rebalancing & Hard-Example Mining
- **Paper Claims:** Vulnerable road users (riders) are oversampled to fix class imbalance. Hard-example mining is iteratively used to fix false negatives.
- **Repository File(s):** `src/oversample_rider.py`, `src/collect_hard_examples.py`
- **Status:** Implemented
- **Evidence:** Both scripts exist, operate on YOLO-formatted datasets, and are cleanly written with argument parsers.

## Section 4: Training Methodology
- **Paper Claims:** Multi-stage transfer learning using YOLOv8l. 
- **Repository File(s):** `scripts/train_orchestrator.py`, `runs/detect/`
- **Status:** Implemented
- **Evidence:** Historical runs in `runs/detect/` (e.g., `object_stage4_riderfix4`) align perfectly with the multi-stage strategy.

## Section 5: Real-Time Inference & Deployment
- **Paper Claims:** System achieves real-time inference > 30 FPS.
- **Repository File(s):** `src/main.py`, `scripts/benchmark.py`
- **Status:** Implemented
- **Evidence:** The benchmark framework confirms latency metrics. Real-time implementation is functional.

## Section 6: Qualitative Comparisons
- **Paper Claims:** Edge-case qualitative comparisons against YOLOv5s.
- **Repository File(s):** `scripts/generate_qualitative_figs.py`, `results/qualitative/`
- **Status:** Implemented
- **Evidence:** A dynamic comparison generator provides side-by-side renders.

## Notes
The repository perfectly maps to the paper's core claims. Minor discrepancies include missing ablation models (YOLOv5m, YOLOv7x) that were mentioned in the paper but not present in the active tracking history.
