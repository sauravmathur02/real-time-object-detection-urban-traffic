# Experiment Reproducibility

An analysis of whether an independent researcher could clone this repository and perfectly reproduce the training metrics.

### YOLOv8 Runs (`runs/detect/`)
- **Dataset:** `bdd100k.yaml`, `bdd_balanced.yaml`
- **Optimizer:** `auto` (Ultralytics default)
- **Batch:** Varies (8, 16, 32)
- **Seed:** Not explicitly set in any tracked configuration (Defaults to Ultralytics 0).
- **Hyperparameters:** Handled entirely by Ultralytics defaults. 
- **Can another researcher reproduce?** **PARTIAL.** 
- **Why?** Since YOLOv8 training was executed via the CLI (e.g., `yolo detect train model=yolov8l.pt`), the exact commands used are not preserved in a Makefile or shell script, except as reconstructed from `args.yaml`. Due to missing explicit seed pinning and CUDA determinism settings, exact mAP numbers may fluctuate slightly (±0.02) on different hardware.

### YOLOv5s Run (`experiments/yolov5/train_yolov5s_v1/`)
- **Dataset:** `bdd_balanced.yaml`
- **Optimizer:** `SGD`
- **Batch:** 16
- **Seed:** 0 (Pinned via `opt.yaml`)
- **Weights:** `yolov5s.pt`
- **Can another researcher reproduce?** **YES.** 
- **Why?** The introduction of `scripts/train_orchestrator.py` systematically tracks the Git commit, Python/Torch/CUDA versions, and creates a highly detailed `environment.json` and `experiment.yaml`. This guarantees hardware and software state tracking, allowing for highly deterministic reproducibility.
