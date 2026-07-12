# Reproducibility Guide

This guide ensures that the experiments, metrics, and models presented in the research paper can be exactly replicated by independent reviewers.

## Hardware & Environment
- **Hardware Used:** Single GPU Workstation (e.g., NVIDIA RTX 3090 / 4090)
- **OS:** Windows / Ubuntu Linux
- **Python Version:** 3.9+
- **CUDA Version:** 11.8+ (Dependent on PyTorch build)
- **PyTorch Version:** 2.0+
- **Ultralytics Version:** 8.0+
- **Random Seed:** Default Ultralytics seeds (`0`) are used throughout to ensure deterministic behavior.

## Dataset Preparation Workflow
The dataset pipeline must be executed in a strict sequence:
1. **Convert BDD100K:** `python src/bdd_to_yolo_prod.py --json_train ... --output_dir data1/bdd_yolo`
2. **Merge IDD:** `python src/merge_idd.py --idd-dir ... --bdd-dir data1/bdd_balanced`
3. **Merge Auto-rickshaws:** `python src/convert_auto_to_yolo.py ...` followed by `src/merge_auto_yolo.py`
4. **Oversample Riders:** `python src/oversample_rider.py --src_dir ... --dst_dir data1/bdd_balanced`

## Training Workflow
Training utilizes a 4-stage pipeline orchestrated natively by the Ultralytics framework.
1. Use `yolov8l.pt` (COCO base) and train on `bdd_yolo`.
2. Extract the `best.pt` from Stage 1, use as the base for Stage 2.
3. Run `src/collect_hard_examples.py` to isolate difficult frames. Train Stage 3 on this subset.
4. Train Stage 4 using the `bdd_balanced.yaml` to address regional Indian vehicles.

*(Alternatively, use `scripts/train_orchestrator.py` for automated multi-model sweeps).*

## Evaluation Workflow
Evaluation metrics are logged via the Ultralytics standard CSV artifacts. 
To aggregate all repository experiments into a unified CSV and Markdown table, run:
```bash
python scripts/experiment_manager.py
```
This extracts mAP50, Precision, Recall, and Hyperparameters.

## Benchmark Workflow
To reproduce latency, FPS, and model size metrics natively on your hardware:
```bash
python scripts/benchmark.py --models yolov5s=runs/yolov5s.pt yolov8l=runs/yolov8l.pt --source data1/bdd_yolo/images/val --output evaluation
```

## Qualitative Comparison Workflow
To generate side-by-side visual renderings of bounding box predictions for paper figures:
```bash
python scripts/generate_qualitative_figs.py --images sample_images/ --models yolov8l=runs/best.pt --output results/qualitative/
```
