# Pipeline Validation Report

This document contains a statically analyzed end-to-end audit of the entire orchestration pipeline developed in `scripts/`, evaluating inter-module consistency, CLI parsing, paths, and dependencies.

## 1. Inconsistent CLI Argument Naming (YAML Dataset)
- **Severity:** Low
- **Problem:** The CLI flag to specify the dataset YAML file is inconsistent across the pipeline. `train_all_models.py` and `dataset_statistics.py` use `--dataset`, while `evaluate_models.py` uses `--data`. 
- **Reason:** The inconsistency stems from mapping to the underlying YOLO framework flags (YOLOv5 often uses `--data` in its `val.py`).
- **Affected files:** `scripts/evaluate_models.py`, `scripts/train_all_models.py`, `scripts/dataset_statistics.py`
- **Recommended fix:** Standardize the orchestrator layer to expose `--dataset` globally to the user, and handle the mapping to `--data` internally during the subprocess framework call in `evaluate_models.py`.

## 2. Qualitative Inference Disconnect
- **Severity:** Medium
- **Problem:** `qualitative_analysis.py` scans `sample_images/` and inherently expects to find prediction label files nested inside `sample_images/predictions/<model>/labels/`. However, the current `evaluate_models.py` executes validation (`val.py`), not raw inference (`predict.py`).
- **Reason:** There is no dedicated `scripts/predict_models.py` orchestrator built in the current task list to specifically generate these text labels on unannotated qualitative images.
- **Affected files:** `scripts/qualitative_analysis.py`
- **Recommended fix:** Build a `predict_models.py` orchestrator that explicitly runs inference on `--img_dir` and routes the `.txt` outputs directly into the expected `predictions/<model>/labels/` directory for the qualitative script to parse.

## 3. Missing Per-Class Metrics in Legacy Frameworks
- **Severity:** Medium
- **Problem:** `generate_tables.py` (Table V) and `per_class_analysis.py` expect a rich `metrics.json` file inside `evaluation/<model>/` containing a `per_class` block. YOLOv8 seamlessly provides this, but the legacy YOLOv5/v7 validation frameworks primarily dump to `results.txt` via stdout without natively generating this JSON structure unless deeply modified.
- **Reason:** Constrained by the rule "Do NOT modify existing evaluation code", the orchestrator relies on best-effort parsing of `results.txt`, which usually only captures aggregate mAP, not per-class splits natively in JSON format.
- **Affected files:** `scripts/generate_tables.py`, `scripts/per_class_analysis.py`, `scripts/evaluate_models.py`
- **Recommended fix:** Enhance the `parse_yolov5_v7_results()` function in `evaluate_models.py` to aggressively scrape the per-class tabular printout from the `stdout` buffer and manually construct the `metrics.json` file.

## 4. Disk Space Overflow in Ablation
- **Severity:** High
- **Problem:** `ablation_runner.py` physically duplicates the entire dataset pipeline for every run (routing to `data1/ablation/<experiment_name>`) to prevent baseline corruption.
- **Reason:** Built to strictly adhere to the "Never modify original datasets" constraint.
- **Affected files:** `scripts/ablation_runner.py`
- **Recommended fix:** Implement a dynamic symlink architecture or custom dataset loading script that filters classes/images at runtime in memory rather than duplicating physical disk files.

## Summary of Success
Beyond these integration edge cases, the **CLI, argparse, module imports, and cross-platform relative pathlib usage** are fully functional and correctly chained. Absolute paths are successfully avoided via `.resolve().parents[1]` logic, ensuring the repository is portable across Windows and Linux environments.
