# Moved Files Tracking

The following table lists the physical location changes for all files during the repository restructuring.

| Original File Path | New File Path | Action Type | Reason |
| :--- | :--- | :---: | :--- |
| `oversample_rider.py` | `archive/legacy_scripts/oversample_rider.py` | MOVE | Hardcoded duplicate of `src/oversample_rider.py`. |
| `analyze_bdd.py` | `archive/legacy_scripts/analyze_bdd.py` | MOVE | Legacy duplicate. |
| `clean_balance_bdd.py` | `archive/legacy_scripts/clean_balance_bdd.py` | MOVE | Legacy duplicate. |
| `count_rider_images.py` | `archive/obsolete/count_rider_images.py` | MOVE | One-off obsolete script. |
| `downsample_bdd.py` | `archive/obsolete/downsample_bdd.py` | MOVE | Obsolete processing step. |
| `fix_labels.py` | `archive/obsolete/fix_labels.py` | MOVE | Obsolete manual fix replaced by `merge_idd.py`. |
| `remove_train_class.py`| `archive/obsolete/remove_train_class.py` | MOVE | Obsolete functionality replaced by mapping logic. |
| `scripts/train_yolov5s.py`| `archive/obsolete/train_yolov5s.py` | MOVE | Obsolete; `train_orchestrator.py` now runs YOLOv5 natively. |
| `CODE_ARCHITECTURE.md` | `docs/CODE_ARCHITECTURE.md` | MOVE | Documentation consolidation. |
| `PIPELINE_MAPPING.md` | `docs/PIPELINE_MAPPING.md` | MOVE | Documentation consolidation. |
| `CALL_GRAPH.md` | `docs/CALL_GRAPH.md` | MOVE | Documentation consolidation. |
| `UNUSED_CODE.md` | `docs/UNUSED_CODE.md` | MOVE | Documentation consolidation. |
| `DUPLICATE_IMPLEMENTATIONS.md` | `docs/DUPLICATE_IMPLEMENTATIONS.md` | MOVE | Documentation consolidation. |
| `EXPERIMENT_REPRODUCIBILITY.md`| `docs/EXPERIMENT_REPRODUCIBILITY.md` | MOVE | Documentation consolidation. |
| `DATA_FLOW.md` | `docs/DATA_FLOW.md` | MOVE | Documentation consolidation. |
| `MODEL_FLOW.md` | `docs/MODEL_FLOW.md` | MOVE | Documentation consolidation. |
| `CODE_QUALITY.md` | `docs/CODE_QUALITY.md` | MOVE | Documentation consolidation. |
| `PERFORMANCE_ANALYSIS.md`| `docs/PERFORMANCE_ANALYSIS.md` | MOVE | Documentation consolidation. |
| `RESEARCH_ALIGNMENT.md` | `docs/RESEARCH_ALIGNMENT.md` | MOVE | Documentation consolidation. |
| `PROJECT_SUMMARY.md` | `docs/reports/PROJECT_SUMMARY.md` | MOVE | Documentation consolidation. |
| `DATASET_REPORT.md` | `docs/reports/DATASET_REPORT.md` | MOVE | Documentation consolidation. |
| `PAPER_MATCH_REPORT.md` | `docs/reports/PAPER_MATCH_REPORT.md` | MOVE | Documentation consolidation. |
| `MODEL_INVENTORY.md` | `docs/reports/MODEL_INVENTORY.md` | MOVE | Documentation consolidation. |
| `FIGURE_REPORT.md` | `docs/reports/FIGURE_REPORT.md` | MOVE | Documentation consolidation. |
| `TRAINING_HISTORY.md` | `docs/reports/TRAINING_HISTORY.md` | MOVE | Documentation consolidation. |
| `REPOSITORY_TREE.md` | `docs/reports/REPOSITORY_TREE.md` | MOVE | Documentation consolidation. |
| `REVIEWER_REQUIREMENTS.md`| `docs/reports/REVIEWER_REQUIREMENTS.md` | MOVE | Documentation consolidation. |
| `REVIEWER_TRACEABILITY.md`| `docs/reports/REVIEWER_TRACEABILITY.md` | MOVE | Documentation consolidation. |
| `MISSING_FILES.md` | `docs/reports/MISSING_FILES.md` | MOVE | Documentation consolidation. |
| `FINAL_SCORE.md` | `docs/reports/FINAL_SCORE.md` | MOVE | Documentation consolidation. |
| `FINAL_ACTION_PLAN.md` | `docs/reports/FINAL_ACTION_PLAN.md` | MOVE | Documentation consolidation. |
| `Project_Validation_Report.md`| `docs/reports/Project_Validation_Report.md` | MOVE | Internal audit documentation. |
| `Project_Validation_Report.pdf`| `docs/reports/Project_Validation_Report.pdf` | MOVE | Internal audit documentation. |
| `runs_inspection_report.json`| `docs/reports/runs_inspection_report.json` | MOVE | Internal audit documentation. |
| `runs/detect/object_stage4_riderfix4/confusion_matrix.png`| `results/figures/confusion_matrix.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `runs/detect/object_stage4_riderfix4/BoxPR_curve.png` | `results/figures/BoxPR_curve.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `runs/detect/object_stage4_riderfix4/BoxP_curve.png` | `results/figures/BoxP_curve.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `runs/detect/object_stage4_riderfix4/BoxR_curve.png` | `results/figures/BoxR_curve.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `runs/detect/object_stage4_riderfix4/BoxF1_curve.png` | `results/figures/BoxF1_curve.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `runs/detect/object_stage4_riderfix4/results.png` | `results/figures/results.png` | **COPY** | Extracted best result figures for reviewer visibility. |
| `output.mp4` | `results/demo.mp4` | **COPY** | Saved visualization demo. |
