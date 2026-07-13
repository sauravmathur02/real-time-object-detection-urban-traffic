# Pre-Training Validation Report

This document serves as the final clearance audit before the commencement of heavy GPU training. Every script in the orchestration pipeline has been statically analyzed for correct dependency chaining, CLI parsing, absolute pathing, and inter-module compatibility.

## SECTION 1: Repository Status
**Ready**

## SECTION 2: Blocking Issues
- **Critical:** None. The repository architecture is fundamentally sound.
- **High:** None.
- **Medium:** None.
- **Low:** The repository assumes `ultralytics` is pip-installed and that YOLOv5/YOLOv7 source codes are cloned directly into `frameworks/yolov5/` and `frameworks/yolov7/`. If these are missing, the orchestrators will gracefully skip them but fail to train the full paper subset.

## SECTION 3: Manual Commands
Execute the following commands sequentially from the repository root:

1. **Dataset Preparation**
```bash
python scripts/prepare_dataset.py --step all
```

2. **Dataset Statistics Generation**
```bash
python scripts/dataset_statistics.py --dataset data1/bdd_balanced.yaml
```

3. **Multi-Model Training** (Example for all models)
```bash
python scripts/train_all_models.py --all --dataset data1/bdd_balanced.yaml --epochs 50 --batch 16
```

4. **Multi-Model Evaluation**
```bash
python scripts/evaluate_models.py --all --dataset data1/bdd_balanced.yaml
```

5. **Multi-Model Prediction (Inference)**
```bash
python scripts/predict_models.py --all --source sample_images
```

6. **Automated Table Generation**
```bash
python scripts/generate_tables.py
```

7. **Automated Figure Generation**
```bash
python scripts/generate_figures.py
```

8. **Per-Class Analysis**
```bash
python scripts/per_class_analysis.py
```

9. **Qualitative Analysis**
```bash
python scripts/qualitative_analysis.py
```

10. **Ablation Setup** (Optional, configure as needed)
```bash
python scripts/ablation_runner.py --disable-rider-balance
```

## SECTION 4: Expected Outputs

1. **Dataset Preparation:** Generates `data1/bdd_balanced.yaml` and formatted image/label directories.
2. **Dataset Statistics:** Populates the `dataset_statistics/` directory with `dataset_statistics.csv`, `.json`, `.md`, and distribution `.png` charts.
3. **Training:** Populates the `experiments/` directory with subfolders for each model (e.g., `experiments/train_yolov8l/weights/best.pt`).
4. **Evaluation:** Populates the `evaluation/` directory with `master_metrics.csv` and individual model folders containing `metrics.json` and `results.txt`.
5. **Prediction:** Populates `sample_images/predictions/<model>/labels/` with raw bounding box inference text files.
6. **Table Generation:** Populates `paper_outputs/` with `Table_IV_Model_Comparison` and `Table_V_Per_Class_AP` in CSV and Markdown formats.
7. **Figure Generation:** Populates `paper_outputs/figures/` with `Figure_1_System_Pipeline.png`, `Figure_2_Model_Comparison.png`, and `Figure_3_PR_Curve.png`.
8. **Per-Class Analysis:** Populates `paper_outputs/per_class/` with detailed `.md` reports and horizontal bar charts ranking class performance.
9. **Qualitative Analysis:** Populates `paper_outputs/qualitative/` with `comparison_matrix.csv`, `QUALITATIVE_ANALYSIS.md`, and physically sorts edge-case imagery into `best_examples/` and `failure_cases/`.
10. **Ablation Runner:** Populates `ablation/experiment_configs/` with isolated YOLO yaml files (e.g., `no_rider_dataset.yaml`) and metadata JSONs.

## SECTION 5: Common Failure Points

- **Missing Base Datasets:** `prepare_dataset.py` relies on BDD100K, IDD, and Datacluster datasets being physically present at expected relative targets. **Fix:** Ensure raw datasets are downloaded and unzipped into their default directory arguments.
- **CUDA Out of Memory (OOM):** Training YOLOv7x or YOLOv8l on smaller GPUs (e.g., 8GB VRAM) may crash during `train_all_models.py`. **Fix:** Manually throttle the batch size via `--batch 8` or `--batch 4` in the CLI command.
- **Missing Legacy Frameworks:** `evaluate_models.py` and `train_all_models.py` will log warnings if `frameworks/yolov5/` is missing. **Fix:** Ensure `git clone https://github.com/ultralytics/yolov5` is run inside the `frameworks/` folder.
- **Empty Qualitative Folders:** `qualitative_analysis.py` will output empty data if no images exist in `sample_images/`. **Fix:** Ensure test imagery is populated in `sample_images/` before running inference.

## SECTION 6: Final Verdict

READY FOR TRAINING
