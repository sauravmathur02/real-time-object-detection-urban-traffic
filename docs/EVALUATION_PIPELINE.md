# Unified Evaluation Pipeline

The `scripts/evaluate_models.py` orchestrator provides a single, unified entry point for validating all trained YOLO models in this repository. It evaluates their performance across our custom balanced dataset (`bdd_balanced.yaml`) without requiring manual environment switching.

## Supported Frameworks & Models
The orchestrator dynamically maps models to their native validation engines:
- **YOLOv5** (`val.py`): `yolov5s`, `yolov5m`
- **YOLOv7** (`test.py`): `yolov7x`
- **YOLOv8** (`ultralytics API`): `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`

## CLI Examples

Evaluate a single specific model pointing directly to its best checkpoint:
```bash
python scripts/evaluate_models.py --model yolov8l --weights experiments/yolov8/train_yolov8l/weights/best.pt
```

Run a complete automated validation sweep across all supported architectures (skipping gracefully if weights are missing):
```bash
python scripts/evaluate_models.py --all
```

Evaluate with custom hardware and batching constraints:
```bash
python scripts/evaluate_models.py --model yolov5s --device 0 --batch 32 --workers 8 --save-json
```

## Output Structure
To prevent cluttering the root training directories, evaluation outputs are stored in a dedicated directory per model:
```text
object-Detection/
`-- evaluation/
    |-- yolov5s/
    |   |-- metrics.json
    |   |-- metrics.csv
    |   `-- metrics.md
    |-- yolov8l/
    |   |-- metrics.json
    |   |-- metrics.csv
    |   `-- metrics.md
    `-- master_metrics.csv
```

## Generated Metrics
The script programmatically parses framework outputs (avoiding hardcoded metrics) and aggregates the following details:
- **Precision**
- **Recall**
- **mAP50**
- **mAP50-95**
- **Inference Time (ms/img)**

*(Note: Validation loss and per-class AP are dynamically logged via the underlying framework plots/console outputs).*

## Generated Plots
While the orchestrator standardizes the raw metrics into CSV and JSON files, the native framework scripts concurrently generate the following visual plots within the `evaluation/<model_name>` directories:
- Confusion Matrix (`confusion_matrix.png`)
- Precision-Recall Curve (`PR_curve.png`)
- F1-Confidence Curve (`F1_curve.png`)
- Batch Predictions (`val_batchX_pred.jpg`)
