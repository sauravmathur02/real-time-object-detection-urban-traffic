# Unified Multi-Model Training Pipeline

The `scripts/train_all_models.py` orchestrator serves to unify and automate the training of multiple YOLO architectures across different underlying frameworks, exactly replicating the hyperparameters and methodologies specified in the research paper.

## Architecture
The orchestrator acts as a dispatch layer. Instead of reinventing the training loops, it safely delegates execution directly to the native training engines provided by the respective frameworks:
- **YOLOv5** runs via local `frameworks/yolov5_src/train.py`
- **YOLOv7** runs via local `frameworks/yolov7/train.py`
- **YOLOv8** runs natively via the `ultralytics` Python package API

## Supported Models
The orchestrator explicitly supports the following architectural variants evaluated in the paper:
- `yolov5s`, `yolov5m`
- `yolov7x`
- `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`

## Folder Layout
To maintain strict segregation of experiment logs and prevent framework collisions, the orchestrator routes outputs into specific directories based on the model family:
```text
object-Detection/
|-- experiments/
|   |-- yolov5/       # All YOLOv5 runs
|   `-- yolov7/       # All YOLOv7 runs
`-- runs/
    `-- detect/       # All native Ultralytics YOLOv8 runs
```

## CLI Examples

Train a specific model using default paper hyperparameters:
```bash
python scripts/train_all_models.py --model yolov8l
```

Run a complete automated sweep across all supported architectures sequentially:
```bash
python scripts/train_all_models.py --all
```

Override default hyperparameters (e.g., for quick debugging or ablation studies):
```bash
python scripts/train_all_models.py --model yolov5s --epochs 10 --batch 32 --lr0 0.001 --optimizer Adam
```

Resume training on a specific device using custom dataloader workers:
```bash
python scripts/train_all_models.py --model yolov8l --resume --device 0,1 --workers 8
```

## Output Structure
Upon completion of a successful run, the orchestrator guarantees that the framework has generated the following core artifacts within the target experiment folder:
1. `weights/best.pt` and `weights/last.pt` (The trained weights)
2. `results.csv` (The empirical metric logs per epoch)
3. `args.yaml` or `opt.yaml` (The frozen hyperparameters for reproducibility)
4. Assorted validation plots (PR curves, confusion matrices, batch renderings)
