# Advanced YOLO Training Pipeline

## 1. Automated Mining & Balancing Scripts

The required data curation scripts live in `src/`. Run these commands from the repository root.

1. `src/verify_dataset.py`: validates dataset alignment and checks for empty labels or orphaned images without modifying source data.
2. `src/oversample_rider.py`: creates a balanced target dataset in `data1/bdd_balanced` by duplicating images that contain the underrepresented rider class.
3. `src/collect_hard_examples.py`: runs a stream-efficient inference loop on the validation dataset to mine missed detections and low-confidence predictions.

## 2. Advanced YOLOv8 Training Command

Paste this directly into PowerShell after preparing the dataset:

```powershell
yolo detect train model=runs/detect/train/weights/best.pt data=data1/bdd100k.yaml epochs=50 imgsz=960 batch=-1 multi_scale=True mosaic=1.0 mixup=0.15 copy_paste=0.2 cos_lr=True warmup_epochs=5 weight_decay=0.0007 name=object_elite_stage
```

## 3. Professional Resume Explanation

You can use the following paragraph to describe the engineering work:

> Engineered a comprehensive real-time urban object detection pipeline utilizing YOLOv8, processing a 10K frame subset of the BDD100K dataset via a custom curation workflow. Addressed class imbalance by building automated tools to oversample minority classes and improve sensitivity to vulnerable road users. Devised a hard-example mining stage to isolate low-confidence inferences and false negatives, feeding high-value edge cases back into the training loop. Led a multi-stage fine-tuning strategy with dynamic augmentation, multi-scale training, and hyperparameter tuning to improve detection robustness in dense urban scenes.
