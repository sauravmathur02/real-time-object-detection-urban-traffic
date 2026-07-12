# Experiments Management

This document tracks all historical and active object detection training runs in the repository.

## Repository Experiment Overview
The repository iteratively trains YOLO models using a multi-stage approach. Early stages act as pre-training on base datasets, while later stages focus on mined hard examples and heavily balanced custom datasets.

## Training Progression
1. **Stage1:** Initial transfer learning from COCO.
2. **Stage2:** Extended fine-tuning on the base dataset.
3. **Stage3:** Training exclusively on mined hard examples (false negatives and low confidence detections).
4. **Stage4:** Final tuning on the fully merged, auto-rickshaw integrated, and rider-balanced dataset.

## Reproducibility Instructions
To reproduce any experiment, locate its `args.yaml` or `opt.yaml` in its respective run directory, and supply those arguments to the Ultralytics CLI or the YOLOv5 train orchestrator.

## Final Production Model
- **object_stage4_riderfix2**: Dataset: bdd_balanced, mAP50: 0.3133
- **object_stage4_riderfix4**: Dataset: bdd_balanced, mAP50: 0.4618

## Baselines
- **object_stage1**: Dataset: bdd100k, mAP50: 0.4106
- **object_stage2**: Dataset: bdd100k, mAP50: 0.3901
- **object_stage3_strong2**: Dataset: bdd_no_train, mAP50: 0.4732

## Experimental Runs
- **train5**: Dataset: bdd100k, mAP50: 0.1601
- **train7**: Dataset: bdd_balanced, mAP50: 0.4978
- **train_auto_v1**: Dataset: bdd_balanced, mAP50: 0.4828
- **train_fast2**: Dataset: bdd_balanced, mAP50: 0.5524
- **train_yolov8l_high**: Dataset: bdd100k, mAP50: 0.192
- **train_yolov8l_high2**: Dataset: bdd100k, mAP50: 0.1901
- **train_yolov5s_v1**: Dataset: bdd_balanced, mAP50: 0.5244

## Archived Experiments
- **train3**
- **train4**
- **train6**
- **validation_yolov5s**
