# Master Experiment Results

All completed experiments, sorted by mAP50.

| Experiment Name | Category | Model | Dataset YAML | Dataset Name | Epochs | Image Size | Batch Size | Optimizer | Learning Rate | Device | Training Date | Best Checkpoint | Last Checkpoint | Precision | Recall | mAP50 | mAP50-95 | Fitness | Training Time | Number of Parameters | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_fast2 | Experimental | runs/detect/object_stage3_strong2/weights/best.pt | data1/bdd_balanced.yaml | bdd_balanced | 30 | 512 | 32 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.7147 | 0.5065 | 0.5524 | 0.348 | 0 | Unknown | Unknown |  |
| train_yolov5s_v1 | Experimental | experiments\yolov5\train_yolov5s_v1\weights\last.pt | C:\Repo\object-Detection\data1\bdd_balanced.yaml | bdd_balanced | 50 | 640 | 16 | SGD | Unknown | 0 | Unknown | Yes | Yes | 0.7065 | 0.4749 | 0.5244 | 0.3094 | 0 | Unknown | Unknown |  |
| train7 | Experimental | runs/detect/object_stage3_strong2/weights/best.pt | data1/bdd_balanced.yaml | bdd_balanced | 50 | 960 | 16 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.6102 | 0.4805 | 0.4978 | 0.3018 | 0 | Unknown | Unknown |  |
| train_auto_v1 | Experimental | yolov8l.pt | c:\Repo\object-Detection\data1\bdd_balanced.yaml | bdd_balanced | 50 | 640 | 4 | auto | 0.01 | None | Unknown | Yes | Yes | 0.5554 | 0.4742 | 0.4828 | 0.2952 | 0 | Unknown | Unknown |  |
| object_stage3_strong2 | Baseline | runs/detect/object_stage2/weights/best.pt | data1/bdd_no_train.yaml | bdd_no_train | 50 | 960 | 6 | auto | 0.005 | 0 | Unknown | Yes | Yes | 0.6116 | 0.4538 | 0.4732 | 0.2552 | 0 | Unknown | Unknown |  |
| object_stage4_riderfix4 | Final | runs/detect/object_stage3_strong2/weights/best.pt | data1/bdd_balanced.yaml | bdd_balanced | 50 | 960 | 6 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.5998 | 0.4463 | 0.4618 | 0.2572 | 0 | Unknown | Unknown |  |
| object_stage1 | Baseline | yolov8l.pt | data1/bdd100k.yaml | bdd100k | 30 | 832 | 8 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.6443 | 0.4012 | 0.4106 | 0.2256 | 0 | Unknown | Unknown |  |
| train6 | Archived | yolov8l.pt | data1/bdd100k.yaml | bdd100k | 50 | 832 | 8 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.6526 | 0.3983 | 0.4086 | 0.2244 | 0 | Unknown | Unknown |  |
| object_stage2 | Baseline | runs/detect/object_stage1/weights/best.pt | data1/bdd100k.yaml | bdd100k | 70 | 832 | 8 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.5058 | 0.3834 | 0.3901 | 0.2121 | 0 | Unknown | Unknown |  |
| object_stage4_riderfix2 | Final | runs/detect/object_stage3_strong2/weights/best.pt | data1/bdd_balanced.yaml | bdd_balanced | 50 | 960 | 6 | auto | 0.005 | 0 | Unknown | Yes | Yes | 0.4553 | 0.3289 | 0.3133 | 0.1806 | 0 | Unknown | Unknown |  |
| validation_yolov5s | Archived | C:\Repo\object-Detection\yolov5s.pt | C:\Repo\object-Detection\data1\bdd_balanced.yaml | bdd_balanced | 1 | 640 | 16 | SGD | Unknown | 0 | Unknown | Yes | Yes | 0.5072 | 0.2769 | 0.2726 | 0.1494 | 0 | Unknown | Unknown |  |
| train3 | Archived | yolov8n.pt | data/bdd100k.yaml | bdd100k | 10 | 640 | 16 | auto | 0.01 | cpu | Unknown | Yes | Yes | 0.4049 | 0.0018 | 0.2034 | 0.122 | 0 | Unknown | Unknown |  |
| train_yolov8l_high | Experimental | yolov8l.pt | data/bdd100k.yaml | bdd100k | 100 | 832 | 8 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.3675 | 0.0171 | 0.192 | 0.1171 | 0 | Unknown | Unknown |  |
| train_yolov8l_high2 | Experimental | yolov8l.pt | data/bdd100k.yaml | bdd100k | 100 | 832 | 8 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.3682 | 0.0119 | 0.1901 | 0.1159 | 0 | Unknown | Unknown |  |
| train5 | Experimental | yolov8n.pt | data1/bdd100k.yaml | bdd100k | 2 | 640 | 16 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.5697 | 0.1809 | 0.1601 | 0.089 | 0 | Unknown | Unknown |  |
| train4 | Archived | yolov8n.pt | data/bdd100k.yaml | bdd100k | 1 | 640 | 16 | auto | 0.01 | 0 | Unknown | Yes | Yes | 0.0 | 0.0 | 0.0 | 0.0 | 0 | Unknown | Unknown |  |
