# Comprehensive Evaluation Report: YOLOv8l & YOLOv7 Benchmarks

**Project:** Real-Time Object Detection in Urban Traffic Scenes  
**Dataset:** BDD100k / Balanced Urban Traffic Dataset (`bdd_balanced`)  
**Report Date:** July 19, 2026  

---

## 1. Executive Summary

This report presents a detailed empirical analysis of **YOLOv8l (YOLOv8 Large)** and recently validated **YOLOv7** models trained on urban traffic detection tasks. 

* **Top Performer Overall:** **YOLOv8l (`train_fast2`)** achieved the highest accuracy across the entire repository with **`0.5524 mAP@0.50`** and **`0.3480 mAP@0.50:0.95`**.
* **Key Finding:** YOLOv8l performance is highly sensitive to **batch size**. Training with a batch size of 32 at 512px significantly outperformed training with a batch size of 6 at 960px resolution (+9% mAP boost).
* **YOLOv7 Validation Benchmark:** The recently completed YOLOv7 training run achieved **`0.5140 mAP@0.50`** and **`0.3305 mAP@0.50:0.95`**, placing it 2nd overall in strict bounding box localization precision.

---

## 2. YOLOv8l Architecture & Model Specifications

| Parameter | Value / Detail |
| :--- | :--- |
| **Model Family** | Ultralytics YOLOv8 |
| **Model Variant** | YOLOv8 Large (`YOLOv8l`) |
| **Pretrained Checkpoint** | `yolov8l.pt` (87.8 MB) |
| **Parameter Count** | **43.64 Million** (43,643,307) |
| **Model Disk Size** | **83.6 MB** |
| **GFLOPs (at 640x640)** | **165.2 GFLOPs** |
| **Backbone & Neck** | CSP-based backbone with C2f modules for enhanced gradient propagation |
| **Head Architecture** | Anchor-free, Decoupled Classification and Bounding Box Regression Heads |
| **Loss Functions** | CIoU + Distribution Focal Loss (DFL) for box regression, BCE for classification |

---

## 3. YOLOv8l Experiment Comparison Table

Below is the complete performance breakdown across all YOLOv8l training runs in the project repository:

| Experiment Name | Dataset Config | Image Size | Batch Size | Epochs | Precision | Recall | mAP @ 0.50 | mAP @ 0.50:0.95 | Performance Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train_fast2`** | `bdd_balanced` | 512x512 | 32 | 30 | **`0.7147`** | **`0.5065`** | **`0.5524`** | **`0.3480`** | 🥇 **Rank 1 Overall (Best in Repo)** |
| **`train7`** | `bdd_balanced` | 960x960 | 16 | 50 | `0.6102` | `0.4805` | `0.4978` | `0.3018` | 🥈 Rank 2 YOLOv8l Run |
| **`train_auto_v1`** | `bdd_balanced` | 640x640 | 4 | 50 | `0.5554` | `0.4742` | `0.4828` | `0.2952` | Base 640px run |
| **`object_stage3_strong2`** | `bdd_no_train` | 960x960 | 6 | 50 | `0.6116` | `0.4538` | `0.4732` | `0.2552` | Baseline Stage 3 |
| **`object_stage4_riderfix4`** | `bdd_balanced` | 960x960 | 6 | 50 | `0.5998` | `0.4463` | `0.4618` | `0.2572` | Stage 4 Final Run |
| **`object_stage1`** | `bdd100k` | 832x832 | 8 | 30 | `0.6443` | `0.4012` | `0.4106` | `0.2256` | Early Baseline |
| **`object_stage2`** | `bdd100k` | 832x832 | 8 | 70 | `0.5058` | `0.3834` | `0.3901` | `0.2121` | Iterative Training |

---

## 4. Hyperparameter Setup (`train_fast2`)

```yaml
task: detect
mode: train
model: yolov8l.pt
data: data1/bdd_balanced.yaml
epochs: 30
batch: 32
imgsz: 512
optimizer: auto (SGD)
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
box_loss_weight: 7.5
cls_loss_weight: 0.5
dfl_loss_weight: 1.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0 (disabled during final 10 epochs)
```

---

## 5. Comparative Evaluation Across All Architectures

| Architecture | Model Variant | Image Size | Batch Size | Precision | Recall | mAP @ 0.50 | mAP @ 0.50:0.95 | Parameters (M) | Model Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8** | **YOLOv8l (`train_fast2`)** | 512x512 | 32 | **0.7147** | **0.5065** | **0.5524** | **0.3480** | 43.64M | 83.6 MB |
| **YOLOv5** | **YOLOv5s (`v1`)** | 640x640 | 16 | 0.7065 | 0.4749 | 0.5244 | 0.3094 | 7.05M | 13.8 MB |
| **YOLOv7** | **YOLOv7 (`train_yolov7`)** | 640x640 | 16 | 0.6401 | 0.4932 | 0.5140 | **0.3305** | ~36.9M | 298.8 MB |
| **YOLOv8** | **YOLOv8m** | 640x640 | 8 | 0.6452 | 0.4504 | 0.4895 | 0.3277 | 25.86M | 49.6 MB |
| **YOLOv5** | **YOLOv5m** | 640x640 | 16 | 0.6331 | 0.4424 | 0.4705 | 0.3044 | 20.91M | 40.3 MB |
| **YOLOv8** | **YOLOv8n** | 640x640 | 16 | 0.5811 | 0.3605 | 0.3927 | 0.2555 | 3.01M | 6.0 MB |

---

## 6. Key Takeaways & Recommendations

1. **Batch Size Overshadows Resolution:**
   Training YOLOv8l with a larger batch size (`bs=32`) at `512px` yielded a significantly higher mAP (+9.1%) than high-resolution `960px` training forced into a low batch size (`bs=6`) due to VRAM limits.
2. **Localization Accuracy:**
   YOLOv8l achieves superior bounding box regression capability (**`0.3480 mAP@0.50:0.95`**), outperforming all other model families.
3. **Deployment Recommendation:**
   * **For Maximum Detection Precision:** Deploy **YOLOv8l (`train_fast2`)**.
   * **For Edge / Constrained Devices:** Deploy **YOLOv5s (v1)**, which achieves high accuracy (`0.5244 mAP@0.50`) with only `7.05M` parameters and `13.8 MB` model size.
