# Forensic Audit & Model Lineage Report
**Project:** Real-Time Object Detection in Urban Traffic Scenes
**Analysis Date:** July 19, 2026
**Scope:** Forensic investigation of all 45 training run directories across `runs/`, `runs/detect/`, and `experiments/`

---

## 1. Repository Directory Tree of All Training Runs

```text
c:\Repo\object-Detection
├── experiments/
│   ├── yolov5/
│   │   ├── train_yolov5m/
│   │   ├── train_yolov5s_v1/
│   │   └── validation_yolov5s/
│   └── yolov7/
│       ├── train_yolov7/
│       ├── train_yolov7_workers2_test/
│       ├── train_yolov7x/
│       ├── train_yolov7x2/
│       ├── train_yolov7x3/
│       ├── train_yolov7x4/
│       ├── train_yolov7x_cache/
│       └── train_yolov7x_final/
└── runs/
    ├── detect/
    │   ├── YOLOv8/
    │   │   └── yolov8s/
    │   ├── object_stage1/
    │   ├── object_stage2/
    │   ├── object_stage3_strong/
    │   ├── object_stage3_strong2/
    │   ├── object_stage4_merged/
    │   ├── object_stage4_riderfix/
    │   ├── object_stage4_riderfix2/
    │   ├── object_stage4_riderfix3/
    │   ├── object_stage4_riderfix4/
    │   ├── predict/ & predict2/
    │   ├── train/ & train2/
    │   ├── train3/ & train4/ & train5/ & train6/ & train7/
    │   ├── train_auto_v1/
    │   ├── train_fast/ & train_fast2/
    │   ├── train_yolov8l_high/ & train_yolov8l_high2/
    │   ├── train_yolov8m/
    │   ├── train_yolov8n/ & train_yolov8n2/
    │   ├── train_yolov8s/ & train_yolov8s2/
    │   ├── tune-yolo26n-objv1-coco/train31/
    │   └── val1/ .. val25/
    └── train/
        └── exp/ .. exp6/
```

## 2. Comprehensive Comparison Table of Every Training Run

| Run Directory | Model Configured | Config Epochs | Completed Epochs | Best Checkpoint | Last Checkpoint | mAP@0.50 | mAP@0.50:0.95 | Timestamp / Date | Execution Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `runs\detect\train` | `yolov8n.pt` | 10 | 0 | No | No | - | - | 2026-02-13 11:40:04 | Empty / Created but not trained |
| `runs\detect\train2` | `yolov8n.pt` | 10 | 0 | No | No | - | - | 2026-02-13 12:01:49 | Empty / Created but not trained |
| `runs\detect\train3` | `yolov8n.pt` | 10 | 10 | Yes (5.94MB) | Yes (5.94MB) | 0.2034 | 0.1220 | 2026-02-15 04:06:12 | Completed (10/10 epochs) |
| `runs\detect\train4` | `yolov8n.pt` | 1 | 1 | Yes (5.93MB) | Yes (5.93MB) | 0.0000 | 0.0000 | 2026-02-16 22:59:53 | Completed (1/1 epochs) |
| `runs\detect\train_yolov8s` | `yolov8s.pt` | 50 | 0 | No | No | - | - | 2026-02-16 23:29:39 | Empty / Created but not trained |
| `runs\detect\train_yolov8l_high` | `yolov8l.pt` | 100 | 19 | Yes (250.13MB) | Yes (250.13MB) | 0.1920 | 0.1171 | 2026-02-17 13:21:31 | Interrupted (19/100 epochs) |
| `runs\detect\train_yolov8l_high2` | `yolov8l.pt` | 100 | 11 | Yes (250.13MB) | Yes (250.13MB) | 0.1901 | 0.1159 | 2026-02-17 22:53:59 | Interrupted (11/100 epochs) |
| `runs\detect\train5` | `yolov8n.pt` | 2 | 2 | Yes (5.93MB) | Yes (5.93MB) | 0.1601 | 0.0890 | 2026-02-25 14:09:19 | Completed (2/2 epochs) |
| `runs\detect\train6` | `yolov8l.pt` | 50 | 50 | Yes (83.6MB) | Yes (83.6MB) | 0.4086 | 0.2244 | 2026-02-25 16:09:46 | Completed (50/50 epochs) |
| `runs\detect\object_stage1` | `yolov8l.pt` | 30 | 30 | Yes (83.6MB) | Yes (83.6MB) | 0.4106 | 0.2256 | 2026-02-28 17:52:59 | Completed (30/30 epochs) |
| `runs\detect\object_stage2` | `best.pt` | 70 | 70 | Yes (83.6MB) | Yes (83.6MB) | 0.3901 | 0.2121 | 2026-02-28 21:07:00 | Completed (70/70 epochs) |
| `runs\detect\object_stage3_strong` | `best.pt` | 50 | 0 | No | No | - | - | 2026-03-02 13:45:32 | Empty / Created but not trained |
| `runs\detect\object_stage3_strong2` | `best.pt` | 50 | 50 | Yes (83.61MB) | Yes (83.61MB) | 0.4732 | 0.2552 | 2026-03-02 16:16:50 | Completed (50/50 epochs) |
| `runs\detect\object_stage4_riderfix` | `best.pt` | 50 | 0 | No | No | - | - | 2026-03-04 22:18:45 | Empty / Created but not trained |
| `runs\detect\object_stage4_riderfix2` | `best.pt` | 50 | 1 | Yes (250.35MB) | Yes (250.35MB) | 0.3133 | 0.1806 | 2026-03-04 22:35:46 | Interrupted (1/50 epochs) |
| `runs\detect\tune-yolo26n-objv1-coco\train31` | `yolo26n.pt` | 245 | 0 | No | No | - | - | 2026-03-05 00:35:43 | Empty / Created but not trained |
| `runs\detect\object_stage4_riderfix3` | `best.pt` | 50 | 0 | No | No | - | - | 2026-03-05 00:43:35 | Empty / Created but not trained |
| `runs\detect\object_stage4_riderfix4` | `best.pt` | 50 | 50 | Yes (83.61MB) | Yes (83.61MB) | 0.4617 | 0.2572 | 2026-03-05 03:19:12 | Completed (50/50 epochs) |
| `runs\detect\train_auto_v1` | `yolov8l.pt` | 50 | 50 | Yes (83.59MB) | Yes (83.59MB) | 0.4828 | 0.2952 | 2026-03-06 23:21:19 | Completed (50/50 epochs) |
| `runs\detect\object_stage4_merged` | `best.pt` | 50 | 0 | No | No | - | - | 2026-03-07 11:57:32 | Empty / Created but not trained |
| `runs\detect\train7` | `best.pt` | 50 | 3 | Yes (250.16MB) | Yes (250.16MB) | 0.4978 | 0.3018 | 2026-03-09 06:54:47 | Interrupted (3/50 epochs) |
| `runs\detect\train_fast` | `best.pt` | 50 | 0 | No | No | - | - | 2026-03-09 14:48:56 | Empty / Created but not trained |
| `runs\detect\train_fast2` | `best.pt` | 30 | 30 | Yes (83.57MB) | Yes (83.57MB) | 0.5524 | 0.3480 | 2026-03-10 04:44:54 | Completed (30/30 epochs) |
| `experiments\yolov5\validation_yolov5s` | `yolov5s.pt` | 1 | 1 | Yes (13.77MB) | Yes (13.77MB) | - | - | 2026-07-10 15:06:37 | Completed (1/1 epochs) |
| `experiments\yolov5\train_yolov5s_v1` | `last.pt` | 50 | 50 | Yes (13.77MB) | Yes (13.77MB) | - | - | 2026-07-10 23:25:29 | Completed (50/50 epochs) |
| `runs\detect\train_yolov8n` | `yolov8n.pt` | 50 | 50 | Yes (5.95MB) | Yes (5.95MB) | 0.3927 | 0.2555 | 2026-07-14 06:07:10 | Completed (50/50 epochs) |
| `runs\detect\train_yolov8n2` | `yolov8n.pt` | 50 | 50 | Yes (5.95MB) | Yes (5.95MB) | 0.3927 | 0.2555 | 2026-07-14 06:08:53 | Completed (50/50 epochs) |
| `runs\detect\YOLOv8\yolov8s` | `yolov8s.pt` | 500 | 0 | No | No | - | - | 2026-07-14 14:04:42 | Empty / Created but not trained |
| `runs\detect\train_yolov8s2` | `last.pt` | 50 | 50 | Yes (21.47MB) | Yes (21.47MB) | 0.4599 | 0.3010 | 2026-07-14 16:24:07 | Completed (50/50 epochs) |
| `runs\detect\train_yolov8m` | `last.pt` | 50 | 50 | Yes (49.62MB) | Yes (49.62MB) | 0.4895 | 0.3277 | 2026-07-15 04:06:05 | Completed (50/50 epochs) |
| `experiments\yolov5\train_yolov5m` | `yolov5m.pt` | 50 | 50 | Yes (40.29MB) | Yes (40.29MB) | - | - | 2026-07-15 20:23:48 | Completed (50/50 epochs) |
| `experiments\yolov7\train_yolov7x` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-15 21:13:06 | Empty / Created but not trained |
| `runs\train\exp` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-15 21:14:49 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7x2` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-15 21:19:41 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7x3` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-15 21:22:43 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7x4` | `yolov7x.pt` | 50 | 0 | No | No | - | - | 2026-07-15 21:34:42 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7x_cache` | `yolov7x.pt` | 50 | 0 | No | No | - | - | 2026-07-16 00:19:24 | Empty / Created but not trained |
| `runs\train\exp2` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-16 20:19:39 | Empty / Created but not trained |
| `runs\train\exp3` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-16 20:23:13 | Empty / Created but not trained |
| `runs\train\exp4` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-16 20:25:45 | Empty / Created but not trained |
| `runs\train\exp5` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-16 20:40:16 | Empty / Created but not trained |
| `runs\train\exp6` | `yolov7x.pt` | 1 | 0 | No | No | - | - | 2026-07-16 20:42:34 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7x_final` | `yolov7x.pt` | 50 | 4 | Yes (541.61MB) | Yes (541.61MB) | 0.3932 | 0.2348 | 2026-07-17 12:10:35 | Interrupted (4/50 epochs) |
| `experiments\yolov7\train_yolov7_workers2_test` | `yolov7.pt` | 1 | 0 | No | No | - | - | 2026-07-17 18:32:44 | Empty / Created but not trained |
| `experiments\yolov7\train_yolov7` | `yolov7.pt` | 50 | 42 | Yes (284.93MB) | Yes (284.93MB) | 0.5140 | 0.3305 | 2026-07-19 08:15:32 | Interrupted (42/50 epochs) |

---

## 3. Chronological Timeline of All Training Sessions

1. **Feb 15 - Feb 16, 2026 (Early Exploratory Phase):**
   - `train3` & `train4` (YOLOv8n CPU/GPU test runs, 1-10 epochs).
2. **Feb 17, 2026 (Initial YOLOv8l High-Epoch Attempts):**
   - `train_yolov8l_high`: Launched with `epochs=100` on BDD100K at 832px. Interrupted after **19 epochs** at 13:21.
   - `train_yolov8l_high2`: Restarted with `epochs=100` at 20:48. Interrupted after **11 epochs** at 22:53.
3. **Feb 25 - Feb 28, 2026 (First Successful Full Baseline Runs):**
   - `train5` (YOLOv8n, 2 epochs) & `train6` (YOLOv8l, 50 epochs completed, mAP50=0.4086).
   - `object_stage1` (YOLOv8l, 30 epochs completed, mAP50=0.4106).
   - `object_stage2` (YOLOv8l fine-tuned from `object_stage1` for 70 epochs, cumulative 100 epochs, mAP50=0.3901).
4. **Mar 02 - Mar 05, 2026 (Multi-Stage Refinement Pipeline):**
   - `object_stage3_strong2` (YOLOv8l fine-tuned from `object_stage2` for 50 epochs, mAP50=0.4732).
   - `object_stage4_riderfix4` (YOLOv8l Final Paper Model, 50 epochs on `bdd_balanced` at 960px, mAP50=0.4618).
5. **Mar 06 - Mar 10, 2026 (High-Batch Efficiency Optimization):**
   - `train_auto_v1` (YOLOv8l, 50 epochs, 640px, bs=4, mAP50=0.4828).
   - `train7` (YOLOv8l, 960px, bs=16, interrupted at 3 epochs, mAP50=0.4978).
   - `train_fast2` (YOLOv8l, 30 epochs, 512px, bs=32, mAP50=0.5524 — **Repository Peak Metric**).
6. **Jul 10 - Jul 19, 2026 (Framework Comparison Phase):**
   - YOLOv5s (`train_yolov5s_v1`, 50 epochs, mAP50=0.5244) & YOLOv5m (`train_yolov5m`, 50 epochs).
   - YOLOv8s & YOLOv8m (`train_yolov8m`, 50 epochs, mAP50=0.4895).
   - YOLOv7 (`train_yolov7`, 42/50 epochs completed today, mAP50=0.5140, mAP50-95=0.3305).