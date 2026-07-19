import json
import os

with open("scratch/forensic_full.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

# Sort runs chronologically by last_mod
runs_sorted = sorted(runs, key=lambda x: x['last_mod'])

out_path = os.path.join("scratch", "forensic_final_report.md")

lines = []

lines.append("# Forensic Audit & Model Lineage Report")
lines.append("**Project:** Real-Time Object Detection in Urban Traffic Scenes")
lines.append("**Analysis Date:** July 19, 2026")
lines.append("**Scope:** Forensic investigation of all 45 training run directories across `runs/`, `runs/detect/`, and `experiments/`")
lines.append("\n---\n")

# Section 1: Repository Tree
lines.append("## 1. Repository Directory Tree of All Training Runs\n")
lines.append("```text")
lines.append("c:\\Repo\\object-Detection")
lines.append("├── experiments/")
lines.append("│   ├── yolov5/")
lines.append("│   │   ├── train_yolov5m/")
lines.append("│   │   ├── train_yolov5s_v1/")
lines.append("│   │   └── validation_yolov5s/")
lines.append("│   └── yolov7/")
lines.append("│       ├── train_yolov7/")
lines.append("│       ├── train_yolov7_workers2_test/")
lines.append("│       ├── train_yolov7x/")
lines.append("│       ├── train_yolov7x2/")
lines.append("│       ├── train_yolov7x3/")
lines.append("│       ├── train_yolov7x4/")
lines.append("│       ├── train_yolov7x_cache/")
lines.append("│       └── train_yolov7x_final/")
lines.append("└── runs/")
lines.append("    ├── detect/")
lines.append("    │   ├── YOLOv8/")
lines.append("    │   │   └── yolov8s/")
lines.append("    │   ├── object_stage1/")
lines.append("    │   ├── object_stage2/")
lines.append("    │   ├── object_stage3_strong/")
lines.append("    │   ├── object_stage3_strong2/")
lines.append("    │   ├── object_stage4_merged/")
lines.append("    │   ├── object_stage4_riderfix/")
lines.append("    │   ├── object_stage4_riderfix2/")
lines.append("    │   ├── object_stage4_riderfix3/")
lines.append("    │   ├── object_stage4_riderfix4/")
lines.append("    │   ├── predict/ & predict2/")
lines.append("    │   ├── train/ & train2/")
lines.append("    │   ├── train3/ & train4/ & train5/ & train6/ & train7/")
lines.append("    │   ├── train_auto_v1/")
lines.append("    │   ├── train_fast/ & train_fast2/")
lines.append("    │   ├── train_yolov8l_high/ & train_yolov8l_high2/")
lines.append("    │   ├── train_yolov8m/")
lines.append("    │   ├── train_yolov8n/ & train_yolov8n2/")
lines.append("    │   ├── train_yolov8s/ & train_yolov8s2/")
lines.append("    │   ├── tune-yolo26n-objv1-coco/train31/")
lines.append("    │   └── val1/ .. val25/")
lines.append("    └── train/")
lines.append("        └── exp/ .. exp6/")
lines.append("```\n")

# Section 2: Comprehensive Table
lines.append("## 2. Comprehensive Comparison Table of Every Training Run\n")
lines.append("| Run Directory | Model Configured | Config Epochs | Completed Epochs | Best Checkpoint | Last Checkpoint | mAP@0.50 | mAP@0.50:0.95 | Timestamp / Date | Execution Status |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

for r in runs_sorted:
    m = r['metrics']
    p = m.get('metrics/precision(B)', m.get('Precision', '-'))
    rec = m.get('metrics/recall(B)', m.get('Recall', '-'))
    m50 = m.get('metrics/mAP50(B)', m.get('mAP50', '-'))
    m5095 = m.get('metrics/mAP50-95(B)', m.get('mAP50-95', '-'))

    m50_str = f"{m50:.4f}" if isinstance(m50, float) else "-"
    m5095_str = f"{m5095:.4f}" if isinstance(m5095, float) else "-"
    
    best_str = f"Yes ({r['best_mb']}MB)" if r['best_pt'] else "No"
    last_str = f"Yes ({r['last_mb']}MB)" if r['last_pt'] else "No"
    
    model_name = os.path.basename(r['model']) if r['model'] != "Unknown" else "Unknown"

    lines.append(f"| `{r['rel_path']}` | `{model_name}` | {r['epochs_cfg']} | {r['epochs_done']} | {best_str} | {last_str} | {m50_str} | {m5095_str} | {r['last_mod']} | {r['status']} |")

# Section 3: Timeline
lines.append("\n---\n")
lines.append("## 3. Chronological Timeline of All Training Sessions\n")
lines.append("1. **Feb 15 - Feb 16, 2026 (Early Exploratory Phase):**")
lines.append("   - `train3` & `train4` (YOLOv8n CPU/GPU test runs, 1-10 epochs).")
lines.append("2. **Feb 17, 2026 (Initial YOLOv8l High-Epoch Attempts):**")
lines.append("   - `train_yolov8l_high`: Launched with `epochs=100` on BDD100K at 832px. Interrupted after **19 epochs** at 13:21.")
lines.append("   - `train_yolov8l_high2`: Restarted with `epochs=100` at 20:48. Interrupted after **11 epochs** at 22:53.")
lines.append("3. **Feb 25 - Feb 28, 2026 (First Successful Full Baseline Runs):**")
lines.append("   - `train5` (YOLOv8n, 2 epochs) & `train6` (YOLOv8l, 50 epochs completed, mAP50=0.4086).")
lines.append("   - `object_stage1` (YOLOv8l, 30 epochs completed, mAP50=0.4106).")
lines.append("   - `object_stage2` (YOLOv8l fine-tuned from `object_stage1` for 70 epochs, cumulative 100 epochs, mAP50=0.3901).")
lines.append("4. **Mar 02 - Mar 05, 2026 (Multi-Stage Refinement Pipeline):**")
lines.append("   - `object_stage3_strong2` (YOLOv8l fine-tuned from `object_stage2` for 50 epochs, mAP50=0.4732).")
lines.append("   - `object_stage4_riderfix4` (YOLOv8l Final Paper Model, 50 epochs on `bdd_balanced` at 960px, mAP50=0.4618).")
lines.append("5. **Mar 06 - Mar 10, 2026 (High-Batch Efficiency Optimization):**")
lines.append("   - `train_auto_v1` (YOLOv8l, 50 epochs, 640px, bs=4, mAP50=0.4828).")
lines.append("   - `train7` (YOLOv8l, 960px, bs=16, interrupted at 3 epochs, mAP50=0.4978).")
lines.append("   - `train_fast2` (YOLOv8l, 30 epochs, 512px, bs=32, mAP50=0.5524 — **Repository Peak Metric**).")
lines.append("6. **Jul 10 - Jul 19, 2026 (Framework Comparison Phase):**")
lines.append("   - YOLOv5s (`train_yolov5s_v1`, 50 epochs, mAP50=0.5244) & YOLOv5m (`train_yolov5m`, 50 epochs).")
lines.append("   - YOLOv8s & YOLOv8m (`train_yolov8m`, 50 epochs, mAP50=0.4895).")
lines.append("   - YOLOv7 (`train_yolov7`, 42/50 epochs completed today, mAP50=0.5140, mAP50-95=0.3305).")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Generated report at {out_path}")
