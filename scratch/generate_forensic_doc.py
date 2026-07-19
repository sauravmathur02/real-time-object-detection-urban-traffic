import json
import os

with open("scratch/forensic_full.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

# Sort runs chronologically by last_mod
runs_sorted = sorted(runs, key=lambda x: x['last_mod'])

# Group runs by category
yolov8l_runs = []
yolov8_other = []
yolov5_runs = []
yolov7_runs = []
other_runs = []

for r in runs:
    p = r['rel_path'].lower()
    m = str(r['model']).lower()
    if 'yolov8l' in p or 'yolov8l' in m or 'stage' in p or 'train_fast' in p or 'train_auto' in p or 'high' in p or 'train6' in p or 'train7' in p:
        yolov8l_runs.append(r)
    elif 'yolov8' in p or 'yolov8' in m or 'yolo26' in p:
        yolov8_other.append(r)
    elif 'yolov5' in p or 'yolov5' in m:
        yolov5_runs.append(r)
    elif 'yolov7' in p or 'yolov7' in m or 'exp' in p:
        yolov7_runs.append(r)
    else:
        other_runs.append(r)

print(f"YOLOv8l runs count: {len(yolov8l_runs)}")
print(f"YOLOv8 other count: {len(yolov8_other)}")
print(f"YOLOv5 count: {len(yolov5_runs)}")
print(f"YOLOv7 count: {len(yolov7_runs)}")
print(f"Other count: {len(other_runs)}")
