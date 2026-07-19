import os
import glob
import json
import yaml

workspace = r"c:\Repo\object-Detection"

# Find all PNG/JPG figures and evaluation artifacts
eval_artifacts = []
for root, dirs, files in os.walk(workspace):
    if ".git" in root or "venv" in root or ".gemini" in root:
        continue
    for f in files:
        if f.endswith((".png", ".jpg", ".jpeg", ".svg")):
            if any(k in f.lower() for k in ["confusion", "pr_curve", "f1_curve", "p_curve", "r_curve", "val_batch", "train_batch", "map_vs", "precision_recall"]):
                rel = os.path.relpath(os.path.join(root, f), workspace)
                eval_artifacts.append(rel)

print(f"Total evaluation artifacts found: {len(eval_artifacts)}")

# Find dataset configs & stats
dataset_reports = {}
for root, dirs, files in os.walk(os.path.join(workspace, "dataset_statistics")):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), workspace)
        dataset_reports[f] = rel

# Inspect data1/ yaml files
yaml_info = {}
data1_dir = os.path.join(workspace, "data1")
if os.path.exists(data1_dir):
    for f in os.listdir(data1_dir):
        if f.endswith(".yaml"):
            p = os.path.join(data1_dir, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as yf:
                try:
                    yaml_info[f] = yaml.safe_load(yf)
                except Exception as e:
                    yaml_info[f] = str(e)

with open(os.path.join(workspace, "scratch", "eval_artifacts.json"), "w", encoding="utf-8") as f:
    json.dump({
        "eval_artifacts": sorted(eval_artifacts),
        "dataset_statistics_files": dataset_reports,
        "data1_yamls": yaml_info
    }, f, indent=2)

print("Saved evaluation artifacts and dataset info to scratch/eval_artifacts.json")
