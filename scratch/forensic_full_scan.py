import os
import glob
import csv
import json
import yaml
from datetime import datetime

workspace = r"c:\Repo\object-Detection"

# We want to scan all runs under runs/ and experiments/ and any other folder
search_dirs = [os.path.join(workspace, "runs"), os.path.join(workspace, "experiments")]

run_dirs = []

for base in search_dirs:
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        # A directory is a run directory if it contains args.yaml, opt.yaml, results.csv, results.txt, or a weights directory
        if "args.yaml" in files or "opt.yaml" in files or "results.csv" in files or "results.txt" in files or "weights" in dirs:
            if os.path.basename(root) != "weights":
                run_dirs.append(root)

run_dirs = sorted(list(set(run_dirs)))

print(f"Total training run folders identified: {len(run_dirs)}")

runs_data = []

for r in run_dirs:
    rel_path = os.path.relpath(r, workspace)
    folder_name = os.path.basename(r)
    
    # Check weights
    weights_dir = os.path.join(r, "weights")
    best_pt = os.path.exists(os.path.join(weights_dir, "best.pt"))
    last_pt = os.path.exists(os.path.join(weights_dir, "last.pt"))
    
    best_size = os.path.getsize(os.path.join(weights_dir, "best.pt")) if best_pt else 0
    last_size = os.path.getsize(os.path.join(weights_dir, "last.pt")) if last_pt else 0
    
    # Check config
    args_path = os.path.join(r, "args.yaml")
    opt_path = os.path.join(r, "opt.yaml")
    
    model = "Unknown"
    epochs_cfg = "Unknown"
    imgsz = "Unknown"
    batch = "Unknown"
    dataset = "Unknown"
    optimizer = "Unknown"
    
    if os.path.exists(args_path):
        try:
            with open(args_path, 'r', encoding='utf-8', errors='ignore') as f:
                d = yaml.safe_load(f)
                if isinstance(d, dict):
                    model = d.get('model', 'Unknown')
                    epochs_cfg = d.get('epochs', 'Unknown')
                    imgsz = d.get('imgsz', 'Unknown')
                    batch = d.get('batch', 'Unknown')
                    dataset = d.get('data', 'Unknown')
                    optimizer = d.get('optimizer', 'Unknown')
        except Exception as e:
            pass
            
    if os.path.exists(opt_path) and model == "Unknown":
        try:
            with open(opt_path, 'r', encoding='utf-8', errors='ignore') as f:
                d = yaml.safe_load(f)
                if isinstance(d, dict):
                    model = d.get('weights', d.get('cfg', 'Unknown'))
                    epochs_cfg = d.get('epochs', 'Unknown')
                    imgsz = d.get('img_size', 'Unknown')
                    batch = d.get('batch_size', 'Unknown')
                    dataset = d.get('data', 'Unknown')
                    optimizer = d.get('adam', 'Unknown')
        except Exception as e:
            pass

    # Results
    results_csv = os.path.join(r, "results.csv")
    results_txt = os.path.join(r, "results.txt")
    epochs_done = 0
    last_mod = "Unknown"
    metrics = {}
    
    if os.path.exists(results_csv):
        mtime = os.path.getmtime(results_csv)
        last_mod = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(results_csv, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) > 1:
                    epochs_done = len(lines) - 1
                    header = [h.strip() for h in lines[0].split(',')]
                    last_line = [v.strip() for v in lines[-1].split(',')]
                    for h, v in zip(header, last_line):
                        try:
                            metrics[h] = float(v)
                        except:
                            metrics[h] = v
        except Exception as e:
            pass
    elif os.path.exists(results_txt):
        mtime = os.path.getmtime(results_txt)
        last_mod = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(results_txt, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
                epochs_done = len(lines)
                if lines:
                    last_line = lines[-1].split()
                    if len(last_line) >= 11:
                        metrics["metrics/precision(B)"] = float(last_line[8])
                        metrics["metrics/recall(B)"] = float(last_line[9])
                        metrics["metrics/mAP50(B)"] = float(last_line[10])
                        if len(last_line) >= 12:
                            metrics["metrics/mAP50-95(B)"] = float(last_line[11])
        except Exception as e:
            pass

    # Directory timestamp if results file missing
    if last_mod == "Unknown":
        mtime = os.path.getmtime(r)
        last_mod = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

    # Status classification
    if epochs_done == 0:
        if best_pt or last_pt:
            status = "Interrupted (No log lines)"
        else:
            status = "Empty / Created but not trained"
    elif isinstance(epochs_cfg, int) and epochs_done < epochs_cfg:
        status = f"Interrupted ({epochs_done}/{epochs_cfg} epochs)"
    elif isinstance(epochs_cfg, int) and epochs_done >= epochs_cfg:
        status = f"Completed ({epochs_done}/{epochs_cfg} epochs)"
    else:
        status = f"Finished ({epochs_done} epochs)"

    runs_data.append({
        "rel_path": rel_path,
        "folder_name": folder_name,
        "model": str(model),
        "epochs_cfg": epochs_cfg,
        "epochs_done": epochs_done,
        "imgsz": imgsz,
        "batch": batch,
        "dataset": dataset,
        "best_pt": best_pt,
        "best_mb": round(best_size / (1024*1024), 2),
        "last_pt": last_pt,
        "last_mb": round(last_size / (1024*1024), 2),
        "last_mod": last_mod,
        "status": status,
        "metrics": metrics
    })

with open(os.path.join(workspace, "scratch", "forensic_full.json"), "w", encoding="utf-8") as f:
    json.dump(runs_data, f, indent=2)

print("Forensic scan complete. Output saved to scratch/forensic_full.json")
