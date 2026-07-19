import os
import glob
import csv
import json
import yaml
from datetime import datetime

workspace = r"c:\Repo\object-Detection"

artifacts_to_find = ["results.csv", "results.txt", "args.yaml", "opt.yaml", "weights"]

runs = set()
for root, dirs, files in os.walk(workspace):
    if ".git" in root or "venv" in root or ".gemini" in root:
        continue
    for f in files + dirs:
        if f in artifacts_to_find or f.startswith("events.out.tfevents"):
            runs.add(root)

clean_runs = sorted(list(runs))
final_runs = []
for r in clean_runs:
    if os.path.basename(r) == "weights":
        parent = os.path.dirname(r)
        final_runs.append(parent)
    else:
        final_runs.append(r)

final_runs = sorted(list(set(final_runs)))

report_data = []

for run_dir in final_runs:
    rel_path = os.path.relpath(run_dir, workspace)
    folder_name = os.path.basename(run_dir)
    
    best_pt = os.path.exists(os.path.join(run_dir, "weights", "best.pt"))
    last_pt = os.path.exists(os.path.join(run_dir, "weights", "last.pt"))
    
    best_size = os.path.getsize(os.path.join(run_dir, "weights", "best.pt")) if best_pt else 0
    last_size = os.path.getsize(os.path.join(run_dir, "weights", "last.pt")) if last_pt else 0

    model = "Unknown"
    epochs_configured = "Unknown"
    dataset = "Unknown"

    args_yaml = os.path.join(run_dir, "args.yaml")
    opt_yaml = os.path.join(run_dir, "opt.yaml")

    if os.path.exists(args_yaml):
        try:
            with open(args_yaml, 'r', encoding='utf-8', errors='ignore') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    model = data.get("model", "Unknown")
                    epochs_configured = data.get("epochs", "Unknown")
                    dataset = data.get("data", "Unknown")
        except Exception as e:
            model = f"Error reading args: {e}"

    if os.path.exists(opt_yaml) and model == "Unknown":
        try:
            with open(opt_yaml, 'r', encoding='utf-8', errors='ignore') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    model = data.get("weights", data.get("cfg", "Unknown"))
                    epochs_configured = data.get("epochs", "Unknown")
                    dataset = data.get("data", "Unknown")
        except Exception as e:
            pass

    epochs_completed = 0
    metrics = {}
    last_modified_str = "Unknown"

    results_csv = os.path.join(run_dir, "results.csv")
    results_txt = os.path.join(run_dir, "results.txt")

    if os.path.exists(results_csv):
        mtime = os.path.getmtime(results_csv)
        last_modified_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(results_csv, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) > 1:
                    header = [h.strip() for h in lines[0].split(',')]
                    epochs_completed = len(lines) - 1
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
        last_modified_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(results_txt, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
                epochs_completed = len(lines)
                if lines:
                    last_line = lines[-1].split()
                    if len(last_line) >= 11:
                        metrics["Precision"] = float(last_line[8])
                        metrics["Recall"] = float(last_line[9])
                        metrics["mAP50"] = float(last_line[10])
                        if len(last_line) >= 12:
                            metrics["mAP50-95"] = float(last_line[11])
        except Exception as e:
            pass

    tf_events = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    tf_events_count = len(tf_events)
    latest_event_mtime = max([os.path.getmtime(e) for e in tf_events]) if tf_events else None
    if latest_event_mtime and last_modified_str == "Unknown":
        last_modified_str = datetime.fromtimestamp(latest_event_mtime).strftime('%Y-%m-%d %H:%M:%S')

    report_data.append({
        "rel_path": rel_path,
        "folder_name": folder_name,
        "model": model,
        "epochs_configured": epochs_configured,
        "epochs_completed": epochs_completed,
        "best_pt": best_pt,
        "best_size_mb": round(best_size / (1024*1024), 2),
        "last_pt": last_pt,
        "last_size_mb": round(last_size / (1024*1024), 2),
        "dataset": dataset,
        "last_modified": last_modified_str,
        "tf_events_count": tf_events_count,
        "metrics": metrics
    })

out_file = os.path.join(workspace, "scratch", "forensic_report.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

print(f"Analysis written to {out_file}. Total runs found: {len(report_data)}")
