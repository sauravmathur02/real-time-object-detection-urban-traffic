import os
import json
import yaml
import glob
from datetime import datetime

workspace = r"c:\Repo\object-Detection"

# 1. Dataset Audit
dataset_info = {}
dataset_report_path = os.path.join(workspace, "results", "dataset_report.json")
if os.path.exists(dataset_report_path):
    with open(dataset_report_path, "r", encoding="utf-8") as f:
        dataset_info["dataset_report.json"] = json.load(f)

dataset_stats_dir = os.path.join(workspace, "dataset_statistics")
if os.path.exists(dataset_stats_dir):
    dataset_info["stats_files"] = os.listdir(dataset_stats_dir)

yaml_files = glob.glob(os.path.join(workspace, "**", "*.yaml"), recursive=True)
dataset_yamls = {}
for yf in yaml_files:
    if "data" in os.path.basename(yf) or "bdd" in os.path.basename(yf) or "idd" in os.path.basename(yf) or "coco" in os.path.basename(yf):
        try:
            with open(yf, "r", encoding="utf-8", errors="ignore") as f:
                dataset_yamls[os.path.relpath(yf, workspace)] = yaml.safe_load(f)
        except Exception as e:
            dataset_yamls[os.path.relpath(yf, workspace)] = f"Error: {e}"

# 2. System / Hardware / Environment Audit
env_info = {}
requirements_path = os.path.join(workspace, "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        env_info["requirements.txt"] = f.read()

# 3. Read evaluation/master_results.csv
master_results_path = os.path.join(workspace, "evaluation", "master_results.csv")
master_results = []
if os.path.exists(master_results_path):
    with open(master_results_path, "r", encoding="utf-8") as f:
        master_results = f.read()

# 4. Read evaluation/master_metrics.csv
master_metrics_path = os.path.join(workspace, "evaluation", "master_metrics.csv")
master_metrics = ""
if os.path.exists(master_metrics_path):
    with open(master_metrics_path, "r", encoding="utf-8") as f:
        master_metrics = f.read()

audit_data = {
    "dataset_report": dataset_info,
    "dataset_yamls": dataset_yamls,
    "env_info": env_info,
    "master_results_csv": master_results,
    "master_metrics_csv": master_metrics
}

with open(os.path.join(workspace, "scratch", "paper_audit_raw.json"), "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)

print("Gathered raw audit data. Output written to scratch/paper_audit_raw.json")
