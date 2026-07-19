import json
import yaml
import os

workspace = r"c:\Repo\object-Detection"

# dataset_report.json
report_file = os.path.join(workspace, "results", "dataset_report.json")
if os.path.exists(report_file):
    with open(report_file, "r") as f:
        print("=== dataset_report.json ===")
        print(json.dumps(json.load(f), indent=2))

# bdd_balanced.yaml
yaml_file = os.path.join(workspace, "data1", "bdd_balanced.yaml")
if os.path.exists(yaml_file):
    with open(yaml_file, "r") as f:
        print("\n=== data1/bdd_balanced.yaml ===")
        print(yaml.safe_load(f))

# dataset_statistics directory
stats_dir = os.path.join(workspace, "dataset_statistics")
if os.path.exists(stats_dir):
    print("\n=== dataset_statistics files ===")
    for root, dirs, files in os.walk(stats_dir):
        for file in files:
            p = os.path.join(root, file)
            print(os.path.relpath(p, workspace))
            if file.endswith((".json", ".yaml", ".txt")):
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if len(content) < 2000:
                            print(content)
                except Exception as e:
                    pass
