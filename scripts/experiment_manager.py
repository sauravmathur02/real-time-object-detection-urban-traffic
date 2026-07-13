import os
import sys
import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_yolov8_args(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            args = yaml.safe_load(f)
        if not args: return {}
        return {
            'model': args.get('model', 'Unknown'),
            'dataset': args.get('data', 'Unknown'),
            'epochs': args.get('epochs', 'Unknown'),
            'imgsz': args.get('imgsz', 'Unknown'),
            'batch': args.get('batch', 'Unknown'),
            'optimizer': args.get('optimizer', 'Unknown'),
            'lr': args.get('lr0', 'Unknown'),
            'device': args.get('device', 'Unknown')
        }
    except Exception as e:
        logging.error(f"Error parsing {yaml_path}: {e}")
        return {}


def parse_yolov5_opt(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            opt = yaml.safe_load(f)
        if not opt: return {}
        return {
            'model': opt.get('weights', 'Unknown'),
            'dataset': opt.get('data', 'Unknown'),
            'epochs': opt.get('epochs', 'Unknown'),
            'imgsz': opt.get('imgsz', 'Unknown'),
            'batch': opt.get('batch_size', 'Unknown'),
            'optimizer': opt.get('optimizer', 'Unknown'),
            'lr': opt.get('lr0', 'Unknown'),
            'device': opt.get('device', 'Unknown')
        }
    except Exception as e:
        logging.error(f"Error parsing {yaml_path}: {e}")
        return {}


def parse_results_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        
        # YOLOv8 format mapping
        if 'metrics/precision(B)' in df.columns:
            return {
                'precision': last_row.get('metrics/precision(B)', 0),
                'recall': last_row.get('metrics/recall(B)', 0),
                'map50': last_row.get('metrics/mAP50(B)', 0),
                'map50_95': last_row.get('metrics/mAP50-95(B)', 0),
                'fitness': last_row.get('fitness', 0) # sometimes available
            }
        # YOLOv5 format mapping
        elif 'metrics/precision' in df.columns:
            return {
                'precision': last_row.get('metrics/precision', 0),
                'recall': last_row.get('metrics/recall', 0),
                'map50': last_row.get('metrics/mAP_0.5', 0),
                'map50_95': last_row.get('metrics/mAP_0.5:0.95', 0),
                'fitness': 0 # compute manually if needed
            }
        else:
            return {}
    except Exception as e:
        logging.error(f"Error parsing {csv_path}: {e}")
        return {}

def categorize_experiment(name):
    name_lower = name.lower()
    if 'stage4' in name_lower or 'riderfix4' in name_lower:
        return 'Final'
    elif 'stage1' in name_lower or 'stage2' in name_lower or 'stage3' in name_lower:
        return 'Baseline'
    elif 'train_auto_v1' in name_lower or 'fast2' in name_lower or 'yolov5s_v1' in name_lower:
        return 'Experimental'
    elif 'val2' in name_lower or 'val4' in name_lower or 'validation' in name_lower or 'train3' in name_lower or 'train4' in name_lower or 'train6' in name_lower:
        return 'Archived'
    return 'Experimental'

def generate_experiments_md(experiments, output_path):
    md = "# Experiments Management\n\n"
    md += "This document tracks all historical and active object detection training runs in the repository.\n\n"
    
    md += "## Repository Experiment Overview\n"
    md += "The repository iteratively trains YOLO models using a multi-stage approach. Early stages act as pre-training on base datasets, while later stages focus on mined hard examples and heavily balanced custom datasets.\n\n"
    
    md += "## Training Progression\n"
    md += "1. **Stage1:** Initial transfer learning from COCO.\n"
    md += "2. **Stage2:** Extended fine-tuning on the base dataset.\n"
    md += "3. **Stage3:** Training exclusively on mined hard examples (false negatives and low confidence detections).\n"
    md += "4. **Stage4:** Final tuning on the fully merged, auto-rickshaw integrated, and rider-balanced dataset.\n\n"
    
    md += "## Reproducibility Instructions\n"
    md += "To reproduce any experiment, locate its `args.yaml` or `opt.yaml` in its respective run directory, and supply those arguments to the Ultralytics CLI or the YOLOv5 train orchestrator.\n\n"
    
    final_runs = [e for e in experiments if e['Category'] == 'Final']
    baseline_runs = [e for e in experiments if e['Category'] == 'Baseline']
    exp_runs = [e for e in experiments if e['Category'] == 'Experimental']
    archived_runs = [e for e in experiments if e['Category'] == 'Archived']
    
    md += "## Final Production Model\n"
    for r in final_runs:
        md += f"- **{r['Experiment Name']}**: Dataset: {r['Dataset Name']}, mAP50: {r['mAP50']}\n"
    
    md += "\n## Baselines\n"
    for r in baseline_runs:
        md += f"- **{r['Experiment Name']}**: Dataset: {r['Dataset Name']}, mAP50: {r['mAP50']}\n"
        
    md += "\n## Experimental Runs\n"
    for r in exp_runs:
        md += f"- **{r['Experiment Name']}**: Dataset: {r['Dataset Name']}, mAP50: {r['mAP50']}\n"
        
    md += "\n## Archived Experiments\n"
    for r in archived_runs:
        md += f"- **{r['Experiment Name']}**\n"
        
    with open(output_path, "w") as f:
        f.write(md)
    logging.info(f"Generated {output_path}")

def main():
    root_dir = Path(__file__).resolve().parents[1]
    
    runs_dir = root_dir / "runs"
    exp_dir = root_dir / "experiments"
    
    search_dirs = []
    if runs_dir.exists(): search_dirs.append(runs_dir)
    if exp_dir.exists(): search_dirs.append(exp_dir)
    
    experiments = []
    
    for base_dir in search_dirs:
        for run_path in base_dir.rglob("*"):
            if not run_path.is_dir():
                continue
                
            args_path = run_path / "args.yaml"
            opt_path = run_path / "opt.yaml"
            csv_path = run_path / "results.csv"
            
            if args_path.exists() or opt_path.exists():
                # Avoid capturing nested subdirectories if they don't contain the actual run files
                if not csv_path.exists():
                    # We might have an active run that hasn't produced a CSV yet, or a crashed run
                    continue
                    
                exp_name = run_path.name
                
                params = {}
                if args_path.exists():
                    params = parse_yolov8_args(args_path)
                elif opt_path.exists():
                    params = parse_yolov5_opt(opt_path)
                    
                metrics = parse_results_csv(csv_path)
                
                best_ckpt = run_path / "weights" / "best.pt"
                last_ckpt = run_path / "weights" / "last.pt"
                
                exp_data = {
                    "Experiment Name": exp_name,
                    "Category": categorize_experiment(exp_name),
                    "Model": params.get('model', 'Unknown'),
                    "Dataset YAML": params.get('dataset', 'Unknown'),
                    "Dataset Name": Path(str(params.get('dataset', 'Unknown'))).stem,
                    "Epochs": params.get('epochs', 'Unknown'),
                    "Image Size": params.get('imgsz', 'Unknown'),
                    "Batch Size": params.get('batch', 'Unknown'),
                    "Optimizer": params.get('optimizer', 'Unknown'),
                    "Learning Rate": params.get('lr', 'Unknown'),
                    "Device": params.get('device', 'Unknown'),
                    "Training Date": "Unknown",
                    "Best Checkpoint": "Yes" if best_ckpt.exists() else "No",
                    "Last Checkpoint": "Yes" if last_ckpt.exists() else "No",
                    "Precision": round(metrics.get('precision', 0), 4),
                    "Recall": round(metrics.get('recall', 0), 4),
                    "mAP50": round(metrics.get('map50', 0), 4),
                    "mAP50-95": round(metrics.get('map50_95', 0), 4),
                    "Fitness": round(metrics.get('fitness', 0), 4),
                    "Training Time": "Unknown",
                    "Number of Parameters": "Unknown",
                    "Notes": ""
                }
                
                # Deduplicate if we somehow scan the same dir (shouldn't happen with strict glob)
                experiments.append(exp_data)
                
    if not experiments:
        logging.warning("No experiments found.")
        return
        
    df = pd.DataFrame(experiments)
    df = df.sort_values(by="mAP50", ascending=False)
    
    eval_dir = root_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    csv_out = eval_dir / "master_results.csv"
    df.to_csv(csv_out, index=False)
    logging.info(f"Generated {csv_out}")
    
    md_out = eval_dir / "master_results.md"
    with open(md_out, "w") as f:
        f.write("# Master Experiment Results\n\n")
        f.write("All completed experiments, sorted by mAP50.\n\n")
        
        # Manual markdown table generation to avoid 'tabulate' dependency
        columns = df.columns.tolist()
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---" for _ in columns]) + "|\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")
            
    logging.info(f"Generated {md_out}")
    
    docs_dir = root_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    exp_md_out = docs_dir / "EXPERIMENTS.md"
    
    generate_experiments_md(experiments, exp_md_out)
    
if __name__ == "__main__":
    main()
