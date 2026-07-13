import argparse
import logging
import subprocess
import sys
import json
import csv
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_MODELS = [
    "yolov5s", "yolov5m",
    "yolov7x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l"
]

def parse_yolov5_v7_results(log_dir, captured_stdout, class_names):
    """Attempt to parse validation metrics from YOLOv5/v7."""
    metrics = {}
    per_class = {}
    
    # 1. Basic fallback: results.txt often has the total line at the bottom
    results_txt = log_dir / "results.txt"
    if results_txt.exists():
        with open(results_txt, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[-1].strip().split()
                if len(parts) >= 7:
                    metrics['Precision'] = parts[3]
                    metrics['Recall'] = parts[4]
                    metrics['mAP50'] = parts[5]
                    metrics['mAP50-95'] = parts[6]
                    
    # 2. Aggressively parse stdout for per-class tabular breakdown
    # YOLO prints a table like:
    # Class     Images  Instances          P          R      mAP50   mAP50-95:
    # all          100       1000      0.800      0.700      0.750      0.500
    # pedestrian   100        200      0.750      0.710      0.720      0.450
    if captured_stdout:
        lines = captured_stdout.split('\n')
        parsing_table = False
        for line in lines:
            line_clean = line.strip()
            if "Class" in line_clean and "Images" in line_clean and "Instances" in line_clean:
                parsing_table = True
                continue
            
            if parsing_table and line_clean:
                parts = line_clean.split()
                if len(parts) >= 6:
                    cls_name = parts[0]
                    if cls_name == 'all':
                        # Overall metrics usually populated from results.txt, but we can override here
                        metrics['Precision'] = parts[3]
                        metrics['Recall'] = parts[4]
                        metrics['mAP50'] = parts[5]
                        metrics['mAP50-95'] = parts[6] if len(parts) > 6 else "N/A"
                    else:
                        # Per class metrics
                        per_class[cls_name] = {
                            "Precision": float(parts[3]) if parts[3].replace('.','',1).isdigit() else "N/A",
                            "Recall": float(parts[4]) if parts[4].replace('.','',1).isdigit() else "N/A",
                            "AP": float(parts[5]) if parts[5].replace('.','',1).isdigit() else "N/A",
                            "Support": int(parts[2]) if parts[2].isdigit() else "N/A",
                            "Mean Confidence": "N/A" # Usually missing in val.py table
                        }
            elif parsing_table and not line_clean:
                parsing_table = False # End of table
                
    if per_class:
        metrics['per_class'] = per_class
    else:
        # Fallback to N/A for known classes if not found
        metrics['per_class'] = {c: {"AP": "N/A", "Precision": "N/A", "Recall": "N/A", "Support": "N/A"} for c in class_names}
        
    return metrics

def evaluate_yolov5(model_name, args, eval_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov5" / "yolov5_src" / "val.py"
    if not script_path.exists():
        logging.error(f"YOLOv5 val.py missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--img", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--data", args.dataset,
        "--weights", args.weights or f"{model_name}.pt",
        "--project", str(eval_dir.parent),
        "--name", eval_dir.name,
    ]
    if args.device: cmd.extend(["--device", str(args.device)])
    if args.workers: cmd.extend(["--workers", str(args.workers)])
    if args.save_json: cmd.append("--save-json")
    if args.save_txt: cmd.append("--save-txt")
    
    return cmd

def evaluate_yolov7(model_name, args, eval_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov7" / "test.py"
    if not script_path.exists():
        logging.error(f"YOLOv7 test.py missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--img-size", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--data", args.dataset,
        "--weights", args.weights or f"{model_name}.pt",
        "--project", str(eval_dir.parent),
        "--name", eval_dir.name,
    ]
    if args.device: cmd.extend(["--device", str(args.device)])
    if args.workers: cmd.extend(["--workers", str(args.workers)])
    if args.save_json: cmd.append("--save-json")
    if args.save_txt: cmd.append("--save-txt")
    
    return cmd

def evaluate_yolov8(model_name, args, eval_dir):
    try:
        from ultralytics import YOLO
    except ImportError:
        logging.error("Ultralytics package missing. Skipping YOLOv8.")
        return None

    # For YOLOv8, we use the Python API to get rich metrics directly
    weights = args.weights or f"{model_name}.pt"
    if not Path(weights).exists():
        logging.error(f"Weights missing at {weights} for {model_name}. Skipping.")
        return None

    logging.info(f"Loading Ultralytics YOLO model from {weights}")
    model = YOLO(weights)
    
    metrics_obj = model.val(
        data=args.dataset,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers or 8,
        project=eval_dir.parent,
        name=eval_dir.name,
        save_json=args.save_json
    )
    
    metrics_dict = {
        'Precision': metrics_obj.results_dict.get('metrics/precision(B)', 0),
        'Recall': metrics_obj.results_dict.get('metrics/recall(B)', 0),
        'mAP50': metrics_obj.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': metrics_obj.results_dict.get('metrics/mAP50-95(B)', 0),
        'Inference time (ms/img)': sum(metrics_obj.speed.values()) if hasattr(metrics_obj, 'speed') else 0
    }
    return metrics_dict

def run_evaluation(model_name, args, master_records):
    logging.info(f"--- Starting Evaluation for {model_name} ---")
    
    root_dir = Path(__file__).resolve().parents[1]
    eval_dir = root_dir / "evaluation" / model_name
    
    weights = args.weights or f"{model_name}.pt"
    if not Path(weights).exists():
        logging.warning(f"Weights missing at {weights}. Skipping {model_name}.")
        return

    metrics = {}
    cmd = None
    
    if "yolov5" in model_name:
        cmd = evaluate_yolov5(model_name, args, eval_dir)
    elif "yolov7" in model_name:
        cmd = evaluate_yolov7(model_name, args, eval_dir)
    elif "yolov8" in model_name:
        # YOLOv8 handles via Python API natively, returning metrics immediately
        metrics = evaluate_yolov8(model_name, args, eval_dir)
    else:
        logging.error(f"Unsupported model family for {model_name}.")
        return

    if cmd:
        logging.info(f"Executing: {' '.join(cmd)}")
        captured_out = ""
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                captured_out += line
                sys.stdout.flush()
            process.stdout.close()
            process.wait()
            
            # Extract class names heuristically or provide defaults
            class_names = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign", "auto"]
            metrics = parse_yolov5_v7_results(eval_dir, captured_out, class_names)
        except Exception as e:
            logging.error(f"Exception during evaluation of {model_name}: {e}")

    if metrics is not None:
        metrics['Model'] = model_name
        save_model_metrics(eval_dir, metrics)
        master_records.append(metrics)
        
def save_model_metrics(eval_dir, metrics):
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(eval_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Write CSV
    if metrics:
        df = pd.DataFrame([metrics])
        df.to_csv(eval_dir / "metrics.csv", index=False)
        
        # Write MD
        with open(eval_dir / "metrics.md", 'w') as f:
            f.write(f"# Evaluation Metrics for {metrics.get('Model')}\n\n")
            columns = df.columns.tolist()
            f.write("| " + " | ".join(columns) + " |\n")
            f.write("|" + "|".join(["---" for _ in columns]) + "|\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Model Evaluation Orchestrator")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, help="Specific model to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all supported models sequentially")
    
    parser.add_argument("--weights", type=str, help="Path to weights file (best.pt)")
    parser.add_argument("--dataset", type=str, default="data1/bdd_balanced.yaml", help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, help="CUDA device (e.g. 0 or 0,1) or cpu")
    parser.add_argument("--workers", type=int, help="Number of dataloader workers")
    
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON format")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt format")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        logging.error("Must specify either --model <name> or --all")
        sys.exit(1)
        
    models_to_eval = SUPPORTED_MODELS if args.all else [args.model]
    master_records = []
    
    for m in models_to_eval:
        run_evaluation(m, args, master_records)
        
    # Generate master metrics
    if master_records:
        master_df = pd.DataFrame(master_records)
        master_csv = Path("evaluation/master_metrics.csv")
        master_csv.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(master_csv, index=False)
        logging.info(f"Master evaluation metrics saved to {master_csv}")

if __name__ == "__main__":
    main()
