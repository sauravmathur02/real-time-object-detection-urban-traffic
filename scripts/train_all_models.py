import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_MODELS = [
    "yolov5s", "yolov5m",
    "yolov7x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l"
]

def get_yolov5_cmd(model_name, args, project_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov5" / "yolov5_src" / "train.py"
    if not script_path.exists():
        logging.error(f"YOLOv5 framework missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--img", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", args.dataset,
        "--weights", f"{model_name}.pt",
        "--optimizer", args.optimizer,
        "--project", str(project_dir),
        "--name", f"train_{model_name}",
    ]
    
    if args.resume:
        cmd.append("--resume")
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
        
    # YOLOv5 hyperparameter overrides (usually requires a custom hyp.yaml, but some can be passed directly or we assume defaults match paper)
    # Limitation: lr0, momentum, and weight_decay cannot be passed directly via CLI to YOLOv5; requires a hyp.yaml file.
    return cmd

def get_yolov7_cmd(model_name, args, project_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov7" / "train.py"
    if not script_path.exists():
        logging.error(f"YOLOv7 framework missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--img-size", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", args.dataset,
        "--weights", str(
            Path(__file__).resolve().parents[1]
            / "frameworks"
            / "yolov7"
            / "weights"
            / f"{model_name}.pt"
        ),
        "--project", str(project_dir),
        "--name", f"train_{model_name}",
    ]
    if args.optimizer.lower() == 'adam':
        cmd.append("--adam")
        
    if args.resume:
        cmd.append("--resume")
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
        
    # Limitation: lr0, momentum, and weight_decay cannot be passed directly via CLI to YOLOv7; requires a hyp.yaml file.
    return cmd

def get_yolov8_cmd(model_name, args, project_dir):
    try:
        import ultralytics
    except ImportError:
        logging.error("Ultralytics package missing. Skipping YOLOv8.")
        return None
        
    cmd = [
        "yolo", "detect", "train",
        f"model={model_name}.pt",
        f"data={args.dataset}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"optimizer={args.optimizer}",
        f"lr0={args.lr0}",
        f"momentum={args.momentum}",
        f"weight_decay={args.weight_decay}",
        f"seed={args.seed}",
        f"project={project_dir}",
        f"name=train_{model_name}"
    ]
    
    if args.resume:
        cmd.append("resume=True")
    if args.device:
        cmd.append(f"device={args.device}")
    if args.workers:
        cmd.append(f"workers={args.workers}")
        
    return cmd

def train_model(model_name, args):
    logging.info(f"--- Preparing to train {model_name} ---")
    
    root_dir = Path(__file__).resolve().parents[1]
    
    if "yolov5" in model_name:
        project_dir = root_dir / "experiments" / "yolov5"
        cmd = get_yolov5_cmd(model_name, args, project_dir)
    elif "yolov7" in model_name:
        project_dir = root_dir / "experiments" / "yolov7"
        cmd = get_yolov7_cmd(model_name, args, project_dir)
    elif "yolov8" in model_name:
        project_dir = root_dir / "runs" / "detect"
        cmd = get_yolov8_cmd(model_name, args, project_dir)
    else:
        logging.error(f"Unsupported model family for {model_name}.")
        return

    if not cmd:
        return

    logging.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # Popen is used to stream output to console while training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
            
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logging.error(f"Training failed for {model_name} with exit code {return_code}")
        else:
            logging.info(f"--- Training completed successfully for {model_name} ---")
            
    except Exception as e:
        logging.error(f"Exception during training {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Model Training Orchestrator")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, help="Specific model to train")
    parser.add_argument("--all", action="store_true", help="Train all supported models sequentially")
    
    # Paper default hyperparameters
    parser.add_argument("--dataset", type=str, default="data1/bdd_balanced.yaml", help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer (SGD, Adam, etc.)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # New CLI features
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--device", type=str, help="CUDA device (e.g. 0 or 0,1) or cpu")
    parser.add_argument("--workers", type=int, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        logging.error("Must specify either --model <name> or --all")
        sys.exit(1)
        
    models_to_train = SUPPORTED_MODELS if args.all else [args.model]
    
    for m in models_to_train:
        train_model(m, args)
        
    logging.info("Unified orchestration completed.")

if __name__ == "__main__":
    main()
