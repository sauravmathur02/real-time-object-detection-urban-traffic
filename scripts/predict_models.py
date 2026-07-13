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

def predict_yolov5(model_name, args, out_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov5" / "yolov5_src" / "detect.py"
    if not script_path.exists():
        logging.error(f"YOLOv5 detect.py missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--source", args.source,
        "--weights", args.weights or f"{model_name}.pt",
        "--project", str(out_dir.parent),
        "--name", out_dir.name,
        "--save-txt",
        "--save-conf",
        "--exist-ok"
    ]
    return cmd

def predict_yolov7(model_name, args, out_dir):
    script_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov7" / "detect.py"
    if not script_path.exists():
        logging.error(f"YOLOv7 detect.py missing at {script_path}. Skipping.")
        return None
        
    cmd = [
        sys.executable, str(script_path),
        "--source", args.source,
        "--weights", args.weights or f"{model_name}.pt",
        "--project", str(out_dir.parent),
        "--name", out_dir.name,
        "--save-txt",
        "--save-conf",
        "--exist-ok"
    ]
    return cmd

def predict_yolov8(model_name, args, out_dir):
    try:
        from ultralytics import YOLO
    except ImportError:
        logging.error("Ultralytics package missing. Skipping YOLOv8.")
        return None

    weights = args.weights or f"{model_name}.pt"
    if not Path(weights).exists():
        logging.error(f"Weights missing at {weights} for {model_name}. Skipping.")
        return None

    logging.info(f"Loading Ultralytics YOLO model from {weights}")
    model = YOLO(weights)
    
    # YOLOv8 API execution
    model.predict(
        source=args.source,
        project=out_dir.parent,
        name=out_dir.name,
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True
    )
    return "YOLOv8_API_SUCCESS"

def run_prediction(model_name, args):
    logging.info(f"--- Starting Inference for {model_name} ---")
    
    root_dir = Path(__file__).resolve().parents[1]
    source_dir = root_dir / args.source
    if not source_dir.exists():
        logging.error(f"Source directory {source_dir} missing. Aborting.")
        return
        
    out_dir = source_dir / "predictions" / model_name
    
    weights = args.weights or f"{model_name}.pt"
    if not Path(weights).exists():
        logging.warning(f"Weights missing at {weights}. Skipping {model_name}.")
        return

    cmd = None
    if "yolov5" in model_name:
        cmd = predict_yolov5(model_name, args, out_dir)
    elif "yolov7" in model_name:
        cmd = predict_yolov7(model_name, args, out_dir)
    elif "yolov8" in model_name:
        cmd = predict_yolov8(model_name, args, out_dir)
    else:
        logging.error(f"Unsupported model family for {model_name}.")
        return

    if isinstance(cmd, list):
        logging.info(f"Executing: {' '.join(cmd)}")
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                sys.stdout.flush()
            process.stdout.close()
            process.wait()
        except Exception as e:
            logging.error(f"Exception during prediction of {model_name}: {e}")
            
    logging.info(f"Finished inference for {model_name}. Outputs routed to {out_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Unified Inference/Prediction Pipeline")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, help="Specific model to evaluate")
    parser.add_argument("--all", action="store_true", help="Run inference on all supported models sequentially")
    parser.add_argument("--weights", type=str, help="Path to weights file (best.pt)")
    parser.add_argument("--source", type=str, default="sample_images", help="Target imagery directory")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        logging.error("Must specify either --model <name> or --all")
        sys.exit(1)
        
    models_to_eval = SUPPORTED_MODELS if args.all else [args.model]
    
    for m in models_to_eval:
        run_prediction(m, args)
        
    logging.info("Prediction pipeline complete. Ready for qualitative analysis.")

if __name__ == "__main__":
    main()
