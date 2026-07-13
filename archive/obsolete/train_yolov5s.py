import subprocess
import sys
import os
from pathlib import Path
import yaml

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Define paths
    python_exe = project_root / "venv_gpu" / "Scripts" / "python.exe"
    yolov5_train_script = project_root / "frameworks" / "yolov5" / "yolov5_src" / "train.py"
    data_yaml = project_root / "data1" / "bdd_balanced.yaml"
    weights = project_root / "yolov5s.pt"
    project_dir = project_root / "experiments" / "yolov5"
    run_name = "train_yolov5s_v1"
    
    epochs = 50
    imgsz = 640
    batch_size = 16
    
    print(f"Starting YOLOv5s training on BDD100K + IDD + Auto dataset...")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size}")
    print(f"Weights path: {weights}")
    print(f"Data config: {data_yaml}")
    print(f"Project directory: {project_dir}")
    print(f"Run name: {run_name}")
    
    cmd = [
        str(python_exe),
        str(yolov5_train_script),
        "--img", str(imgsz),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--project", str(project_dir),
        "--name", run_name,
        "--device", "0"
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # Run the training process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream the output
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error: YOLOv5 training failed with exit code {return_code}")
        sys.exit(return_code)
        
    print("\nTraining completed successfully!")
    
    # Automatically generate YOLOv8-compatible args.yaml
    run_dir = project_dir / run_name
    opt_yaml_path = run_dir / "opt.yaml"
    args_yaml_path = run_dir / "args.yaml"
    
    if opt_yaml_path.exists():
        try:
            with open(opt_yaml_path, "r") as f:
                opt_data = yaml.safe_load(f)
                
            # Map opt.yaml properties to args.yaml format
            args_data = {
                "task": "detect",
                "mode": "train",
                "model": opt_data.get("weights"),
                "data": opt_data.get("data"),
                "epochs": opt_data.get("epochs"),
                "patience": opt_data.get("patience"),
                "batch": opt_data.get("batch_size"),
                "imgsz": opt_data.get("imgsz"),
                "save": not opt_data.get("nosave", False),
                "device": opt_data.get("device"),
                "workers": opt_data.get("workers"),
                "project": opt_data.get("project"),
                "name": opt_data.get("name"),
                "exist_ok": opt_data.get("exist_ok"),
                "optimizer": opt_data.get("optimizer"),
                "seed": opt_data.get("seed"),
                "cos_lr": opt_data.get("cos_lr"),
                "lr0": opt_data.get("lr0", 0.01),
                "weight_decay": opt_data.get("weight_decay", 0.0005)
            }
            
            with open(args_yaml_path, "w") as f_out:
                yaml.safe_dump(args_data, f_out, default_flow_style=False)
                
            print(f"Generated YOLOv8-compatible args.yaml at: {args_yaml_path}")
        except Exception as e:
            print(f"Error generating args.yaml: {e}")
    else:
        print(f"Warning: could not locate opt.yaml at {opt_yaml_path}")

if __name__ == "__main__":
    main()
