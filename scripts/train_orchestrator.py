import subprocess
import sys
import os
import shutil
import datetime
import platform
import json
from pathlib import Path
import yaml

def run_command(cmd, log_prefix=""):
    print(f"[{log_prefix}] Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        print(f"[{log_prefix}] {line}", end="")
        sys.stdout.flush()
        output_lines.append(line)
        
    process.stdout.close()
    return_code = process.wait()
    return return_code, "".join(output_lines)

def generate_reproducibility_metadata(run_dir, model_name, dataset_yaml_path):
    import torch
    import git
    
    timestamp = datetime.datetime.now().isoformat()
    python_ver = platform.python_version()
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit = repo.head.object.hexsha
    except Exception:
        git_commit = "Unknown"
        
    # Read dataset yaml info
    try:
        with open(dataset_yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            dataset_path = yaml_data.get("path", "Unknown")
    except Exception:
        dataset_path = str(dataset_yaml_path)

    # Read opt.yaml for hyperparameters
    opt_data = {}
    opt_yaml_path = run_dir / "opt.yaml"
    if opt_yaml_path.exists():
        try:
            with open(opt_yaml_path, "r") as f:
                opt_data = yaml.safe_load(f)
        except Exception:
            pass

    # Save environment.json
    env_data = {
        "python_version": python_ver,
        "pytorch_version": torch_ver,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "git_commit": git_commit,
        "timestamp": timestamp
    }
    
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env_data, f, indent=2)

    # Save experiment.yaml
    experiment_data = {
        "Model": model_name,
        "Dataset": str(dataset_yaml_path.name),
        "DatasetPath": dataset_path,
        "Epochs": opt_data.get("epochs", 50),
        "Optimizer": opt_data.get("optimizer", "SGD"),
        "LearningRate": opt_data.get("lr0", 0.01),
        "WeightDecay": opt_data.get("weight_decay", 0.0005),
        "BatchSize": opt_data.get("batch_size", 16),
        "ImageSize": opt_data.get("imgsz", 640),
        "Seed": opt_data.get("seed", 0),
        "GitCommit": git_commit,
        "Timestamp": timestamp,
        "PythonVersion": python_ver,
        "TorchVersion": torch_ver,
        "CUDAVersion": cuda_ver,
        "GPU": gpu_name
    }
    
    with open(run_dir / "experiment.yaml", "w") as f:
        yaml.safe_dump(experiment_data, f, default_flow_style=False)
        
    print(f"Saved reproducibility metadata (experiment.yaml & environment.json) in {run_dir}")

def generate_args_yaml(run_dir):
    opt_yaml_path = run_dir / "opt.yaml"
    args_yaml_path = run_dir / "args.yaml"
    
    if opt_yaml_path.exists():
        try:
            with open(opt_yaml_path, "r") as f:
                opt_data = yaml.safe_load(f)
                
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

def main():
    project_root = Path(__file__).resolve().parents[1]
    python_exe = project_root / "venv_gpu" / "Scripts" / "python.exe"
    yolov5_train_script = project_root / "frameworks" / "yolov5" / "yolov5_src" / "train.py"
    data_yaml = project_root / "data1" / "bdd_balanced.yaml"
    weights = project_root / "yolov5s.pt"
    project_dir = project_root / "experiments" / "yolov5"
    
    validation_name = "validation_yolov5s"
    full_train_name = "train_yolov5s_v1"
    
    # Check if validation run already exists and succeeds
    validation_dir = project_dir / validation_name
    best_val_pt = validation_dir / "weights" / "best.pt"
    opt_val_yaml = validation_dir / "opt.yaml"
    
    validation_succeeded = best_val_pt.exists() and opt_val_yaml.exists()
    
    if not validation_succeeded:
        # Clean previous validation runs if any
        if validation_dir.exists():
            shutil.rmtree(validation_dir)
            
        print("==================================================")
        print("STAGE 1: RUNNING VALIDATION EPOCH (EPOCHS = 1)")
        print("==================================================")
        
        val_cmd = [
            str(python_exe), str(yolov5_train_script),
            "--img", "640",
            "--batch-size", "16",
            "--epochs", "1",
            "--data", str(data_yaml),
            "--weights", str(weights),
            "--project", str(project_dir),
            "--name", validation_name,
            "--device", "0"
        ]
        
        val_code, val_output = run_command(val_cmd, log_prefix="VAL_RUN")
        
        if val_code != 0:
            print(f"\n[FAIL] Validation run failed with exit code: {val_code}")
            sys.exit(val_code)
            
        # Check outputs of validation run
        if not best_val_pt.exists() or not opt_val_yaml.exists():
            print(f"\n[FAIL] Validation artifacts missing. best.pt: {best_val_pt.exists()}, opt.yaml: {opt_val_yaml.exists()}")
            sys.exit(1)
            
        print("\n[SUCCESS] Stage 1 Validation Run Completed Successfully!")
    else:
        print("\n[SUCCESS] Stage 1 Validation Run has ALREADY succeeded! Reusing validated models.")
        
    print("  - Dataset loaded successfully.")
    print("  - No CUDA, out-of-memory, or dataloader errors.")
    print("  - Weights and configs generated successfully.")
    
    # Immediately proceed to full training
    print("\n==================================================")
    print("STAGE 2: LAUNCHING FULL TRAINING RUN (EPOCHS = 50)")
    print("==================================================")
    
    # Clean previous full training runs if any
    full_train_dir = project_dir / full_train_name
    if full_train_dir.exists():
        shutil.rmtree(full_train_dir)
        
    train_cmd = [
        str(python_exe), str(yolov5_train_script),
        "--img", "640",
        "--batch-size", "16",
        "--epochs", "50",
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--project", str(project_dir),
        "--name", full_train_name,
        "--device", "0"
    ]
    
    train_code, train_output = run_command(train_cmd, log_prefix="TRAIN_RUN")
    
    if train_code != 0:
        print(f"\n[FAIL] Full training run failed with exit code: {train_code}")
        sys.exit(train_code)
        
    print("\n==================================================")
    print("STAGE 3: VERIFYING ARTIFACTS AND POST-PROCESSING")
    print("==================================================")
    
    # Expected artifacts
    best_pt = full_train_dir / "weights" / "best.pt"
    last_pt = full_train_dir / "weights" / "last.pt"
    results_csv = full_train_dir / "results.csv"
    opt_yaml = full_train_dir / "opt.yaml"
    
    missing_artifacts = []
    for art in [best_pt, last_pt, results_csv, opt_yaml]:
        if not art.exists():
            missing_artifacts.append(art.name)
            
    # Check for plots (e.g. results.png)
    plots = list(full_train_dir.glob("*.png"))
    if not plots:
        missing_artifacts.append("plots (*.png)")
        
    if missing_artifacts:
        print(f"\n[FAIL] Artifact Verification Failed. Missing: {', '.join(missing_artifacts)}")
        sys.exit(1)
        
    print("[SUCCESS] All expected artifacts verified successfully:")
    print(f"  - best.pt: Found")
    print(f"  - last.pt: Found")
    print(f"  - results.csv: Found")
    print(f"  - opt.yaml: Found")
    print(f"  - Plots: Found {len(plots)} plot files")
    
    # Post-processing: generate args.yaml and metadata
    generate_args_yaml(full_train_dir)
    generate_reproducibility_metadata(full_train_dir, "YOLOv5s", data_yaml)
    
    print("\n[SUCCESS] Orchestrated Training pipeline completed successfully and reproducibility metadata generated!")

if __name__ == "__main__":
    main()
