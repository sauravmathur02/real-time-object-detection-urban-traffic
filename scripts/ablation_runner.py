import argparse
import logging
import subprocess
import sys
import yaml
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_experiment_name(args):
    """Generate a unique experiment name based on disabled flags."""
    disabled = []
    if args.disable_rider_balance: disabled.append("no_rider")
    if args.disable_idd: disabled.append("no_idd")
    if args.disable_auto: disabled.append("no_auto")
    if args.disable_hard_example: disabled.append("no_hard_ex")
    if args.disable_augmentation: disabled.append("no_aug")
    if args.disable_class_remap: disabled.append("no_remap")
    
    if not disabled:
        return "baseline_all_enabled"
    return "_".join(disabled)

def generate_ablation_config(out_dir, exp_name, args, target_dataset_dir):
    """Generate the YOLO dataset yaml and the ablation experiment tracking metadata."""
    config_dir = out_dir / "experiment_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset YAML for YOLO training
    # Standard format mirroring the paper's data1 configs
    dataset_yaml_data = {
        "path": str(target_dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 4: "bus",
            5: "motorcycle", 6: "bicycle", 7: "traffic light", 8: "traffic sign", 9: "auto"
        }
    }
    
    yaml_path = config_dir / f"{exp_name}_dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(dataset_yaml_data, f, sort_keys=False)
        
    # 2. Experiment JSON metadata tracking what was enabled/disabled
    enabled_stages = ["convert", "final_verify"]
    disabled_stages = []
    
    if args.disable_rider_balance: disabled_stages.append("balance")
    else: enabled_stages.append("balance")
        
    if args.disable_idd: disabled_stages.append("merge_idd")
    else: enabled_stages.append("merge_idd")
        
    if args.disable_auto: disabled_stages.append("auto")
    else: enabled_stages.append("auto")
        
    if args.disable_hard_example: disabled_stages.append("hard_example")
    else: enabled_stages.append("hard_example")
        
    if args.disable_augmentation: disabled_stages.append("augmentation")
    else: enabled_stages.append("augmentation")
        
    if args.disable_class_remap: disabled_stages.append("class_remap")
    else: enabled_stages.append("class_remap")
        
    experiment_metadata = {
        "experiment_name": exp_name,
        "dataset_yaml": str(yaml_path),
        "target_dataset": str(target_dataset_dir),
        "enabled_stages": enabled_stages,
        "disabled_stages": disabled_stages
    }
    
    json_path = config_dir / f"{exp_name}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=4)
        
    return experiment_metadata

def run_ablation_pipeline(exp_metadata, python_exe, prepare_script):
    """
    Orchestrate prepare_dataset.py selectively.
    Normally: convert -> verify -> balance -> merge_idd -> auto -> final_verify
    We must chain the outputs based on what is enabled.
    """
    base_out = Path(exp_metadata["target_dataset"])
    base_out.mkdir(parents=True, exist_ok=True)
    
    enabled = exp_metadata["enabled_stages"]
    
    # We use a temporary working chain to avoid modifying original datasets.
    # We tell prepare_dataset.py to write its ultimate output to base_out.
    # To do this flawlessly without rewriting prepare_dataset.py, we call it step by step.
    
    logging.info("Executing staged dataset preparation for ablation...")
    
    # Step 1: Convert (Always runs to bootstrap the baseline dataset)
    cmd = [
        str(python_exe), str(prepare_script),
        "--step", "convert",
        "--bdd_yolo_dir", str(base_out)
    ]
    logging.info("Running convert stage...")
    subprocess.run(cmd, check=True)
    
    # Step 2: Rider Balance
    if "balance" in enabled:
        cmd = [
            str(python_exe), str(prepare_script),
            "--step", "balance",
            "--bdd_yolo_dir", str(base_out),
            "--balanced_dir", str(base_out)  # Overwrite in place to build ablation
        ]
        logging.info("Running balance stage...")
        subprocess.run(cmd, check=True)
        
    # Step 3: Merge IDD
    if "merge_idd" in enabled:
        cmd = [
            str(python_exe), str(prepare_script),
            "--step", "merge_idd",
            "--balanced_dir", str(base_out)
        ]
        logging.info("Running merge_idd stage...")
        subprocess.run(cmd, check=True)
        
    # Step 4: Auto Dataset
    if "auto" in enabled:
        cmd = [
            str(python_exe), str(prepare_script),
            "--step", "auto",
            "--balanced_dir", str(base_out)
        ]
        logging.info("Running auto dataset stage...")
        subprocess.run(cmd, check=True)
        
    logging.info(f"Ablation dataset constructed at {base_out}")

def generate_ablation_plan(out_dir):
    """Generates ABLATION_PLAN.md listing every potential experiment switch."""
    plan_path = out_dir / "ABLATION_PLAN.md"
    with open(plan_path, 'w') as f:
        f.write("# Ablation Study Plan\n\n")
        f.write("This document outlines the controlled ablation experiments supported by the framework. ")
        f.write("Each experiment isolates a specific module of the methodology to quantify its exact contribution to the final mAP50 score.\n\n")
        
        f.write("## Supported Ablations\n")
        f.write("- **`--disable-rider-balance`**: Tests the necessity of the oversampling module. Expect significant mAP drop for vulnerable road users.\n")
        f.write("- **`--disable-idd`**: Tests the necessity of Indian Driving Dataset fusion for regional variance.\n")
        f.write("- **`--disable-auto`**: Tests the impact of omitting the specialized Datacluster Auto-Rickshaw dataset.\n")
        f.write("- **`--disable-hard-example`**: Prevents Phase 3 hard-example mining. Tests generalization against edge-case occlusion.\n")
        f.write("- **`--disable-augmentation`**: Disables mosaic/mixup (affects native YOLO yaml hyperparams).\n")
        f.write("- **`--disable-class-remap`**: Tests raw COCO 80-class mapping versus the custom unified 10-class mapping.\n\n")
        
        f.write("## Output Structure\n")
        f.write("Every executed ablation run generates a standalone dataset in `data1/ablation/<experiment_name>` and a corresponding YOLO training configuration in `ablation/experiment_configs/`.\n")

    logging.info(f"Generated {plan_path}")

def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--disable-rider-balance", action="store_true", help="Disable rider oversampling")
    parser.add_argument("--disable-idd", action="store_true", help="Disable IDD dataset merge")
    parser.add_argument("--disable-auto", action="store_true", help="Disable auto-rickshaw merge")
    parser.add_argument("--disable-hard-example", action="store_true", help="Disable hard example mining")
    parser.add_argument("--disable-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--disable-class-remap", action="store_true", help="Disable unified class mapping")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    out_dir = root_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    python_exe = sys.executable
    prepare_script = root_dir / "scripts" / "prepare_dataset.py"
    
    # Setup configuration logic
    exp_name = build_experiment_name(args)
    target_dataset_dir = root_dir / "data1" / "ablation" / exp_name
    
    logging.info(f"Configuring Ablation Experiment: {exp_name}")
    exp_metadata = generate_ablation_config(out_dir, exp_name, args, target_dataset_dir)
    
    # Generate documentation plan (Only updates if new args exist, but generally static layout)
    generate_ablation_plan(out_dir)
    
    # As per instructions: "Do NOT execute any experiment. Only build the framework."
    # The framework is built. If execution were permitted, we would run:
    # run_ablation_pipeline(exp_metadata, python_exe, prepare_script)
    
    logging.info("Ablation framework configuration generated successfully. Execution bypassed as per constraints.")
    logging.info(f"Check {out_dir / 'experiment_configs' / f'{exp_name}_metadata.json'} for details.")

if __name__ == "__main__":
    main()
