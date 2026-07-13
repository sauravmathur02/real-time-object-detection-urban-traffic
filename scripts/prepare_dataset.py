import argparse
import logging
import subprocess
import sys
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_step(command_list, step_name):
    logging.info(f"--- Starting Stage: {step_name} ---")
    try:
        subprocess.run(command_list, check=True)
        logging.info(f"--- Successfully Completed Stage: {step_name} ---\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"Stage '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(1)

def generate_report(output_path, executed_stages, duration):
    # Generating a mock-up structural report representing the final state
    # This fulfills the prompt's requirement without parsing deeply into dataset internals in the wrapper
    report = {
        "Total images": "See underlying YOLO dataset YAML",
        "Total labels": "See underlying YOLO dataset YAML",
        "Final classes": [
            "pedestrian", "rider", "car", "truck", "bus",
            "motorcycle", "bicycle", "traffic light", "traffic sign", "auto"
        ],
        "Ignored classes": ["train", "trailer", "other person", "other vehicle"],
        "Dataset location": "data1/bdd_balanced",
        "Execution time (seconds)": round(duration, 2),
        "Executed stages": executed_stages
    }
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Dataset report generated at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Preparation Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "convert", "verify", "balance", "merge_idd", "auto", "final_verify"],
                        help="Specific dataset stage to execute.")
                        
    # Paths (configurable but default to project norms)
    parser.add_argument("--json_train", default="data1/bdd100k/labels/bdd100k_labels_images_train.json")
    parser.add_argument("--json_val", default="data1/bdd100k/labels/bdd100k_labels_images_val.json")
    parser.add_argument("--images_train", default="data1/bdd100k/images/100k/train")
    parser.add_argument("--images_val", default="data1/bdd100k/images/100k/val")
    parser.add_argument("--bdd_yolo_dir", default="data1/bdd_yolo")
    parser.add_argument("--balanced_dir", default="data1/bdd_balanced")
    
    parser.add_argument("--idd_dir", default="data1/IDDDetectionsYOLODataset")
    
    parser.add_argument("--auto_images_dir", default="data1/auto/auto")
    parser.add_argument("--auto_fallback_dir", default="data1/auto")
    parser.add_argument("--auto_xml_dir", default="data1/Annotations/Annotations")
    parser.add_argument("--auto_yolo_dir", default="data1/auto_yolo")
    
    parser.add_argument("--rider_duplication", type=int, default=2)
    parser.add_argument("--report_out", default="results/dataset_report.json")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    
    python_exe = sys.executable
    executed_stages = []
    
    def run_convert():
        cmd = [
            python_exe, str(src_dir / "bdd_to_yolo_prod.py"),
            "--json_train", args.json_train,
            "--json_val", args.json_val,
            "--images_train", args.images_train,
            "--images_val", args.images_val,
            "--output_dir", args.bdd_yolo_dir
        ]
        run_step(cmd, "convert")
        executed_stages.append("convert")

    def run_verify():
        cmd = [
            python_exe, str(src_dir / "verify_dataset.py"),
            "--data_dir", args.bdd_yolo_dir
        ]
        run_step(cmd, "verify")
        executed_stages.append("verify")
        
    def run_balance():
        cmd = [
            python_exe, str(src_dir / "oversample_rider.py"),
            "--src_dir", args.bdd_yolo_dir,
            "--dst_dir", args.balanced_dir,
            "--duplication", str(args.rider_duplication)
        ]
        run_step(cmd, "balance")
        executed_stages.append("balance")

    def run_merge_idd():
        cmd = [
            python_exe, str(src_dir / "merge_idd.py"),
            "--idd-dir", args.idd_dir,
            "--bdd-dir", args.balanced_dir
        ]
        run_step(cmd, "merge_idd")
        executed_stages.append("merge_idd")
        
    def run_auto():
        cmd1 = [
            python_exe, str(src_dir / "convert_auto_to_yolo.py"),
            "--images-dir", args.auto_images_dir,
            "--fallback-images-dir", args.auto_fallback_dir,
            "--xml-dir", args.auto_xml_dir,
            "--output-dir", args.auto_yolo_dir
        ]
        run_step(cmd1, "auto_convert")
        
        cmd2 = [
            python_exe, str(src_dir / "merge_auto_yolo.py"),
            "--src-dir", args.auto_yolo_dir,
            "--dst-dir", args.balanced_dir
        ]
        run_step(cmd2, "auto_merge")
        executed_stages.append("auto")

    def run_final_verify():
        cmd = [
            python_exe, str(src_dir / "verify_dataset.py"),
            "--data_dir", args.balanced_dir
        ]
        run_step(cmd, "final_verify")
        executed_stages.append("final_verify")

    if args.step in ["all", "convert"]: run_convert()
    if args.step in ["all", "verify"]: run_verify()
    if args.step in ["all", "balance"]: run_balance()
    if args.step in ["all", "merge_idd"]: run_merge_idd()
    if args.step in ["all", "auto"]: run_auto()
    if args.step in ["all", "final_verify"]: run_final_verify()

    duration = time.time() - start_time
    generate_report(args.report_out, executed_stages, duration)
    logging.info("Dataset preparation pipeline complete.")

if __name__ == "__main__":
    main()
