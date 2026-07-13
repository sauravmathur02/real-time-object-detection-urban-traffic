import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics is not installed. Please install it using `pip install ultralytics`")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL_DIR = PROJECT_ROOT / "data1" / "bdd_yolo" / "images" / "val"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data1" / "hard_examples"


def main():
    parser = argparse.ArgumentParser(description="Collect hard examples from a validation set using YOLO inference.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO weights.")
    parser.add_argument("--val", type=str, default=str(DEFAULT_VAL_DIR), help="Path to validation images.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Path to output hard examples.")
    parser.add_argument("--thresh", type=float, default=0.3, help="Confidence threshold below which is considered hard.")
    args = parser.parse_args()

    if YOLO is None:
        return

    model_path = Path(args.model).expanduser().resolve()
    val_dir = Path(args.val).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please point to a valid 'best.pt' file using --model.")
        return

    if not val_dir.exists():
        print(f"Error: Validation directory not found at {val_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    image_files = [f for f in os.listdir(val_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"Running inference on {len(image_files)} validation images...")

    hard_samples_count = 0
    low_conf_count = 0
    missed_det_count = 0

    results = model.predict(source=val_dir, stream=True, verbose=False)

    print("Evaluating results...")
    for result in tqdm(results, total=len(image_files), desc="Mining Hard Examples"):
        is_hard = False
        img_path = result.path

        boxes = result.boxes
        if len(boxes) == 0:
            is_hard = True
            missed_det_count += 1
        else:
            confs = boxes.conf
            if (confs < args.thresh).any():
                is_hard = True
                low_conf_count += 1

        if is_hard:
            hard_samples_count += 1
            filename = os.path.basename(img_path)
            dest_path = output_dir / filename
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)

    print("\n==========================================")
    print("HARD EXAMPLE MINING REPORT")
    print("==========================================")
    print(f"Total Validation Images Evaluated: {len(image_files)}")
    print(f"Total Hard Examples Found:         {hard_samples_count}")
    print(f"  --> Missed Detections (0 boxes): {missed_det_count}")
    print(f"  --> Low Confidence (< {args.thresh}):    {low_conf_count}")
    print(f"Hard samples safely copied to:\n  {output_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
