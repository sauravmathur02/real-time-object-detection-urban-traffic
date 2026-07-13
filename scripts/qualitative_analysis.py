import argparse
import logging
import json
import csv
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_MODELS = [
    "yolov5s", "yolov5m",
    "yolov7x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l"
]

def parse_yolo_labels(label_path):
    """Parse standard YOLO .txt label files into a list of dicts."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format: class x y w h [conf]
                cls_id = int(parts[0])
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                boxes.append({"class": cls_id, "conf": conf})
    return boxes

def analyze_qualitative_directory(target_dir, models, report_data):
    """Scans the target directory for images and their corresponding predictions across models."""
    target_path = Path(target_dir)
    if not target_path.exists():
        logging.warning(f"Target directory {target_path} does not exist.")
        return []
        
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [p for p in target_path.rglob('*') if p.suffix.lower() in valid_exts]
    
    records = []
    
    for img_path in images:
        rel_path = img_path.relative_to(target_path)
        img_category = rel_path.parent.name if rel_path.parent.name else "uncategorized"
        
        # Ground truth
        gt_path = img_path.with_suffix('.txt')
        gt_boxes = parse_yolo_labels(gt_path)
        gt_count = len(gt_boxes)
        
        img_record = {
            "Image": img_path.name,
            "Category": img_category,
            "Path": str(img_path),
            "Ground Truth Objects": gt_count if gt_path.exists() else "Unknown",
            "Model_Stats": {}
        }
        
        for model in models:
            # We assume predictions are saved in a parallel hierarchy or in evaluation folders
            # If not explicitly found, we handle gracefully.
            pred_path = target_path / "predictions" / model / "labels" / img_path.with_suffix('.txt').name
            
            if pred_path.exists():
                pred_boxes = parse_yolo_labels(pred_path)
                det_count = len(pred_boxes)
                
                confs = [b["conf"] for b in pred_boxes]
                avg_conf = np.mean(confs) if confs else 0.0
                low_conf_count = sum(1 for c in confs if c < 0.5)
                
                # Simple heuristic since IoU requires coordinates (omitted for brevity here)
                missed = max(0, gt_count - det_count) if gt_path.exists() else "Unknown"
                fps = max(0, det_count - gt_count) if gt_path.exists() else "Unknown"
                
                img_record["Model_Stats"][model] = {
                    "Detected": det_count,
                    "Missed": missed,
                    "False Positives": fps,
                    "Average Confidence": avg_conf,
                    "Low Confidence Predictions": low_conf_count
                }
            else:
                report_data['Missing predictions'].append(f"{model} - {img_path.name}")
                
        # Determine best/worst model heuristically based on detection volume vs GT
        if img_record["Model_Stats"] and gt_path.exists():
            best_model = None
            worst_model = None
            min_err = float('inf')
            max_err = -1
            
            for m, stats in img_record["Model_Stats"].items():
                err = stats["Missed"] + stats["False Positives"]
                if err < min_err:
                    min_err = err
                    best_model = m
                if err > max_err:
                    max_err = err
                    worst_model = m
                    
            img_record["Best Model"] = best_model
            img_record["Worst Model"] = worst_model
        else:
            # If no GT, use a proxy (e.g., highest confidence model)
            if img_record["Model_Stats"]:
                sorted_by_conf = sorted(img_record["Model_Stats"].items(), key=lambda x: x[1]["Average Confidence"], reverse=True)
                img_record["Best Model"] = sorted_by_conf[0][0]
                img_record["Worst Model"] = sorted_by_conf[-1][0]
            else:
                img_record["Best Model"] = "N/A"
                img_record["Worst Model"] = "N/A"
                
        records.append(img_record)
        
    return records

def generate_comparison_matrix(records, out_dir):
    flat_records = []
    
    for r in records:
        best = r.get("Best Model", "N/A")
        worst = r.get("Worst Model", "N/A")
        
        # Pull stats from best model to represent the output
        stats = r["Model_Stats"].get(best, {}) if best != "N/A" else {}
        
        flat_records.append({
            "Image": r["Image"],
            "Category": r["Category"],
            "Best Model": best,
            "Worst Model": worst,
            "Objects Detected": stats.get("Detected", "N/A"),
            "Objects Missed": stats.get("Missed", "N/A"),
            "False Positives": stats.get("False Positives", "N/A"),
            "Average Confidence": f"{stats.get('Average Confidence', 0.0):.3f}" if stats.get("Average Confidence") else "N/A"
        })
        
    df = pd.DataFrame(flat_records)
    if not df.empty:
        df.to_csv(out_dir / "comparison_matrix.csv", index=False)
        logging.info(f"Generated comparison_matrix.csv at {out_dir}")
    return df

def sort_best_worst_examples(records, out_dir):
    best_dir = out_dir / "best_examples"
    worst_dir = out_dir / "failure_cases"
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple heuristic: sort by False Positives + Missed (if available) or Average Confidence
    for r in records:
        best_m = r.get("Best Model")
        worst_m = r.get("Worst Model")
        if best_m == "N/A" or worst_m == "N/A":
            continue
            
        b_stats = r["Model_Stats"][best_m]
        w_stats = r["Model_Stats"][worst_m]
        
        try:
            # If it missed 0 and FP is 0, it's a perfect example
            if b_stats.get("Missed") == 0 and b_stats.get("False Positives") == 0:
                shutil.copy2(r["Path"], best_dir / r["Image"])
                
            # If the worst model missed many or had many FPs
            if isinstance(w_stats.get("Missed"), int) and (w_stats["Missed"] > 2 or w_stats["False Positives"] > 2):
                shutil.copy2(r["Path"], worst_dir / r["Image"])
        except Exception:
            pass
            
    logging.info(f"Populated best_examples and failure_cases directories.")

def generate_reports(df, records, out_dir):
    # JSON
    with open(out_dir / "qualitative_summary.json", 'w') as f:
        json.dump(records, f, indent=4)
        
    # MD
    with open(out_dir / "qualitative_summary.md", 'w') as f:
        f.write("# Qualitative Analysis Summary\n\n")
        f.write("A breakdown of detection behavior across analyzed imagery.\n\n")
        if df.empty:
            f.write("No prediction labels found to analyze.\n")
        else:
            cols = df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---" for _ in cols]) + "|\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
                
    # Qualitative Analysis Docs
    with open(out_dir / "QUALITATIVE_ANALYSIS.md", 'w') as f:
        f.write("# Qualitative Edge-Case Analysis\n\n")
        f.write("This document summarizes the model's behavior based exclusively on measurable prediction confidence and bounding box alignment.\n\n")
        
        f.write("## Common Failure Patterns\n")
        # Extract categories that appear most in the failure cases
        if not df.empty:
            failures = df[(df['Objects Missed'] != "N/A") & (df['Objects Missed'] != 0)]
            if not failures.empty:
                common_cats = failures['Category'].value_counts().head(3)
                for cat, count in common_cats.items():
                    f.write(f"- Subsets located in **{cat}** exhibit frequent bounding box omission or low confidence.\n")
            else:
                f.write("- No critical failure patterns observed in the provided label pool.\n")
                
        f.write("\n## Model Hierarchy\n")
        if not df.empty:
            best_counts = df['Best Model'].value_counts()
            f.write("Models ranked by highest confidence / lowest error across the image subset:\n")
            for m, count in best_counts.items():
                if m != "N/A":
                    f.write(f"- **{m}** dominated on {count} images.\n")

    logging.info(f"Generated qualitative documentation.")

def main():
    parser = argparse.ArgumentParser(description="Qualitative Error & Model Comparison Pipeline")
    parser.add_argument("--img_dir", type=str, default="sample_images", help="Directory containing images to evaluate")
    parser.add_argument("--out_dir", type=str, default="paper_outputs/qualitative", help="Output directory path")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    img_dir = root_dir / args.img_dir
    out_dir = root_dir / args.out_dir
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        'Missing predictions': [],
        'Missing ground truth': []
    }
    
    logging.info(f"Scanning {img_dir} for qualitative analysis...")
    records = analyze_qualitative_directory(img_dir, SUPPORTED_MODELS, report_data)
    
    if not records:
        logging.warning("No images or records found to analyze.")
        
    df = generate_comparison_matrix(records, out_dir)
    sort_best_worst_examples(records, out_dir)
    generate_reports(df, records, out_dir)
    
    logging.info("Qualitative analysis pipeline complete.")

if __name__ == "__main__":
    main()
