import argparse
import logging
import yaml
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_label_path(image_path, images_dir_name="images", labels_dir_name="labels"):
    """Infer YOLO label path from image path."""
    try:
        # Simple replacement of images with labels in the path hierarchy
        parts = list(image_path.parts)
        if images_dir_name in parts:
            idx = len(parts) - 1 - parts[::-1].index(images_dir_name)
            parts[idx] = labels_dir_name
        return Path(*parts).with_suffix('.txt')
    except Exception:
        # Fallback heuristic
        return image_path.parent.parent / "labels" / image_path.stem + ".txt"

def get_dir_size(path):
    total = 0
    for p in Path(path).rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total

def analyze_dataset(yaml_path):
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    yaml_dir = yaml_path.parent
    dataset_root = Path(data_config.get('path', yaml_dir))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_dir / dataset_root).resolve()
        
    train_path = dataset_root / data_config.get('train', 'images/train')
    val_path = dataset_root / data_config.get('val', 'images/val')
    class_names = data_config.get('names', {})
    
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    num_classes = len(class_names)
    
    stats = {
        "Total images": 0,
        "Training images": 0,
        "Validation images": 0,
        "Number of labels (Total Bounding Boxes)": 0,
        "Number of classes": num_classes,
        "Class names": list(class_names.values()),
        "Instances per class": {c: 0 for c in class_names.values()},
        "Images per class": {c: 0 for c in class_names.values()},
        "Missing labels": 0,
        "Empty images": 0,
        "Corrupted images": 0,
        "Average objects per image": 0.0,
        "Maximum objects in one image": 0,
        "Minimum objects in one image": float('inf'),
        "Median objects per image": 0.0,
        "Dataset disk size (MB)": 0.0
    }
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def process_split(split_path):
        count = 0
        objects_per_img = []
        if not split_path.exists():
            logging.warning(f"Split path missing: {split_path}")
            return count, objects_per_img
            
        for img_path in split_path.rglob('*'):
            if img_path.suffix.lower() in valid_exts:
                count += 1
                
                # Check corruption (shallow check by reading header using cv2 if possible, or just file size)
                if img_path.stat().st_size == 0:
                    stats["Corrupted images"] += 1
                    continue
                    
                label_path = get_label_path(img_path)
                
                if not label_path.exists():
                    stats["Missing labels"] += 1
                    stats["Empty images"] += 1
                    objects_per_img.append(0)
                    continue
                    
                with open(label_path, 'r') as lf:
                    lines = [line.strip() for line in lf.readlines() if line.strip()]
                    
                obj_count = len(lines)
                objects_per_img.append(obj_count)
                stats["Number of labels (Total Bounding Boxes)"] += obj_count
                
                if obj_count == 0:
                    stats["Empty images"] += 1
                    
                classes_in_this_img = set()
                for line in lines:
                    parts = line.split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            cls_name = class_names.get(cls_id, f"Unknown_{cls_id}")
                            
                            if cls_name in stats["Instances per class"]:
                                stats["Instances per class"][cls_name] += 1
                            else:
                                stats["Instances per class"][cls_name] = 1
                                
                            classes_in_this_img.add(cls_name)
                        except ValueError:
                            pass
                            
                for c in classes_in_this_img:
                    if c in stats["Images per class"]:
                        stats["Images per class"][c] += 1
                    else:
                        stats["Images per class"][c] = 1

        return count, objects_per_img

    logging.info("Processing training split...")
    tr_count, tr_objs = process_split(train_path)
    stats["Training images"] = tr_count
    
    logging.info("Processing validation split...")
    val_count, val_objs = process_split(val_path)
    stats["Validation images"] = val_count
    
    stats["Total images"] = tr_count + val_count
    all_objs = tr_objs + val_objs
    
    if all_objs:
        stats["Average objects per image"] = float(np.mean(all_objs))
        stats["Maximum objects in one image"] = int(np.max(all_objs))
        stats["Minimum objects in one image"] = int(np.min(all_objs))
        stats["Median objects per image"] = float(np.median(all_objs))
    else:
        stats["Minimum objects in one image"] = 0
        
    try:
        size_bytes = get_dir_size(dataset_root)
        stats["Dataset disk size (MB)"] = size_bytes / (1024 * 1024)
    except Exception as e:
        logging.warning(f"Failed to calculate disk size: {e}")
        
    # Class imbalance ratio (Max instances / Min instances)
    instances = list(stats["Instances per class"].values())
    if instances and min(instances) > 0:
        stats["Class imbalance ratio"] = float(max(instances) / min(instances))
    else:
        stats["Class imbalance ratio"] = float('inf')

    return stats

def generate_reports(stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON
    with open(out_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    # 2. CSV
    # Flatten dict for CSV
    flat_stats = {k: v for k, v in stats.items() if not isinstance(v, (dict, list))}
    df_flat = pd.DataFrame([flat_stats])
    df_flat.to_csv(out_dir / "dataset_statistics.csv", index=False)
    
    # 3. MD
    with open(out_dir / "dataset_statistics.md", "w") as f:
        f.write("# Dataset Statistics\n\n")
        f.write("## Global Metrics\n")
        for k, v in flat_stats.items():
            f.write(f"- **{k}**: {v}\n")
            
        f.write("\n## Instances Per Class\n")
        f.write("| Class | Instances | Images |\n")
        f.write("|---|---|---|\n")
        for c in stats["Class names"]:
            inst = stats["Instances per class"].get(c, 0)
            img = stats["Images per class"].get(c, 0)
            f.write(f"| {c} | {inst} | {img} |\n")
            
    logging.info(f"Generated textual reports in {out_dir}")

def generate_figures(stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    classes = stats["Class names"]
    instances = [stats["Instances per class"].get(c, 0) for c in classes]
    images = [stats["Images per class"].get(c, 0) for c in classes]
    
    # Figure 1: Class Distribution (Instances)
    plt.figure(figsize=(12, 6), dpi=300)
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, instances, color='#1f77b4', edgecolor='black')
    plt.yticks(y_pos, classes, fontsize=10)
    plt.xlabel('Number of Bounding Boxes (Instances)', fontsize=12, fontweight='bold')
    plt.title('Instances Per Class Distribution', fontsize=16, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(out_dir / "class_distribution.png", bbox_inches='tight')
    plt.close()
    
    # Figure 2: Images Per Class
    plt.figure(figsize=(12, 6), dpi=300)
    plt.barh(y_pos, images, color='#ff7f0e', edgecolor='black')
    plt.yticks(y_pos, classes, fontsize=10)
    plt.xlabel('Number of Images Containing Class', fontsize=12, fontweight='bold')
    plt.title('Images Per Class', fontsize=16, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(out_dir / "images_per_class.png", bbox_inches='tight')
    plt.close()

    logging.info(f"Generated visual figures in {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Automatic Dataset Statistics Generator")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--out_dir", type=str, default="dataset_statistics", help="Output directory path")
    args = parser.parse_args()
    
    yaml_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    
    if not yaml_path.exists():
        logging.error(f"Dataset YAML not found at {yaml_path}")
        return
        
    logging.info(f"Starting analysis for dataset: {yaml_path.name}")
    stats = analyze_dataset(yaml_path)
    
    generate_reports(stats, out_dir)
    generate_figures(stats, out_dir)
    
    logging.info("Dataset statistics generation complete.")

if __name__ == "__main__":
    main()
