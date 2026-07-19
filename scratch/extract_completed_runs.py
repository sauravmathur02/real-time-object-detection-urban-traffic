import json
import yaml
import os

workspace = r"c:\Repo\object-Detection"

with open(os.path.join(workspace, "scratch", "forensic_full.json"), "r", encoding="utf-8") as f:
    runs = json.load(f)

completed_runs = [r for r in runs if "Completed" in r['status']]

print(f"Total Completed Runs: {len(completed_runs)}")

full_completed_details = []

for r in completed_runs:
    run_dir = os.path.join(workspace, r['rel_path'])
    args_yaml = os.path.join(run_dir, "args.yaml")
    opt_yaml = os.path.join(run_dir, "opt.yaml")
    
    args_dict = {}
    if os.path.exists(args_yaml):
        with open(args_yaml, "r", encoding="utf-8", errors="ignore") as f:
            args_dict = yaml.safe_load(f) or {}
    elif os.path.exists(opt_yaml):
        with open(opt_yaml, "r", encoding="utf-8", errors="ignore") as f:
            args_dict = yaml.safe_load(f) or {}

    # Extract hyperparameters
    epochs = r['epochs_cfg']
    batch = r['batch']
    imgsz = r['imgsz']
    optimizer = args_dict.get("optimizer", args_dict.get("adam", "auto (SGD)"))
    lr0 = args_dict.get("lr0", args_dict.get("lr0", "0.01"))
    lrf = args_dict.get("lrf", "0.01")
    scheduler = "Linear / Cosine" if args_dict.get("cos_lr") else "Linear (default)"
    weight_decay = args_dict.get("weight_decay", "0.0005")
    device = args_dict.get("device", "0 (GPU)")
    amp = args_dict.get("amp", True)
    workers = args_dict.get("workers", 2)
    seed = args_dict.get("seed", 0)

    # Metrics
    m = r['metrics']
    precision = m.get("metrics/precision(B)", m.get("Precision", "N/A"))
    recall = m.get("metrics/recall(B)", m.get("Recall", "N/A"))
    map50 = m.get("metrics/mAP50(B)", m.get("mAP50", "N/A"))
    map5095 = m.get("metrics/mAP50-95(B)", m.get("mAP50-95", "N/A"))
    
    train_box_loss = m.get("train/box_loss", "N/A")
    train_cls_loss = m.get("train/cls_loss", "N/A")
    train_dfl_loss = m.get("train/dfl_loss", "N/A")
    
    val_box_loss = m.get("val/box_loss", "N/A")
    val_cls_loss = m.get("val/cls_loss", "N/A")
    val_dfl_loss = m.get("val/dfl_loss", "N/A")

    full_completed_details.append({
        "rel_path": r['rel_path'],
        "folder_name": r['folder_name'],
        "model": r['model'],
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "device": device,
        "amp": amp,
        "workers": workers,
        "seed": seed,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map5095": map5095,
        "best_epoch": epochs if isinstance(epochs, int) else "N/A",
        "final_epoch": r['epochs_done'],
        "train_box_loss": train_box_loss,
        "train_cls_loss": train_cls_loss,
        "train_dfl_loss": train_dfl_loss,
        "val_box_loss": val_box_loss,
        "val_cls_loss": val_cls_loss,
        "val_dfl_loss": val_dfl_loss,
        "best_checkpoint": os.path.join(r['rel_path'], "weights", "best.pt"),
        "results_csv": os.path.join(r['rel_path'], "results.csv" if os.path.exists(os.path.join(run_dir, "results.csv")) else "results.txt")
    })

with open(os.path.join(workspace, "scratch", "completed_runs_detailed.json"), "w", encoding="utf-8") as f:
    json.dump(full_completed_details, f, indent=2)

print("Saved detailed completed runs info to scratch/completed_runs_detailed.json")
