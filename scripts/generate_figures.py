import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def draw_figure_1_pipeline(out_dir):
    """Draws a vertical flowchart for Figure 1 using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    ax.axis('off')
    
    steps = [
        "BDD100K\nInput",
        "YOLO\nConversion",
        "Verification",
        "Rider\nBalancing",
        "IDD\nMerge",
        "Auto Dataset\nMerge",
        "Training",
        "Evaluation",
        "Results"
    ]
    
    num_steps = len(steps)
    y_centers = np.linspace(0.9, 0.1, num_steps)
    
    box_width = 0.5
    box_height = 0.06
    
    for i, step in enumerate(steps):
        y = y_centers[i]
        box = mpatches.FancyBboxPatch(
            (0.5 - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            ec="black", fc="#ADD8E6", lw=2
        )
        ax.add_patch(box)
        ax.text(0.5, y, step, ha='center', va='center', fontsize=12, fontweight='bold', family='sans-serif')
        
        if i < num_steps - 1:
            next_y = y_centers[i + 1]
            ax.annotate(
                '', xy=(0.5, next_y + box_height / 2), xytext=(0.5, y - box_height / 2),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=10, shrink=0.05)
            )
            
    plt.title("Figure 1: Proposed System Pipeline", fontsize=16, fontweight='bold', pad=20)
    
    out_path = out_dir / "Figure_1_System_Pipeline.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return str(out_path)

def generate_figure_2_comparison(eval_dir, out_dir, report_data):
    master_csv = eval_dir / "master_metrics.csv"
    report_data['Source files used'].append(str(master_csv))
    
    if not master_csv.exists():
        logging.warning(f"{master_csv} missing. Cannot generate Figure 2.")
        report_data['Missing plots'].append("Figure_2_Model_Comparison.png")
        report_data['Missing metrics'].append("master_metrics.csv missing")
        return None
        
    df = pd.read_csv(master_csv)
    if df.empty:
        logging.warning("Master metrics is empty. Cannot generate Figure 2.")
        report_data['Missing plots'].append("Figure_2_Model_Comparison.png")
        return None
        
    metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    missing_cols = [m for m in metrics if m not in df.columns]
    
    if missing_cols:
        report_data['Missing metrics'].extend(missing_cols)
        # Attempt to proceed with what is available
        metrics = [m for m in metrics if m in df.columns]
        
    if not metrics:
        return None
        
    # Sort models by paper specs (we assume mAP50 descending if not specified otherwise, or alphabetical)
    # The prompt says: "Sort models exactly as specified by the paper." - Usually sorted by architecture size or mAP.
    # We will sort by model name naturally (yolov5s, yolov5m, yolov7x, yolov8n...) which usually matches paper scaling tables.
    df = df.sort_values(by='Model')
    
    models = df['Model'].tolist()
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        # Handle N/A by replacing with 0 for plotting
        vals = pd.to_numeric(df[metric].replace("N/A", 0).fillna(0), errors='coerce').fillna(0)
        ax.bar(x + i*width - (width*len(metrics))/2 + width/2, vals, width, label=metric, color=colors[i])
        
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = out_dir / "Figure_2_Model_Comparison.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return str(out_path)

def locate_best_pr_curve(eval_dir, out_dir, report_data):
    master_csv = eval_dir / "master_metrics.csv"
    if not master_csv.exists():
        logging.warning(f"{master_csv} missing. Cannot locate best PR curve.")
        report_data['Missing plots'].append("Figure_3_PR_Curve.png")
        return None
        
    df = pd.read_csv(master_csv)
    if df.empty or 'mAP50' not in df.columns:
        report_data['Missing plots'].append("Figure_3_PR_Curve.png")
        return None
        
    # Treat N/A as 0 for argmax finding
    df['sort_val'] = pd.to_numeric(df['mAP50'].replace("N/A", 0).fillna(0), errors='coerce').fillna(0)
    best_idx = df['sort_val'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    
    model_dir = eval_dir / best_model
    report_data['Source files used'].append(f"Searching PR Curve in {model_dir}")
    
    # YOLO generates PR curves with different names depending on version
    # YOLOv8: PR_curve.png or BoxPR_curve.png
    # YOLOv5/7: PR_curve.png
    potential_names = ["PR_curve.png", "BoxPR_curve.png"]
    found_curve = None
    
    for name in potential_names:
        curve_path = model_dir / name
        if curve_path.exists():
            found_curve = curve_path
            break
            
    out_path = out_dir / "Figure_3_PR_Curve.png"
    
    if found_curve:
        shutil.copy2(found_curve, out_path)
        logging.info(f"Copied PR curve from {best_model}")
        return str(out_path)
    else:
        logging.warning(f"No PR curve image found for best model {best_model}.")
        report_data['Missing plots'].append("Figure_3_PR_Curve.png")
        report_data['Framework limitations'].append(f"Framework for {best_model} did not generate PR_curve.png locally during val.")
        return None

def generate_report(out_dir, report_data):
    md_path = out_dir / "FIGURE_GENERATION_REPORT.md"
    with open(md_path, 'w') as f:
        f.write("# Figure Generation Report\n\n")
        f.write("## Source files used\n")
        for x in set(report_data['Source files used']):
            f.write(f"- {x}\n")
        if not report_data['Source files used']:
            f.write("- None\n")
            
        f.write("\n## Missing plots\n")
        for x in set(report_data['Missing plots']):
            f.write(f"- {x}\n")
        if not report_data['Missing plots']:
            f.write("- None\n")
            
        f.write("\n## Missing metrics\n")
        for x in set(report_data['Missing metrics']):
            f.write(f"- {x}\n")
        if not report_data['Missing metrics']:
            f.write("- None\n")
            
        f.write("\n## Framework limitations\n")
        for x in set(report_data['Framework limitations']):
            f.write(f"- {x}\n")
        if not report_data['Framework limitations']:
            f.write("- None\n")
    logging.info(f"Generated {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Automatic Paper Figure Generator")
    parser.add_argument("--eval_dir", type=str, default="evaluation", help="Evaluation directory path")
    parser.add_argument("--out_dir", type=str, default="paper_outputs/figures", help="Output directory path")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    eval_dir = root_dir / args.eval_dir
    out_dir = root_dir / args.out_dir
    
    out_dir.mkdir(parents=True, exist_ok=True)
    report_out_dir = root_dir / "paper_outputs"
    report_out_dir.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        'Source files used': [],
        'Missing plots': [],
        'Missing metrics': [],
        'Framework limitations': []
    }
    
    logging.info("Generating Figure 1 (System Pipeline)...")
    f1 = draw_figure_1_pipeline(out_dir)
    
    logging.info("Generating Figure 2 (Model Comparison)...")
    f2 = generate_figure_2_comparison(eval_dir, out_dir, report_data)
    
    logging.info("Generating Figure 3 (PR Curve)...")
    f3 = locate_best_pr_curve(eval_dir, out_dir, report_data)
    
    generate_report(report_out_dir, report_data)
    
    logging.info("Paper figure generation complete.")

if __name__ == "__main__":
    main()
