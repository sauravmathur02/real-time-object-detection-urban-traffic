import argparse
import logging
import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_best_model(eval_dir):
    master_csv = eval_dir / "master_metrics.csv"
    if not master_csv.exists():
        logging.error(f"Cannot find {master_csv} to determine best model.")
        return None
        
    df = pd.read_csv(master_csv)
    if df.empty or 'mAP50' not in df.columns:
        logging.error("master_metrics.csv is empty or missing mAP50 column.")
        return None
        
    df['sort_val'] = pd.to_numeric(df['mAP50'].replace("N/A", 0).fillna(0), errors='coerce').fillna(0)
    best_idx = df['sort_val'].idxmax()
    return df.loc[best_idx, 'Model']

def parse_metrics_for_model(eval_dir, model_name):
    model_dir = eval_dir / model_name
    json_path = model_dir / "metrics.json"
    
    if not json_path.exists():
        logging.error(f"Metrics JSON missing for {model_name} at {json_path}")
        return None
        
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            return data.get('per_class', {})
        except Exception as e:
            logging.error(f"Error reading {json_path}: {e}")
            return None

def compute_derived_metrics(per_class_data):
    records = []
    
    # Calculate total support for relative contribution
    total_support = sum(c_data.get('Support', 0) for c_data in per_class_data.values() if isinstance(c_data.get('Support'), (int, float)))
    
    for cls_name, c_data in per_class_data.items():
        p = c_data.get('Precision', 0.0)
        r = c_data.get('Recall', 0.0)
        ap = c_data.get('AP', 0.0)
        sup = c_data.get('Support', 0)
        conf = c_data.get('Mean Confidence', 0.0)
        
        # F1 Score
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        # Relative Contribution
        rel_cont = (sup / total_support) * 100 if total_support > 0 else 0.0
        
        records.append({
            'Class': cls_name,
            'AP': float(ap),
            'Precision': float(p),
            'Recall': float(r),
            'F1 Score': float(f1),
            'Support': int(sup),
            'Mean Confidence': float(conf),
            'Relative Contribution (%)': float(rel_cont)
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        # Ranking by AP descending
        df['Ranking'] = df['AP'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values(by='Ranking')
        
    return df

def identify_summary_insights(df):
    if df.empty:
        return {}
        
    # Safely get extremes
    try:
        top_3 = df.nlargest(3, 'AP')['Class'].tolist()
        bottom_3 = df.nsmallest(3, 'AP')['Class'].tolist()
        
        highest_conf = df.loc[df['Mean Confidence'].idxmax(), 'Class'] if df['Mean Confidence'].sum() > 0 else "N/A"
        lowest_conf = df.loc[df['Mean Confidence'].idxmin(), 'Class'] if df['Mean Confidence'].sum() > 0 else "N/A"
        
        most_difficult = df.loc[df['F1 Score'].idxmin(), 'Class']
        
        most_freq = df.loc[df['Support'].idxmax(), 'Class'] if df['Support'].sum() > 0 else "N/A"
        least_freq = df.loc[df['Support'].idxmin(), 'Class'] if df['Support'].sum() > 0 else "N/A"
        
        return {
            "Top 3 classes (AP)": top_3,
            "Bottom 3 classes (AP)": bottom_3,
            "Highest confidence class": highest_conf,
            "Lowest confidence class": lowest_conf,
            "Most difficult class (F1)": most_difficult,
            "Most frequent class": most_freq,
            "Least frequent class": least_freq
        }
    except Exception as e:
        logging.warning(f"Could not compute summary insights: {e}")
        return {}

def generate_reports(df, insights, model_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        logging.warning("DataFrame is empty. Writing empty reports.")
        with open(out_dir / "per_class_metrics.csv", "w") as f:
            f.write("No metrics available.\n")
        return

    # CSV
    df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    
    # JSON Summary
    summary = {
        "Model Analyzed": model_name,
        "Insights": insights
    }
    with open(out_dir / "per_class_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    # Metrics MD
    with open(out_dir / "per_class_metrics.md", 'w') as f:
        f.write(f"# Per-Class Metrics for {model_name}\n\n")
        cols = df.columns.tolist()
        
        # Rounding for presentation in MD
        display_df = df.copy()
        for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
            display_df[col] = display_df[col].map('{:.3f}'.format)
            
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---" for _ in cols]) + "|\n")
        for _, row in display_df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
            
    # Analysis MD
    with open(out_dir / "PER_CLASS_ANALYSIS.md", 'w') as f:
        f.write(f"# Per-Class Analysis ({model_name})\n\n")
        f.write("This document summarizes the empirical performance of the model on a per-class basis, driven entirely by validation metrics.\n\n")
        
        f.write("## Best Classes\n")
        if insights.get("Top 3 classes (AP)"):
            f.write("The classes that yielded the highest Average Precision (AP) are:\n")
            for c in insights["Top 3 classes (AP)"]:
                val = df.loc[df['Class'] == c, 'AP'].values[0]
                f.write(f"- **{c}** (AP: {val:.3f})\n")
        f.write("\n")
        
        f.write("## Weakest Classes\n")
        if insights.get("Bottom 3 classes (AP)"):
            f.write("The classes exhibiting the lowest performance metrics are:\n")
            for c in insights["Bottom 3 classes (AP)"]:
                val = df.loc[df['Class'] == c, 'AP'].values[0]
                f.write(f"- **{c}** (AP: {val:.3f})\n")
        f.write("\n")
        
        f.write("## Dataset Imbalance Observations\n")
        f.write(f"The most frequent class is **{insights.get('Most frequent class', 'N/A')}**, while the least frequent is **{insights.get('Least frequent class', 'N/A')}**. ")
        f.write(f"Often, the most difficult class (lowest F1: **{insights.get('Most difficult class (F1)', 'N/A')}**) corresponds with low confidence (**{insights.get('Lowest confidence class', 'N/A')}**) and potentially low support.\n")
        
    logging.info(f"Generated textual reports in {out_dir}")

def generate_plots(df, out_dir):
    if df.empty:
        return
        
    # Helper to plot horizontal bar charts
    def plot_hbar(metric, filename, color):
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Sort specifically for this plot
        temp_df = df.sort_values(by=metric, ascending=True)
        
        y_pos = np.arange(len(temp_df))
        plt.barh(y_pos, temp_df[metric], color=color, edgecolor='black')
        plt.yticks(y_pos, temp_df['Class'], fontsize=10)
        plt.xlabel(metric, fontsize=12, fontweight='bold')
        plt.title(f'Class {metric}', fontsize=16, fontweight='bold', pad=15)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        out_path = out_dir / filename
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        
    # 1. Ranking (Based on AP, but just visualizing AP descending is ranking)
    plot_hbar('AP', 'class_ranking.png', '#1f77b4')
    
    # 2. Precision
    plot_hbar('Precision', 'class_precision.png', '#2ca02c')
    
    # 3. Recall
    plot_hbar('Recall', 'class_recall.png', '#ff7f0e')
    
    # 4. AP
    plot_hbar('AP', 'class_AP.png', '#9467bd')
    
    logging.info(f"Generated visualization plots in {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Per-Class Performance Analysis Generator")
    parser.add_argument("--eval_dir", type=str, default="evaluation", help="Evaluation directory path")
    parser.add_argument("--out_dir", type=str, default="paper_outputs/per_class", help="Output directory path")
    parser.add_argument("--model", type=str, help="Specifically force a model rather than finding best")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    eval_dir = root_dir / args.eval_dir
    out_dir = root_dir / args.out_dir
    
    target_model = args.model
    if not target_model:
        logging.info("Model not explicitly specified. Automatically finding best model...")
        target_model = get_best_model(eval_dir)
        
    if not target_model:
        logging.error("Failed to determine a model to analyze.")
        return
        
    logging.info(f"Targeting model: {target_model}")
    
    per_class_data = parse_metrics_for_model(eval_dir, target_model)
    
    if not per_class_data:
        logging.warning("No per-class metrics found. Exiting gracefully.")
        return
        
    df = compute_derived_metrics(per_class_data)
    insights = identify_summary_insights(df)
    
    generate_reports(df, insights, target_model, out_dir)
    generate_plots(df, out_dir)
    
    logging.info("Per-class performance analysis complete.")

if __name__ == "__main__":
    main()
