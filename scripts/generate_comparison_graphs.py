import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Setup plot styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def main():
    root = Path(r'c:\Repo\object-Detection')
    eval_dir = root / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Completed models data
    data = [
        {"name": "YOLOv5s", "params": 7.05, "size": 13.77, "map50": 0.5244, "map95": 0.3094, "precision": 0.7065, "recall": 0.4749, "gflops": 16.5, "complete": True},
        {"name": "YOLOv5m", "params": 20.91, "size": 40.29, "map50": 0.4705, "map95": 0.3044, "precision": 0.6331, "recall": 0.4424, "gflops": 48.0, "complete": True},
        {"name": "YOLOv7x (Incomplete)", "params": 70.87, "size": 541.61, "map50": 0.3932, "map95": 0.2348, "precision": 0.5578, "recall": 0.3896, "gflops": 188.0, "complete": False},
        {"name": "YOLOv8n", "params": 3.01, "size": 5.95, "map50": 0.3927, "map95": 0.2555, "precision": 0.5811, "recall": 0.3605, "gflops": 8.7, "complete": True},
        {"name": "YOLOv8s", "params": 11.14, "size": 21.47, "map50": 0.4599, "map95": 0.3010, "precision": 0.6285, "recall": 0.4178, "gflops": 28.6, "complete": True},
        {"name": "YOLOv8m", "params": 25.86, "size": 49.62, "map50": 0.4895, "map95": 0.3277, "precision": 0.6452, "recall": 0.4504, "gflops": 78.9, "complete": True},
        {"name": "YOLOv8l (Final)", "params": 43.64, "size": 83.61, "map50": 0.4617, "map95": 0.2572, "precision": 0.5998, "recall": 0.4463, "gflops": 165.2, "complete": True},
        {"name": "YOLOv8l (Fast)", "params": 43.64, "size": 83.57, "map50": 0.5524, "map95": 0.3480, "precision": 0.7147, "recall": 0.5065, "gflops": 165.2, "complete": True},
        {"name": "YOLOv8l (Base)", "params": 43.64, "size": 83.59, "map50": 0.4828, "map95": 0.2952, "precision": 0.5554, "recall": 0.4742, "gflops": 165.2, "complete": True}
    ]
    
    df = pd.DataFrame(data)
    
    # 1. Plot mAP@0.5 vs. Parameters
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        color = 'red' if not row['complete'] else ('blue' if 'v8' in row['name'].lower() else 'green')
        marker = 'x' if not row['complete'] else 'o'
        plt.scatter(row['params'], row['map50'], s=150, color=color, marker=marker, zorder=3)
        plt.text(row['params'] + 1.2, row['map50'] - 0.005, row['name'], fontsize=11, zorder=4)
        
    plt.title("Model Accuracy (mAP@0.5) vs. Parameter Count", pad=15)
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("mAP@0.5")
    plt.xlim(0, 80)
    plt.ylim(0.35, 0.60)
    plt.tight_layout()
    map_params_path = eval_dir / 'map_vs_params.png'
    plt.savefig(map_params_path, dpi=300)
    plt.close()
    print(f"Generated {map_params_path}")
    
    # 2. Plot mAP@0.5 vs. Model Size (excluding YOLOv7x for scaling or keeping it with a broken axis/inset)
    plt.figure(figsize=(10, 6))
    # Let's plot with YOLOv7x included but with text, using log scale or just normal
    for i, row in df.iterrows():
        color = 'red' if not row['complete'] else ('blue' if 'v8' in row['name'].lower() else 'green')
        marker = 'x' if not row['complete'] else 'o'
        plt.scatter(row['size'], row['map50'], s=150, color=color, marker=marker, zorder=3)
        # Shift text to prevent overlap
        y_offset = 0.005 if 'Fast' in row['name'] else -0.005
        plt.text(row['size'] + 5 if row['size'] < 100 else row['size'] - 110, row['map50'] + y_offset, row['name'], fontsize=11, zorder=4)
        
    plt.title("Model Accuracy (mAP@0.5) vs. Model Size (MB)", pad=15)
    plt.xlabel("Model Size (MB)")
    plt.ylabel("mAP@0.5")
    plt.xlim(-10, 600)
    plt.ylim(0.35, 0.60)
    plt.tight_layout()
    map_size_path = eval_dir / 'map_vs_size.png'
    plt.savefig(map_size_path, dpi=300)
    plt.close()
    print(f"Generated {map_size_path}")
    
    # 3. Precision-Recall Trade-off Chart (Scatter plot of Precision vs. Recall)
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        color = 'red' if not row['complete'] else ('blue' if 'v8' in row['name'].lower() else 'green')
        marker = 'x' if not row['complete'] else 'o'
        plt.scatter(row['recall'], row['precision'], s=150, color=color, marker=marker, zorder=3)
        plt.text(row['recall'] + 0.005, row['precision'] - 0.003, row['name'], fontsize=11, zorder=4)
        
    plt.title("Precision vs. Recall Comparison", pad=15)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.30, 0.55)
    plt.ylim(0.50, 0.75)
    plt.tight_layout()
    pr_tradeoff_path = eval_dir / 'precision_recall_comparison.png'
    plt.savefig(pr_tradeoff_path, dpi=300)
    plt.close()
    print(f"Generated {pr_tradeoff_path}")
    
if __name__ == '__main__':
    main()
