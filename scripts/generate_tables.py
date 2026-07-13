import json
import csv
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def format_val(v):
    if pd.isna(v) or v is None or str(v).lower() in ["nan", "none", "n/a"]:
        return "N/A"
    try:
        f = float(v)
        return f"{f:.3f}"
    except (ValueError, TypeError):
        return str(v)

def generate_table_iv(eval_dir, out_dir, report_data):
    master_csv = eval_dir / "master_metrics.csv"
    report_data['Input files used'].append(str(master_csv))
    
    if not master_csv.exists():
        logging.warning(f"{master_csv} missing. Generating empty Table IV.")
        df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'FPS', 'Inference Time'])
        report_data['Missing metrics'].append("Entire master_metrics.csv is missing")
    else:
        df_raw = pd.read_csv(master_csv)
        
        # Mapping to required paper column names
        records = []
        for _, row in df_raw.iterrows():
            model_name = row.get('Model', 'Unknown')
            p = row.get('Precision', "N/A")
            r = row.get('Recall', "N/A")
            map50 = row.get('mAP50', "N/A")
            map95 = row.get('mAP50-95', "N/A")
            
            inf_time = row.get('Inference time (ms/img)', "N/A")
            fps = "N/A"
            try:
                if inf_time != "N/A" and float(inf_time) > 0:
                    fps = 1000.0 / float(inf_time)
            except (ValueError, TypeError):
                pass
                
            records.append({
                'Model': model_name,
                'Precision': format_val(p),
                'Recall': format_val(r),
                'mAP@0.5': format_val(map50),
                'mAP@0.5:0.95': format_val(map95),
                'FPS': format_val(fps),
                'Inference Time': format_val(inf_time)
            })
            
        df = pd.DataFrame(records)
        
        # Attempt to sort by mAP@0.5 descending if data exists
        if 'mAP@0.5' in df.columns:
            # Temporarily convert to float for sorting, treating N/A as -1
            df['sort_key'] = df['mAP@0.5'].apply(lambda x: -1 if x == "N/A" else float(x))
            df = df.sort_values(by='sort_key', ascending=False).drop(columns=['sort_key'])

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "Table_IV_Model_Comparison.csv"
    md_path = out_dir / "Table_IV_Model_Comparison.md"
    
    df.to_csv(csv_path, index=False)
    
    with open(md_path, 'w') as f:
        f.write("# Table IV: Model Comparison\n\n")
        if df.empty:
            f.write("No data available.\n")
        else:
            cols = df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---" for _ in cols]) + "|\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
                
    logging.info(f"Generated {csv_path} and {md_path}")

def generate_table_v(eval_dir, out_dir, report_data):
    # Iterate through individual model directories looking for per-class AP in metrics.json
    all_class_records = {}
    
    # Check all subdirectories in evaluation
    if eval_dir.exists():
        for model_dir in eval_dir.iterdir():
            if model_dir.is_dir():
                metrics_json = model_dir / "metrics.json"
                if metrics_json.exists():
                    report_data['Input files used'].append(str(metrics_json))
                    with open(metrics_json, 'r') as f:
                        try:
                            data = json.load(f)
                            # Look for class dict natively
                            class_metrics = data.get('per_class', {})
                            if not class_metrics:
                                report_data['Framework limitations'].append(
                                    f"No per-class dict found in {metrics_json.name} for {model_dir.name}"
                                )
                                report_data['Missing classes'].append(f"All missing for {model_dir.name}")
                            else:
                                for c_name, c_data in class_metrics.items():
                                    if c_name not in all_class_records:
                                        all_class_records[c_name] = {
                                            'Class': c_name,
                                            'AP': format_val(c_data.get('AP', 'N/A')),
                                            'Support': format_val(c_data.get('Support', 'N/A')),
                                            'Mean Confidence': format_val(c_data.get('Mean Confidence', 'N/A'))
                                        }
                        except Exception as e:
                            logging.error(f"Error parsing {metrics_json}: {e}")
                            
    records = list(all_class_records.values())
    df = pd.DataFrame(records, columns=['Class', 'AP', 'Support', 'Mean Confidence'])
    
    if df.empty:
        logging.warning("No per-class metrics found. Generating empty Table V.")
        report_data['Missing metrics'].append("Per-class metrics completely unavailable.")
        df = pd.DataFrame([{"Class": "N/A", "AP": "N/A", "Support": "N/A", "Mean Confidence": "N/A"}])
        
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "Table_V_Per_Class_AP.csv"
    md_path = out_dir / "Table_V_Per_Class_AP.md"
    
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write("# Table V: Per-Class AP\n\n")
        cols = df.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---" for _ in cols]) + "|\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

    logging.info(f"Generated {csv_path} and {md_path}")

def generate_report(out_dir, report_data):
    md_path = out_dir / "TABLE_GENERATION_REPORT.md"
    with open(md_path, 'w') as f:
        f.write("# Table Generation Report\n\n")
        f.write("## Input files used\n")
        for x in set(report_data['Input files used']):
            f.write(f"- {x}\n")
        if not report_data['Input files used']:
            f.write("- None\n")
            
        f.write("\n## Missing metrics\n")
        for x in set(report_data['Missing metrics']):
            f.write(f"- {x}\n")
        if not report_data['Missing metrics']:
            f.write("- None\n")
            
        f.write("\n## Missing classes\n")
        for x in set(report_data['Missing classes']):
            f.write(f"- {x}\n")
        if not report_data['Missing classes']:
            f.write("- None\n")
            
        f.write("\n## Framework limitations\n")
        for x in set(report_data['Framework limitations']):
            f.write(f"- {x}\n")
        if not report_data['Framework limitations']:
            f.write("- None\n")
    logging.info(f"Generated {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Automatic Paper Table Generator")
    parser.add_argument("--eval_dir", type=str, default="evaluation", help="Evaluation directory path")
    parser.add_argument("--out_dir", type=str, default="paper_outputs", help="Output directory path")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    eval_dir = root_dir / args.eval_dir
    out_dir = root_dir / args.out_dir
    
    report_data = {
        'Input files used': [],
        'Missing metrics': [],
        'Missing classes': [],
        'Framework limitations': []
    }
    
    generate_table_iv(eval_dir, out_dir, report_data)
    generate_table_v(eval_dir, out_dir, report_data)
    generate_report(out_dir, report_data)
    
    logging.info("Paper table generation complete.")

if __name__ == "__main__":
    main()
