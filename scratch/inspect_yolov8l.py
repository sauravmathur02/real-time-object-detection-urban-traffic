import json

with open('scratch/forensic_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total runs: {len(data)}")
print("="*80)

for r in data:
    rel = r['rel_path']
    rel_lower = rel.lower()
    model_str = str(r['model']).lower()
    
    # Filter YOLOv8l or high or stage runs
    if 'yolov8l' in rel_lower or 'yolov8l' in model_str or 'stage' in rel_lower or 'high' in rel_lower or 'train7' in rel_lower or 'fast' in rel_lower or 'auto' in rel_lower:
        print(f"Folder: {r['rel_path']}")
        print(f"  Model Input: {r['model']}")
        print(f"  Dataset: {r['dataset']}")
        print(f"  Configured Epochs: {r['epochs_configured']} | Completed Epochs: {r['epochs_completed']}")
        print(f"  Last Modified: {r['last_modified']}")
        print(f"  best.pt: {r['best_pt']} ({r['best_size_mb']} MB) | last.pt: {r['last_pt']} ({r['last_size_mb']} MB)")
        m = r['metrics']
        p = m.get('metrics/precision(B)', m.get('Precision', 'N/A'))
        rec = m.get('metrics/recall(B)', m.get('Recall', 'N/A'))
        map50 = m.get('metrics/mAP50(B)', m.get('mAP50', 'N/A'))
        map5095 = m.get('metrics/mAP50-95(B)', m.get('mAP50-95', 'N/A'))
        print(f"  Final Metrics -> Precision: {p}, Recall: {rec}, mAP50: {map50}, mAP50-95: {map5095}")
        print("-" * 80)
