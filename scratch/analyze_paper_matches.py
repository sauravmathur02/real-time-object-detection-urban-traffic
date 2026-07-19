import json

with open("scratch/forensic_full.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

print(f"Loaded {len(runs)} runs.")

# Print timeline (sorted by last_mod)
sorted_runs = sorted(runs, key=lambda x: x['last_mod'])

print("\n================ CHRONOLOGICAL TIMELINE OF TRAINING SESSIONS ================\n")
for r in sorted_runs:
    p = r['metrics'].get('metrics/precision(B)', r['metrics'].get('Precision', '-'))
    rec = r['metrics'].get('metrics/recall(B)', r['metrics'].get('Recall', '-'))
    m50 = r['metrics'].get('metrics/mAP50(B)', r['metrics'].get('mAP50', '-'))
    m5095 = r['metrics'].get('metrics/mAP50-95(B)', r['metrics'].get('mAP50-95', '-'))
    
    p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
    rec_str = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
    m50_str = f"{m50:.4f}" if isinstance(m50, float) else str(m50)
    m5095_str = f"{m5095:.4f}" if isinstance(m5095, float) else str(m5095)

    print(f"[{r['last_mod']}] {r['rel_path']}")
    print(f"   Model: {r['model']} | ImgSz: {r['imgsz']} | Batch: {r['batch']} | Dataset: {r['dataset']}")
    print(f"   Configured: {r['epochs_cfg']} | Completed: {r['epochs_done']} | Status: {r['status']}")
    print(f"   best.pt: {r['best_pt']} ({r['best_mb']} MB) | last.pt: {r['last_pt']} ({r['last_mb']} MB)")
    print(f"   Metrics -> P: {p_str}, R: {rec_str}, mAP50: {m50_str}, mAP50-95: {m5095_str}")
    print("-" * 90)
