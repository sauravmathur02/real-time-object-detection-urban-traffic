# Results & Evaluation

This document outlines the final capabilities, benchmarks, and model evaluations conducted within this repository to substantiate the research claims.

## Best Experiment (Final Model)
- **Experiment Run:** `object_stage4_riderfix4`
- **Architecture:** YOLOv8l
- **Dataset:** `bdd_balanced` (Integrated BDD100K + IDD + Auto + Rider Oversampling)
- **Significance:** Represents the culmination of the 4-stage transfer learning pipeline. Achieves the highest robustness on complex urban edge-cases.

## Baseline Models
Initial training stages and smaller architectural variants (e.g., YOLOv5s, YOLOv8n) serve as the experimental baselines.
- **Stage 1 Baselines:** Pre-trained COCO transfers on base BDD100K.
- **YOLOv5s v1:** Lightweight iteration tested via the orchestrator for mobile-friendly profiling.

## Master Results Table
All empirical results (Precision, Recall, mAP50, mAP50-95) across all repository experiments are programmatically tracked.
- **Full Historical Logs:** See [master_results.md](../evaluation/master_results.md) and [master_results.csv](../evaluation/master_results.csv).

## Benchmark Summary
The models undergo rigorous profiling to validate real-time execution speeds on target hardware.
- **Inference Latency:** Measured end-to-end (Preprocessing → Forward Pass → NMS).
- **Benchmark Data:** Stored dynamically in `evaluation/benchmark.md` using `scripts/benchmark.py`. Validates >30 FPS capability on edge/workstation hardware.

## Qualitative Comparison Summary
Empirical metrics do not always reflect edge-case robustness (like heavy occlusion or night lighting). Qualitative evaluations provide visual proof of model superiority.
- **Visuals:** Generated via `scripts/generate_qualitative_figs.py` and saved to `results/qualitative/`.

## Generated Figures
The repository extracts critical YOLO validation artifacts from the final run for reviewer visibility:
- **[Precision-Recall Curve](../results/figures/BoxPR_curve.png)**
- **[Precision Curve](../results/figures/BoxP_curve.png)**
- **[Recall Curve](../results/figures/BoxR_curve.png)**
- **[F1 Curve](../results/figures/BoxF1_curve.png)**
- **[Confusion Matrix](../results/figures/confusion_matrix.png)**
- **[Results Summary Plot](../results/figures/results.png)**
