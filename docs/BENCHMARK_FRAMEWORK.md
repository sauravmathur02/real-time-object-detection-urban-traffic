# Benchmarking Framework

The `scripts/benchmark.py` module is a standardized evaluation tool designed to accurately measure the computational cost, latency, and hardware constraints of different object detection models operating on urban traffic datasets.

## Features
- **Latency Breakdown:** Separates Preprocessing (resize/normalize), Inference (forward pass), and Postprocessing (NMS) times.
- **Real-World FPS:** Calculates exact frames-per-second based on end-to-end latency.
- **Hardware Agnostic:** Automatically profiles the environment (CPU, GPU, Python, PyTorch, CUDA versions) to append to the report, ensuring metric validity across different researcher machines.
- **Model Profiling:** Extracts model weight sizes, parameter counts, and load times.

## How to Run

To run the benchmark, provide an input folder containing test frames (e.g., extracted video frames or the BDD validation set), the list of models, and the output directory.

```bash
python scripts/benchmark.py \
    --models yolov5s=experiments/yolov5/train_yolov5s_v1/weights/best.pt yolov8l=runs/detect/object_stage4_riderfix4/weights/best.pt \
    --source data1/bdd_yolo/images/val \
    --output evaluation
```

## Generated Artifacts

Executing the script will produce two files inside the specified `--output` directory:

1. **`benchmark.csv`**: A machine-readable CSV containing raw numerical data for all profiled metrics.
2. **`benchmark.md`**: A human-readable markdown report that includes the hardware environment profile, methodology description, and a rendered markdown table comparing the models side-by-side.
