# Real-Time Object Detection in Urban Traffic Scenes Using YOLO-Based Architectures

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper](https://img.shields.io/badge/Paper-Published-brightgreen)
![GitHub stars](https://img.shields.io/github/stars/sauravmathur02/real-time-object-detection-urban-traffic?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/sauravmathur02/real-time-object-detection-urban-traffic)

This repository contains the official implementation of the real-time urban traffic object detection workflow. It is specifically optimized for developing economies, targeting vulnerable road users (riders) and regional vehicles (auto-rickshaws) under diverse lighting and traffic conditions.

---

## 🌟 Repository Highlights
- **Multi-stage training pipeline:** Robust transfer learning from COCO to custom balanced datasets.
- **BDD100K + IDD + Auto dataset fusion:** A rich, highly complex dataset merging strategy tailored for developing economies.
- **Rider class balancing:** Mitigating extreme data imbalances for vulnerable road users.
- **Hard-example mining:** Identifying and retraining on severe false negatives iteratively.
- **YOLOv5 vs YOLOv8 comparison:** Support for multi-architecture evaluation.
- **Real-time inference:** Fully optimized end-to-end OpenCV execution.
- **FastAPI dashboard:** A threaded Web UI for zero-latency camera streaming.
- **Benchmark framework:** Automated FPS and latency breakdown across preprocessing, inference, and NMS.
- **Qualitative comparison tool:** Side-by-side visual outputs of edge-cases for architectural review.

---

## 📄 Paper Overview
The repository perfectly maps to the theoretical claims made in the research paper. For a direct, section-by-section breakdown bridging the paper's claims to this codebase, see the [**Paper Implementation Map**](docs/PAPER_IMPLEMENTATION_MAP.md).

## 🧠 Methodology & Pipelines

### Dataset Pipeline
The project utilizes a complex dataset merging strategy, combining BDD100K, IDD, and Auto-rickshaw datasets, alongside hard-example mining and minority class oversampling. 
👉 **Detailed Documentation:** [**DATASET.md**](docs/DATASET.md)

### Training Pipeline
Training employs a strict 4-stage transfer learning methodology utilizing YOLOv8l, gradually exposing the model to harder datasets and balanced labels.
👉 **Detailed Documentation:** [**EXPERIMENTS.md**](docs/EXPERIMENTS.md)

### Evaluation & Results Pipeline
All models undergo rigorous empirical and qualitative evaluation, producing Precision-Recall curves, confusion matrices, and side-by-side inference comparisons. 
👉 **Detailed Documentation:** [**RESULTS.md**](docs/RESULTS.md)

### Reproducibility
We ensure that independent researchers can completely reproduce our findings, from hardware baselines to random seeds.
👉 **Detailed Documentation:** [**REPRODUCIBILITY.md**](docs/REPRODUCIBILITY.md)

---

## 🛠 Setup & Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Local Environment Constraints:**
Keep datasets, training runs, videos, and `.pt` weights local. They are intentionally excluded from git via `.gitignore`.

3. **Verify dataset manifests:**
The repository tracks the YAML schemas used during training without committing large data:
- `data1/bdd100k.yaml`
- `data1/bdd_balanced.yaml`

---

## 🚀 Execution Guide

### 1. Dataset Preparation
*(See [DATASET.md](docs/DATASET.md) for deeper methodology)*

Convert BDD100K into YOLO format:
```bash
python src/bdd_to_yolo_prod.py --json_train path/to/bdd100k_labels_images_train.json --json_val path/to/bdd100k_labels_images_val.json --images_train path/to/bdd100k/images/100k/train --images_val path/to/bdd100k/images/100k/val --output_dir data1/bdd_yolo
```

Oversample the rider class:
```bash
python src/oversample_rider.py --src_dir data1/bdd_yolo --dst_dir data1/bdd_balanced --duplication 2
```

Merge IDD into the BDD label space:
```bash
python src/merge_idd.py --idd-dir data1/IDDDetectionsYOLODataset --bdd-dir data1/bdd_balanced
```

Convert and merge auto-rickshaw data:
```bash
python src/convert_auto_to_yolo.py --images-dir data1/auto/auto --fallback-images-dir data1/auto --xml-dir data1/Annotations/Annotations --output-dir data1/auto_yolo
python src/merge_auto_yolo.py --src-dir data1/auto_yolo --dst-dir data1/bdd_balanced
```

Mine hard examples for Stage 3 tuning:
```bash
python src/collect_hard_examples.py --model runs/detect/train/weights/best.pt --val data1/bdd_yolo/images/val --output data1/hard_examples
```

### 2. Training
*(See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for full training history)*

Train on the base BDD dataset:
```bash
yolo detect train data=data1/bdd100k.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Train on the final balanced dataset with auto-rickshaw integration:
```bash
yolo detect train data=data1/bdd_balanced.yaml model=yolov8l.pt epochs=50 imgsz=640 batch=4 name=train_auto_v1
```

### 3. Inference
Run the main application for real-time webcam or video tracking:
```bash
python src/main.py --source 0 --model runs/detect/train/weights/best.pt
python src/main.py --source path/to/video.mp4 --model runs/detect/train/weights/best.pt --save
```

Quick webcam test:
```bash
python src/predict_webcam.py --source 0 --model runs/detect/train/weights/best.pt
```

### 4. Benchmarking & Qualitative Reporting
*(See [RESULTS.md](docs/RESULTS.md) for benchmark tracking)*

Generate performance latency reports:
```bash
python scripts/benchmark.py --models yolov5s=runs/yolov5s.pt yolov8l=runs/yolov8l.pt --source data1/bdd_yolo/images/val --output evaluation
```

Generate qualitative edge-case comparisons:
```bash
python scripts/generate_qualitative_figs.py --images sample_images --models yolov5s=runs/yolov5s.pt yolov8l=runs/yolov8l.pt --output results/qualitative
```

---

## 📂 Repository Structure

```text
object-Detection/
|-- data1/                     # Local dataset workspace (Ignored by Git)
|-- docs/                      # Extensive research and implementation documentation
|-- evaluation/                # Generated master results and benchmarks
|-- experiments/               # Legacy/Orchestrated training runs
|-- paper/                     # Original training notes and research writeups
|-- results/                   # Final rendered figures, demos, and qualitative renders
|-- runs/                      # Ultralytics native tracking directories
|-- scripts/                   # Benchmarking, profiling, and qualitative generation
|-- src/                       # Core codebase (Inference, Web Dashboard, Preprocessing)
|-- CITATION.cff               # Citation schema
|-- LICENSE                    # Legal licensing
|-- requirements.txt           # Python dependencies
`-- README.md                  # Main entry point
```

## 🗺️ Project Roadmap
- [x] Initial COCO Pretraining (Stage 1)
- [x] Base BDD100K Training (Stage 2)
- [x] Hard-Example Mining Integration (Stage 3)
- [x] Full Dataset Fusion & Balancing (Stage 4)
- [x] Real-time OpenCV Inference
- [x] Web Dashboard Implementation
- [x] Qualitative Benchmarking Suite
- [ ] TensorRT / ONNX Optimization
- [ ] Docker Containerization for Cloud Deployments
- [ ] NCNN Port for Edge Devices

---

## 🙏 Acknowledgements
- [Berkeley DeepDrive (BDD100K)](https://bdd-data.berkeley.edu/) for their incredible foundational dataset.
- [India Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) for their unstructured traffic scenarios.
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the state-of-the-art YOLO architectures.

---

## 📝 Citation
If you utilize this repository or findings in your research, please refer to the `CITATION.cff` file in the root directory.

## ⚖️ License
This project is licensed under the terms described in the `LICENSE` file.
