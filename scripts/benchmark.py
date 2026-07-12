import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import platform
import cv2
import pandas as pd
from tqdm import tqdm
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics package not found. YOLOv8 models will not load.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BenchmarkWrapper:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.model = None
        self.is_v5 = False

        path_lower = path.lower()
        if "yolov5" in name.lower() or "v5" in path_lower:
            self.is_v5 = True

        self.load_time = 0.0
        self.model_size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0
        self.params = "N/A"
        self.gflops = "N/A"

        self._load_model()
        self._extract_model_info()

    def _load_model(self):
        start_time = time.perf_counter()
        if self.is_v5:
            framework_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov5" / "yolov5_src"
            if framework_path.exists():
                self.model = torch.hub.load(str(framework_path), 'custom', path=self.path, source='local')
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.path)
            self.model.conf = 0.3
        else:
            if YOLO is None:
                raise ImportError("ultralytics package is required for YOLOv8 models.")
            self.model = YOLO(self.path)
        self.load_time = time.perf_counter() - start_time

    def _extract_model_info(self):
        try:
            if self.is_v5:
                # For torch hub YOLOv5, self.model.model is the PyTorch module
                total_params = sum(p.numel() for p in self.model.model.parameters())
                self.params = f"{total_params:,}"
            else:
                # For YOLOv8
                total_params = sum(p.numel() for p in self.model.model.parameters())
                self.params = f"{total_params:,}"
                # YOLOv8 might store gflops inside the model object
                if hasattr(self.model.model, 'pt_path'):
                    # Difficult to extract dynamically without thop, we just leave as N/A unless we can parse info()
                    pass
        except Exception as e:
            logging.debug(f"Failed to extract params for {self.name}: {e}")

    def predict_and_profile(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Runs inference and extracts speed metrics. Returns times in ms.
        """
        metrics = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
        
        if self.is_v5:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame)
            # YOLOv5 results.t is a tuple of (preprocess, inference, NMS) in ms
            if hasattr(results, 't'):
                t = results.t
                metrics['preprocess'] = t[0]
                metrics['inference'] = t[1]
                metrics['postprocess'] = t[2]
            else:
                # Fallback manual timing
                metrics['inference'] = 0.0 # difficult to isolate natively
        else:
            results = self.model(frame, verbose=False, conf=0.3)
            speed = results[0].speed
            metrics['preprocess'] = speed.get('preprocess', 0.0)
            metrics['inference'] = speed.get('inference', 0.0)
            metrics['postprocess'] = speed.get('postprocess', 0.0)

        metrics['total'] = metrics['preprocess'] + metrics['inference'] + metrics['postprocess']
        return metrics


def parse_models(args_models: List[str]) -> List[Tuple[str, str]]:
    models = []
    for m in args_models:
        if "=" not in m:
            logging.error(f"Invalid model format '{m}'. Must be name=path")
            sys.exit(1)
        name, path = m.split("=", 1)
        models.append((name, path))
    return models


def generate_markdown_report(df: pd.DataFrame, output_path: Path):
    hardware_info = {
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "CPU": platform.processor(),
        "OpenCV Version": cv2.__version__
    }

    md_content = "# Benchmark Report\n\n"
    md_content += "## Hardware & Environment\n"
    for k, v in hardware_info.items():
        md_content += f"- **{k}:** {v}\n"

    md_content += "\n## Benchmark Methodology\n"
    md_content += "Models were loaded sequentially and evaluated on the identical set of test images. "
    md_content += "Latency metrics include Preprocessing (resizing/normalization), Inference (forward pass), "
    md_content += "and Postprocessing (NMS). FPS is derived from the average Total Latency.\n\n"

    md_content += "## Results\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n"

    with open(output_path, "w") as f:
        f.write(md_content)
    logging.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO models for FPS and latency.")
    parser.add_argument("--models", nargs="+", required=True, help="Models in format name=path")
    parser.add_argument("--source", type=str, required=True, help="Directory of test images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for reports")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in source_dir.rglob("*") if p.suffix.lower() in image_extensions]

    if not all_images:
        logging.error(f"No images found in {source_dir}")
        sys.exit(1)

    logging.info(f"Found {len(all_images)} images for benchmarking.")
    model_args = parse_models(args.models)

    results_data = []

    for name, path in model_args:
        logging.info(f"Loading model {name}...")
        try:
            wrapper = BenchmarkWrapper(name, path)
        except Exception as e:
            logging.error(f"Failed to load {name}: {e}")
            continue

        logging.info(f"Benchmarking {name}...")
        
        # Warmup
        warmup_img = cv2.imread(str(all_images[0]))
        if warmup_img is not None:
            for _ in range(3):
                wrapper.predict_and_profile(warmup_img)

        total_prep = 0.0
        total_inf = 0.0
        total_post = 0.0
        total_total = 0.0
        valid_images = 0

        for img_path in tqdm(all_images, desc=f"Evaluating {name}"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            metrics = wrapper.predict_and_profile(frame)
            total_prep += metrics['preprocess']
            total_inf += metrics['inference']
            total_post += metrics['postprocess']
            total_total += metrics['total']
            valid_images += 1

        if valid_images == 0:
            logging.warning(f"No valid frames evaluated for {name}.")
            continue

        avg_prep = total_prep / valid_images
        avg_inf = total_inf / valid_images
        avg_post = total_post / valid_images
        avg_total = total_total / valid_images
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        results_data.append({
            "Model": wrapper.name,
            "Images": valid_images,
            "FPS": round(fps, 2),
            "Inference(ms)": round(avg_inf, 2),
            "Preprocess(ms)": round(avg_prep, 2),
            "Postprocess(ms)": round(avg_post, 2),
            "Total(ms)": round(avg_total, 2),
            "LoadTime(s)": round(wrapper.load_time, 2),
            "ModelSize(MB)": round(wrapper.model_size_mb, 2),
            "Parameters": wrapper.params,
            "GFLOPs": wrapper.gflops
        })

    if not results_data:
        logging.error("No benchmark results generated.")
        sys.exit(1)

    df = pd.DataFrame(results_data)
    csv_path = output_dir / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"CSV saved to {csv_path}")

    md_path = output_dir / "benchmark.md"
    generate_markdown_report(df, md_path)

if __name__ == "__main__":
    main()
