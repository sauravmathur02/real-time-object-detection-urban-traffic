import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics package not found. YOLOv8 models will not load.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelWrapper:
    """
    Wraps YOLOv5 and YOLOv8 models to provide a unified inference interface.
    """
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.model = None
        self.is_v5 = False

        path_lower = path.lower()
        if "yolov5" in name.lower() or "v5" in path_lower:
            self.is_v5 = True

        logging.info(f"Loading {self.name} (v5={self.is_v5}) from {self.path}")

        if self.is_v5:
            # Attempt to use local frameworks/yolov5_src if present
            framework_path = Path(__file__).resolve().parents[1] / "frameworks" / "yolov5" / "yolov5_src"
            if framework_path.exists():
                self.model = torch.hub.load(str(framework_path), 'custom', path=self.path, source='local')
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.path)
            
            # Configure confidence
            self.model.conf = 0.3
        else:
            if YOLO is None:
                raise ImportError("ultralytics package is required for YOLOv8 models.")
            self.model = YOLO(self.path)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs inference on a BGR image and returns the annotated BGR image.
        """
        if self.is_v5:
            # YOLOv5 expects RGB for correct rendering, but OpenCV reads as BGR.
            # torch.hub models handle BGR/RGB conversion natively if fed properly, 
            # but standard is to pass RGB to v5.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame)
            # render() returns list of numpy arrays in RGB
            rendered_rgb = results.render()[0]
            # Convert back to BGR for OpenCV
            return cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        else:
            # YOLOv8 processes BGR natively
            results = self.model(frame, verbose=False, conf=0.3)
            return results[0].plot()


def add_title_bar(img: np.ndarray, title: str) -> np.ndarray:
    """
    Adds a black title bar with white text to the top of an image.
    """
    bar_height = 40
    bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)
    # Center text roughly
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2
    cv2.putText(bar, title, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return cv2.vconcat([bar, img])


def build_comparison_image(orig_img: np.ndarray, predictions: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """
    Builds a composite image containing the original image on top,
    and all model predictions side-by-side on the bottom.
    """
    base_h, base_w = orig_img.shape[:2]

    # Process bottom row
    bottom_panels = []
    for model_name, pred_img in predictions:
        panel = add_title_bar(pred_img, f"{model_name} Prediction")
        bottom_panels.append(panel)
        
    if not bottom_panels:
        return orig_img

    bottom_row = cv2.hconcat(bottom_panels)
    target_w = bottom_row.shape[1]

    # Process top row
    top_panel = add_title_bar(orig_img, "Original Image")
    
    pad_left = (target_w - top_panel.shape[1]) // 2
    pad_right = target_w - top_panel.shape[1] - pad_left

    if pad_left >= 0 and pad_right >= 0:
        top_row = cv2.copyMakeBorder(top_panel, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        top_row = top_panel

    final_img = cv2.vconcat([top_row, bottom_row])
    return final_img


def parse_models(args_models: List[str]) -> List[Tuple[str, str]]:
    """
    Parses 'name=path' syntax into a list of tuples.
    """
    models = []
    for m in args_models:
        if "=" not in m:
            logging.error(f"Invalid model format '{m}'. Must be name=path")
            sys.exit(1)
        name, path = m.split("=", 1)
        models.append((name, path))
    return models


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative side-by-side comparisons of object detection models.")
    parser.add_argument("--images", type=str, required=True, help="Input directory containing test images.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to compare in format name=path (e.g. yolov5s=runs/best.pt yolov8l=runs/best.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for generated comparisons.")
    
    args = parser.parse_args()

    input_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()
    
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    model_args = parse_models(args.models)
    loaded_models = []
    for name, path in model_args:
        if not Path(path).exists():
            logging.error(f"Model path does not exist: {path}")
            sys.exit(1)
        try:
            loaded_models.append(ModelWrapper(name, path))
        except Exception as e:
            logging.error(f"Failed to load model {name} from {path}: {e}")
            sys.exit(1)

    # Find all images recursively
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_extensions]

    if not all_images:
        logging.warning(f"No images found in {input_dir}")
        sys.exit(0)

    logging.info(f"Found {len(all_images)} images. Starting generation...")

    for img_path in tqdm(all_images, desc="Generating Comparisons"):
        # Calculate relative path to maintain folder structure (e.g. day/, night/)
        rel_path = img_path.relative_to(input_dir)
        dest_path = output_dir / rel_path
        dest_path = dest_path.with_name(f"{dest_path.stem}_comparison{dest_path.suffix}")

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(img_path))
        if frame is None:
            logging.warning(f"Failed to read image {img_path}. Skipping.")
            continue

        predictions = []
        for wrapper in loaded_models:
            try:
                pred_frame = wrapper.predict(frame.copy())
                predictions.append((wrapper.name, pred_frame))
            except Exception as e:
                logging.error(f"Inference failed for {wrapper.name} on {img_path}: {e}")
                # Append blank frame on failure
                predictions.append((wrapper.name, np.zeros_like(frame)))

        composite = build_comparison_image(frame, predictions)
        cv2.imwrite(str(dest_path), composite)

    logging.info(f"Qualitative comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()
