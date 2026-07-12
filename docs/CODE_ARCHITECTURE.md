# Code Architecture

The repository architecture follows a multi-faceted approach, composed of data preprocessing, YOLO-based training (v5 and v8), and a FastAPI dashboard for real-time inference.

## 1. Core Implementation (`src/`)

- **`src/main.py`**
  - **Purpose:** Standalone script for real-time video/webcam inference.
  - **Responsibility:** Captures video stream, calls YOLO wrapper, calculates FPS, and draws bounding boxes using OpenCV.
  - **Imports:** `cv2`, `argparse`, `time`, `logging`, `detector.UrbanDetector`, `os`
  - **Exports:** `main()`
  - **Caller:** User via CLI.
  - **Calls:** `detector.py`
  - **Status:** Experimental / Production testing.

- **`src/detector.py`**
  - **Purpose:** Wrapper class around Ultralytics YOLO inference.
  - **Responsibility:** Loads weights, handles class filtering (10 urban classes), and returns annotated frames.
  - **Imports:** `logging`, `ultralytics.YOLO`
  - **Exports:** `UrbanDetector`
  - **Caller:** `main.py`, `dashboard/camera_manager.py`
  - **Calls:** `ultralytics.YOLO`
  - **Status:** Production.

- **`src/dashboard/app.py`**
  - **Purpose:** FastAPI Backend for web visualization.
  - **Responsibility:** Serves HTML/JS, streams MJPEG frames, exposes REST endpoints for config changes (confidence threshold).
  - **Imports:** `fastapi`, `uvicorn`, `os`, `shutil`, `CameraManager`
  - **Exports:** API Routes
  - **Caller:** HTTP Clients / Browsers.
  - **Calls:** `camera_manager.py`
  - **Status:** Production.

- **`src/dashboard/camera_manager.py`**
  - **Purpose:** Thread-safe background frame processing.
  - **Responsibility:** Continuously reads video stream, performs YOLO inference asynchronously, updates current FPS/detections, buffers JPEG frames for streaming.
  - **Imports:** `cv2`, `threading`, `time`, `logging`, `UrbanDetector`
  - **Exports:** `CameraManager` class
  - **Caller:** `app.py`
  - **Calls:** `detector.py`
  - **Status:** Production.

## 2. Dataset Processing (`src/`)

- **`src/bdd_to_yolo_prod.py`** & **`src/bdd_to_yolo.py`**
  - **Purpose:** JSON to YOLO format converter.
  - **Status:** Production / Data prep.

- **`src/convert_auto_to_yolo.py`**
  - **Purpose:** XML to YOLO normalized bounding box converter for the auto-rickshaw dataset.
  - **Status:** Production / Data prep.

- **`src/merge_idd.py`** & **`src/merge_auto_yolo.py`**
  - **Purpose:** Re-maps class indices and merges IDD/Auto datasets into the BDD label space.
  - **Status:** Production / Data prep.

- **`src/oversample_rider.py`**
  - **Purpose:** Combats class imbalance by finding training images with riders and duplicating them by a factor of 2.
  - **Status:** Production / Data prep.

- **`src/collect_hard_examples.py`**
  - **Purpose:** Implements "Hard Example Mining" by inferring on validation sets and dumping images with false negatives or low confidence `< 0.3` to a `hard_examples/` directory.
  - **Status:** Experimental / Active Research.

- **`src/verify_dataset.py`**
  - **Purpose:** Sanity check for matching image/label pairs and bounding box coordinate validation.
  - **Status:** Utility.

## 3. Training & Orchestration (`scripts/`)

- **`scripts/train_orchestrator.py`**
  - **Purpose:** Automates YOLOv5 training workflow.
  - **Responsibility:** Runs a 1-epoch validation, then kicks off a 50-epoch training run. Automatically logs hardware environment and `experiment.yaml`.
  - **Imports:** `subprocess`, `sys`, `os`, `shutil`, `datetime`, `platform`, `json`, `yaml`
  - **Calls:** Subprocesses `train.py` from `frameworks/yolov5/yolov5_src`.
  - **Status:** Experimental / Active Research.

- **`scripts/train_yolov5s.py`**
  - **Purpose:** Standalone script wrapping YOLOv5 training CLI arguments.
  - **Status:** Obsolete / Replaced by orchestrator.

## 4. Root Directory Duplicates (Unused/Obsolete)
- `oversample_rider.py`, `analyze_bdd.py`, `clean_balance_bdd.py`, `count_rider_images.py`, `downsample_bdd.py`, `fix_labels.py`, `remove_train_class.py`.
- **Status:** All obsolete. Safe to delete. They duplicate functionality natively built into the `src/` modules.
