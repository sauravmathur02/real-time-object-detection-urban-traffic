# Call Graph

This maps the active execution paths for the inference and visualization pipelines.

## 1. CLI Inference Graph (`src/main.py`)

```mermaid
graph TD
    A[main.py CLI] -->|Init| B(detector.UrbanDetector)
    B --> C[ultralytics.YOLO]
    A -->|Read| D[cv2.VideoCapture]
    D --> E{While True}
    E -->|Frame| F[detector.detect]
    F --> C
    C -->|Output| G[results.boxes]
    G --> H[YOLO plotting / cv2 overlay]
    H -->|Annotated Frame| I[cv2.imshow]
    H -->|Write| J[cv2.VideoWriter]
```

## 2. Dashboard Web Interface Graph (`src/dashboard/app.py`)

```mermaid
graph TD
    A[FastAPI Server: app.py] -->|App Init| B(CameraManager)
    B -->|Thread Start| C[CameraManager._capture_loop]
    C -->|Read| D[cv2.VideoCapture]
    C -->|Frame| E[detector.UrbanDetector]
    E --> F[ultralytics.YOLO]
    F -->|Output| G[Calculate Detections / FPS]
    G -->|cv2.imencode| H[Shared Memory Buffer: _annotated_frame]
    
    I[HTTP GET /video_feed] -->|Stream| J[CameraManager.get_frame]
    J -->|Yield| H
    
    K[HTTP POST /api/config] -->|Update| L[CameraManager.set_conf_threshold]
    L -->|Thread Safe Lock| M[Conf Threshold Memory]
    M --> C
```

## 3. Training Execution Graph (`scripts/train_orchestrator.py`)

```mermaid
graph TD
    A[train_orchestrator.py] -->|Step 1| B[Run Validation Epoch]
    B -->|Subprocess| C[python frameworks/yolov5_src/train.py]
    C -->|Wait| B
    B -->|Step 2| D[Run Full Training Epochs]
    D -->|Subprocess| E[python frameworks/yolov5_src/train.py]
    E -->|Wait| D
    D -->|Step 3| F[Artifact Verification]
    F --> G[generate_args_yaml]
    F --> H[generate_reproducibility_metadata]
    H --> I[experiment.yaml & environment.json]
```
