# Data Flow

The following graph visualizes the physical data transformation pipeline, showing how raw assets map through python modules to become model weights.

```mermaid
graph TD
    A[Raw BDD100K JSON + JPG] -->|src/bdd_to_yolo_prod.py| B[bdd_yolo]
    
    C[Raw IDD XML + JPG] -->|src/merge_idd.py| D[bdd_balanced]
    
    E[Raw Auto XML + JPG] -->|src/convert_auto_to_yolo.py| F[auto_yolo]
    F -->|src/merge_auto_yolo.py| D
    
    B -->|Base Copy| D
    
    D -->|src/oversample_rider.py| G[Final bdd_balanced + Duplicated Riders]
    
    G -->|src/verify_dataset.py| H[Validated Dataset YAML]
    
    H -->|Ultralytics Dataloader| I[YOLOv8 Training / YOLOv5 Orchestrator]
    
    I -->|Model Weights best.pt| J[src/detector.py]
    
    J -->|src/main.py / FastAPI| K[Real-Time Inference]
```

## Description of Flow
1. **Conversion:** The base datasets exist in diverse formats (JSON, XML). Independent scripts normalize bounding boxes to `[class x_center y_center width height]`.
2. **Merging & Class Mapping:** The IDD and Auto datasets have custom classes. The scripts rewrite the YOLO `.txt` label files to map to the 10-class BDD format.
3. **Balancing:** Before training, `oversample_rider.py` mathematically duplicates the specific `rider` images.
4. **Validation:** `verify_dataset.py` ensures 1:1 image-to-label parity.
5. **Consumption:** The `YOLO` API consumes the YAML referencing these balanced folders.
6. **Inference:** Final weights (`best.pt`) are loaded by the FastAPI `camera_manager.py` or the `main.py` CLI script for video-feed evaluation.
