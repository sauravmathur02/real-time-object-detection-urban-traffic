# Dataset Methodology

This repository combines multiple distinct datasets into a highly-specialized, unified YOLO label space for detecting vehicles and vulnerable road users in developing economies.

## Source Datasets
1. **BDD100K:** The foundational dataset containing highly diverse urban scenes.
2. **IDD (India Driving Dataset):** Provides dense traffic scenarios specific to Indian roads, unstructured driving behaviors, and distinct vehicle types.
3. **Datacluster Labs Auto-Rickshaw:** A highly targeted subset addressing the `auto-rickshaw` vehicle, a 3-wheeled transport central to Indian urban traffic.

## Class Mapping Table
The original class names are re-mapped into a custom 10-class YOLO space to normalize differences between datasets.

| Original BDD100K | Original IDD | Final Unified Class |
|:---|:---|:---|
| pedestrian | pedestrian | **pedestrian** |
| rider | rider / motorcyclist | **rider** |
| car | car | **car** |
| truck | truck | **truck** |
| bus | bus | **bus** |
| train | N/A | **auto** *(Replaced in balanced)* |
| motorcycle | motorcycle | **motorcycle** |
| bicycle | bicycle | **bicycle** |
| traffic light | traffic light | **traffic light** |
| traffic sign | traffic sign | **traffic sign** |

*(Note: `train` is removed and replaced by `auto` in the `bdd_balanced.yaml` space)*

## Folder Structure
All tracked dataset configurations reside in the root `data1/` folder. The raw images are intentionally omitted via `.gitignore`.
```text
data1/
|-- bdd100k.yaml          # Base BDD configurations
|-- bdd_balanced.yaml     # Final multi-source configuration
|-- bdd_yolo/             # Root target for BDD conversion
|-- auto_yolo/            # Converted auto-rickshaw annotations
`-- hard_examples/        # Mined frames with high FP/FN rates
```

## Data Preprocessing Pipeline
- **Conversion:** `src/bdd_to_yolo_prod.py` parses complex BDD JSON files into standardized normalized XYWH YOLO text files.
- **Rider Balancing:** Vulnerable road users (e.g., scooters, bikes) are severely underrepresented. `src/oversample_rider.py` identifies frames containing riders and artificially duplicates them in the training set to prevent model bias toward dominant classes like `car`.
- **Hard-Example Mining:** `src/collect_hard_examples.py` runs validation inference to extract images where the model heavily misclassifies objects. These are aggregated to form a dedicated Stage 3 fine-tuning dataset.
