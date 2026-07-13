# Model Flow

This documents the lineage, purpose, and final status of every neural network model tracked in the repository.

### YOLOv8 Architecture (Main Research Branch)

- **`object_stage1` (YOLOv8l)**
  - **Dataset:** `bdd100k.yaml` (Base data)
  - **Purpose:** Initial transfer learning from COCO weights to urban traffic context. 
  - **Status:** Obsolete baseline.

- **`object_stage2` (YOLOv8l)**
  - **Dataset:** `bdd100k.yaml` 
  - **Purpose:** Continued fine-tuning of Stage 1 weights.
  - **Status:** Obsolete baseline.

- **`object_stage3_strong2` (YOLOv8l)**
  - **Dataset:** `bdd_no_train.yaml` (Base data minus the `train` class)
  - **Purpose:** Training on Hard-Mined examples (False Negatives) using stronger mosaic augmentations to improve edge-case recall.
  - **Status:** Obsolete baseline.

- **`object_stage4_riderfix4` (YOLOv8l)**
  - **Dataset:** `bdd_balanced.yaml` (Merged with IDD, Auto, and Oversampled Riders)
  - **Purpose:** Final production model containing all dataset augmentations, mapped classes, and rider fixes.
  - **Status:** **FINAL (YOLOv8l benchmark)**.

- **`train_fast2` / `train_auto_v1` / `train3`**
  - **Purpose:** Hyperparameter sweep experiments using smaller image sizes (512, 640) or nano models (`yolov8n`) to test latency vs mAP tradeoffs.
  - **Status:** Experimental dead-ends.

### YOLOv5 Architecture (Comparative Branch)

- **`validation_yolov5s`**
  - **Dataset:** `bdd_balanced.yaml`
  - **Purpose:** Orchestrator dummy run to test environment constraints without crashing a 50-epoch run.
  - **Status:** Utility.

- **`train_yolov5s_v1`**
  - **Dataset:** `bdd_balanced.yaml`
  - **Purpose:** The primary YOLOv5 small architecture comparative run to benchmark against YOLOv8l's Stage 4. 
  - **Status:** **Active / Incomplete (Epoch 10)**.
