# Duplicate Implementations

The repository contains several scripts that exist simultaneously in the project root and inside the `src/` directory.

### 1. `oversample_rider.py`
- **Root Version:** Hardcoded to target `data1/bdd_no_train/images/train`. Cloned functionality. No CLI arguments.
- **Src Version (`src/oversample_rider.py`):** Utilizes `argparse`. Dynamically handles `train` and `val` splits. Tracks metrics (Original Size, New Size) and gracefully handles missing files.
- **Comparison:** Completely different execution architectures.
- **Recommendation:** Keep the `src/` version. Delete the root version.

### 2. `bdd_to_yolo.py` vs `bdd_to_yolo_prod.py` (Both in `src/`)
- **`bdd_to_yolo.py`:** Earlier iteration. Handles basic JSON decoding.
- **`bdd_to_yolo_prod.py`:** Hardened version with multi-processing, error catching for corrupt polygons, and extensive logging.
- **Comparison:** Iterative evolution.
- **Recommendation:** Keep `bdd_to_yolo_prod.py`. Delete `bdd_to_yolo.py`.

### 3. Other Root Duplicates
- `analyze_bdd.py`, `clean_balance_bdd.py`, `count_rider_images.py`, `downsample_bdd.py`, `fix_labels.py`, `remove_train_class.py`
- These are all experimental scripts dumped into the root. Their core features (e.g., removing the train class, mapping IDs) are definitively implemented inside `src/merge_idd.py` and `src/merge_auto_yolo.py`.
- **Recommendation:** Delete all of them from the root.
