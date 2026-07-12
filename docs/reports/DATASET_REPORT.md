# Dataset Report

## Datasets Incorporated
1. **BDD100K** (Berkeley DeepDrive) Subset
2. **IDD** (India Driving Dataset) Subset
3. **Auto Dataset** (Datacluster Labs Auto-Rickshaw)

## Images Inventory
- **Train Images:**
  - Base BDD subset (`bdd_yolo`): 1,153 images
  - Merged Balanced subset (`bdd_balanced`): **34,536** images
- **Validation Images:**
  - Base BDD subset (`bdd_yolo`): 9,999 images
  - Merged Balanced subset (`bdd_balanced`): **14,150** images
- **Test Images:** `NOT FOUND` (The local workspace does not contain isolated test splits).

## Labels & Classes
**Number of Classes:** 10

**Base BDD100K Classes:**
`pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`

**Balanced Merged Classes:**
`pedestrian`, `rider`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `auto`

## Mapping Strategies
- **IDD Integration:** The 13 original IDD class IDs were remapped to BDD equivalents. 
  - Index `1` (autorickshaw) -> `9` (auto)
  - `caravan` and `trailer` -> `truck`
  - `person` -> `pedestrian`
- **Auto-rickshaw Integration:** Pascal VOC XML bounding boxes were parsed and converted to normalized YOLO coordinates with the assigned class ID `9` (`auto`). The base BDD `train` class was removed from the schema to make room for `auto`.

## Preprocessing
- **BDD100K:** Converted complex JSON polygons and metadata into standard YOLO txt format using `src/bdd_to_yolo_prod.py`.
- **Auto Dataset:** XML annotation extraction to normalized YOLO format via `src/convert_auto_to_yolo.py`.
- **Folder Validation:** `src/verify_dataset.py` ensures 1:1 matching of images and labels across the processed splits.

## Augmentation
- Standard YOLO augmentations utilized during training (mosaic, mixup, flipping).
- "Stronger augmentations" applied during Stage 3 (Hard-Example Mining training) to enforce generalized robustness on difficult edge cases.

## Balancing
- **Rider Oversampling:** The `rider` class (Class ID 1) is massively underrepresented. Script `src/oversample_rider.py` was deployed to automatically duplicate training images containing the rider class by a factor of 2, directly mitigating class imbalance.
