# Step-by-Step Guide: Preparing BDD100K for This Repository

This guide assumes you already downloaded the BDD100K images and JSON labels.

## 1. Organize the raw dataset

From the project root, keep the raw files in a local workspace such as:

```text
data1/raw_bdd/bdd100k/images/100k/train
data1/raw_bdd/bdd100k/images/100k/val
data1/raw_bdd/bdd100k/labels/bdd100k_labels_images_train.json
data1/raw_bdd/bdd100k/labels/bdd100k_labels_images_val.json
```

The exact extraction path can vary. The important part is knowing where the train/val image folders and JSON label files are located.

## 2. Convert BDD labels into YOLO format

Run the production conversion flow from the repository root:

```powershell
python src/bdd_to_yolo_prod.py `
  --json_train data1/raw_bdd/bdd100k/labels/bdd100k_labels_images_train.json `
  --json_val data1/raw_bdd/bdd100k/labels/bdd100k_labels_images_val.json `
  --images_train data1/raw_bdd/bdd100k/images/100k/train `
  --images_val data1/raw_bdd/bdd100k/images/100k/val `
  --output_dir data1/bdd_yolo
```

This creates the standard YOLO structure:

```text
data1/bdd_yolo/
|-- images/train
|-- images/val
|-- labels/train
`-- labels/val
```

## 3. Verify the converted dataset

Before training, validate image/label alignment:

```powershell
python src/verify_dataset.py --data_dir data1/bdd_yolo
```

You should end up with matching image and label counts and no empty `.txt` files.

## 4. Train the base detector

The repository already includes a tracked dataset manifest at `data1/bdd100k.yaml`.

```powershell
yolo detect train data=data1/bdd100k.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0
```

If you are training without a GPU, remove `device=0`.

## 5. Review outputs

Ultralytics will save results under `runs/detect/<experiment-name>/`.

Typical artifacts include:

- training curves
- confusion matrix
- precision/recall plots
- `weights/best.pt`
