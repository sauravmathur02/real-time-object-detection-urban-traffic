# Step-by-Step Guide: Implementing BDD100K with YOLOv8

This guide assumes you have downloaded the **BDD100K Images** and **Labels** (JSON).

## 1. Organize Your Files
First, let's create a clean structure in your `data` folder to avoid confusion.

1.  Navigate to `d:\MyProjects\Object_Detection\data`.
2.  Create a folder named `raw_bdd`.
3.  Inside `raw_bdd`, extract your downloaded files so standard paths look like this:
    *   **Images**: `data/raw_bdd/bdd100k/images/100k/train/...` and `.../val/...`
    *   **Labels**: `data/raw_bdd/bdd100k/labels/bdd100k_labels_images_train.json` (and val)

> [!NOTE]
> The exact path might vary depending on how you unzip it. The key is to locate the `.json` label files and the folder containing `.jpg` images.

## 2. Convert Labels to YOLO Format
YOLOv8 cannot read the BDD JSON files directly; it needs a `.txt` file for every image. We have prepared a script for this.

**Run the following command in your terminal:**

```powershell
# Navigate to project root
cd d:\MyProjects\Object_Detection

# Run conversion (Adjust paths if your extracted files are different)
python src/bdd_to_yolo.py --json_train "data/raw_bdd/bdd100k/labels/bdd100k_labels_images_train.json" --json_val "data/raw_bdd/bdd100k/labels/bdd100k_labels_images_val.json" --output_dir "data/bdd_yolo"
```

*   **What this does**: It reads the JSON, expects images to be in the standard BDD structure relative to the json, and creates a new folder `data/bdd_yolo` with organized `images` and `labels`.
*   **Important**: Use the `--output_dir` as `data/bdd_yolo` so our config file works.

## 3. Verify the Dataset
Before training, check if the conversion worked.
1.  Go to `data/bdd_yolo/labels/train`.
2.  Open a `.txt` file. You should see lines like: `2 0.543 0.231 0.111 0.054`.
3.  Check `data/bdd100k.yaml` is pointing to the right place (I have already set it to `../data/bdd_yolo`).

## 4. Run Training
Now you can start training the model on your new dataset.

```powershell
yolo detect train data=data/bdd100k.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0
```
*   `device=0`: Uses your GPU. If you don't have one, remove this (but it will be slow).
*   `epochs=100`: You can reduce this to 5 or 10 just to test it first.

## 5. View Results
Results will be saved in `runs/detect/train`. You can view the metrics (confusion matrix, precision-recall curves) there.
