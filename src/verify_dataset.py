import argparse
import os
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data1" / "bdd_yolo"


def verify_split(images_dir, labels_dir, split_name):
    print(f"\n--- Verifying {split_name.upper()} split ---")

    if os.path.exists(images_dir):
        image_files = {f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))}
    else:
        print(f"Directory not found: {images_dir}")
        image_files = set()

    if os.path.exists(labels_dir):
        label_files = {f for f in os.listdir(labels_dir) if f.endswith(".txt")}
    else:
        print(f"Directory not found: {labels_dir}")
        label_files = set()

    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}

    print(f"Total {split_name} images found: {len(image_files)}")
    print(f"Total {split_name} labels found: {len(label_files)}")

    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames

    print(f"\nMissing Labels (Images without .txt): {len(missing_labels)}")
    if missing_labels:
        for name in list(missing_labels)[:5]:
            print(f"  - {name}.jpg")
        if len(missing_labels) > 5:
            print("  - ...")

    print(f"\nMissing Images (Labels without .jpg/.png): {len(missing_images)}")
    if missing_images:
        for name in list(missing_images)[:5]:
            print(f"  - {name}.txt")
        if len(missing_images) > 5:
            print("  - ...")

    empty_labels = []
    if os.path.exists(labels_dir):
        for label_file in tqdm(label_files, desc=f"Checking {split_name} for empty labels"):
            label_path = os.path.join(labels_dir, label_file)
            if os.path.getsize(label_path) == 0:
                empty_labels.append(label_file)

    print(f"\nEmpty Label Files: {len(empty_labels)}")
    if empty_labels:
        for name in empty_labels[:5]:
            print(f"  - {name}")
        if len(empty_labels) > 5:
            print("  - ...")

    if len(missing_labels) == 0 and len(missing_images) == 0 and len(empty_labels) == 0:
        print(f"\nOK: {split_name.upper()} split is perfectly aligned. 1:1 matching confirmed.")
    else:
        print(f"\nERROR: {split_name.upper()} split has integrity issues.")


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO dataset integrity.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to the YOLO dataset base directory.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"Verifying dataset at: {data_dir}")

    train_images = os.path.join(data_dir, "images", "train")
    train_labels = os.path.join(data_dir, "labels", "train")
    val_images = os.path.join(data_dir, "images", "val")
    val_labels = os.path.join(data_dir, "labels", "val")

    verify_split(train_images, train_labels, "train")
    verify_split(val_images, val_labels, "val")


if __name__ == "__main__":
    main()
