import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = PROJECT_ROOT / "data1" / "bdd_yolo"
DEFAULT_DST_DIR = PROJECT_ROOT / "data1" / "bdd_balanced"


def process_split(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir, is_train=True, rider_class_id=1, duplication_factor=2):
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    if not os.path.exists(src_labels_dir):
        print(f"Source labels directory not found: {src_labels_dir}")
        return 0, 0, 0

    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith(".txt")]

    total_original = len(label_files)
    rider_images_found = 0
    total_new_size = 0

    for label_file in tqdm(label_files, desc=f"Processing {'train' if is_train else 'val'} split"):
        src_label_path = os.path.join(src_labels_dir, label_file)
        base_name = os.path.splitext(label_file)[0]

        image_file = base_name + ".jpg"
        src_image_path = os.path.join(src_images_dir, image_file)

        if not os.path.exists(src_image_path):
            image_file = base_name + ".png"
            src_image_path = os.path.join(src_images_dir, image_file)

        if not os.path.exists(src_image_path):
            print(f"Warning: Image for label {label_file} not found. Skipping.")
            continue

        has_rider = False
        with open(src_label_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id == rider_class_id:
                    has_rider = True
                    break

        if has_rider and is_train:
            rider_images_found += 1

        dst_image_path = os.path.join(dst_images_dir, image_file)
        dst_label_path = os.path.join(dst_labels_dir, label_file)

        shutil.copy2(src_image_path, dst_image_path)
        shutil.copy2(src_label_path, dst_label_path)
        total_new_size += 1

        if has_rider and is_train:
            for index in range(1, duplication_factor + 1):
                aug_img_name = f"{base_name}_rider_dup{index}{os.path.splitext(image_file)[1]}"
                aug_lbl_name = f"{base_name}_rider_dup{index}.txt"

                aug_img_path = os.path.join(dst_images_dir, aug_img_name)
                aug_lbl_path = os.path.join(dst_labels_dir, aug_lbl_name)

                shutil.copy2(src_image_path, aug_img_path)
                shutil.copy2(src_label_path, aug_lbl_path)
                total_new_size += 1

    return total_original, rider_images_found, total_new_size


def main():
    parser = argparse.ArgumentParser(description="Oversample 'rider' class images by duplicating them.")
    parser.add_argument("--src_dir", type=str, default=str(DEFAULT_SRC_DIR), help="Source YOLO dataset base directory.")
    parser.add_argument("--dst_dir", type=str, default=str(DEFAULT_DST_DIR), help="Destination balanced dataset directory.")
    parser.add_argument(
        "--duplication",
        type=int,
        default=2,
        help="Number of extra copies to make for a rider image.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    print(f"Balancing dataset from: {src_dir}")
    print(f"Outputting to: {dst_dir}")

    src_images_train = os.path.join(src_dir, "images", "train")
    src_labels_train = os.path.join(src_dir, "labels", "train")
    src_images_val = os.path.join(src_dir, "images", "val")
    src_labels_val = os.path.join(src_dir, "labels", "val")
    dst_images_train = os.path.join(dst_dir, "images", "train")
    dst_labels_train = os.path.join(dst_dir, "labels", "train")
    dst_images_val = os.path.join(dst_dir, "images", "val")
    dst_labels_val = os.path.join(dst_dir, "labels", "val")

    print("\n--- Processing TRAIN split ---")
    orig_train, riders_train, new_train = process_split(
        src_images_train,
        src_labels_train,
        dst_images_train,
        dst_labels_train,
        is_train=True,
        duplication_factor=args.duplication,
    )

    print("\n--- Processing VAL split ---")
    orig_val, _, new_val = process_split(
        src_images_val,
        src_labels_val,
        dst_images_val,
        dst_labels_val,
        is_train=False,
    )

    print("\n==========================================")
    print("OVERSAMPLING REPORT")
    print("==========================================")
    print(f"Original Train size: {orig_train} images")
    print(f"Original Val size:   {orig_val} images")
    print(f"Rider images found in Train: {riders_train}")
    print(f"-> Each rider image duplicated {args.duplication} times.")
    print(f"New Train size: {new_train} images")
    print(f"New Val size:   {new_val} images")
    print(f"Total New Dataset size: {new_train + new_val} images")
    print("==========================================")
    print(f"Balanced dataset created at: {dst_dir}")


if __name__ == "__main__":
    main()
