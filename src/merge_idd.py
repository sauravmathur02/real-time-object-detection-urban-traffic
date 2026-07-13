import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDD_DIR = PROJECT_ROOT / "data1" / "IDDDetectionsYOLODataset"
DEFAULT_BDD_DIR = PROJECT_ROOT / "data1" / "bdd_balanced"


def merge_datasets(idd_dir, bdd_dir, splits=None):
    if splits is None:
        splits = ["train", "val"]

    class_mapping = {
        1: 9,   # autorickshaw -> auto
        2: 6,   # bicycle
        3: 4,   # bus
        4: 2,   # car
        5: 3,   # caravan -> truck
        6: 5,   # motorcycle
        7: 0,   # person -> pedestrian
        8: 1,   # rider
        9: 7,   # traffic light
        10: 8,  # traffic sign
        11: 3,  # trailer -> truck
        13: 3,  # truck
        14: 2,  # vehicle fallback -> car
    }

    total_images_copied = 0
    total_labels_written = 0

    for split in splits:
        idd_images_dir = idd_dir / split / "images"
        idd_labels_dir = idd_dir / split / "labels"

        bdd_images_dir = bdd_dir / "images" / split
        bdd_labels_dir = bdd_dir / "labels" / split

        bdd_images_dir.mkdir(parents=True, exist_ok=True)
        bdd_labels_dir.mkdir(parents=True, exist_ok=True)

        if not idd_labels_dir.exists():
            print(f"Skipping {split} as {idd_labels_dir} does not exist.")
            continue

        label_files = [file_path for file_path in idd_labels_dir.iterdir() if file_path.suffix == ".txt"]

        for label_file in tqdm(label_files, desc=f"Merging {split} split"):
            id_prefix = "idd_"
            new_label_file = f"{id_prefix}{label_file.name}"
            bdd_label_path = bdd_labels_dir / new_label_file

            with open(label_file, "r", encoding="utf-8") as file_obj:
                lines = file_obj.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id in class_mapping:
                    new_class_id = class_mapping[class_id]
                    new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                    new_lines.append(new_line)

            if new_lines:
                with open(bdd_label_path, "w", encoding="utf-8") as file_obj:
                    file_obj.writelines(new_lines)
                total_labels_written += len(new_lines)

                base_name = label_file.stem
                image_extensions = [".jpg", ".jpeg", ".png"]

                image_found = False
                for ext in image_extensions:
                    idd_image_path = idd_images_dir / f"{base_name}{ext}"
                    if idd_image_path.exists():
                        new_image_file = id_prefix + base_name + ext
                        bdd_image_path = bdd_images_dir / new_image_file
                        shutil.copy2(idd_image_path, bdd_image_path)
                        total_images_copied += 1
                        image_found = True
                        break

                if not image_found:
                    print(f"Warning: Could not find image for label {label_file.name}")

    print("\n==========================================")
    print("MERGE COMPLETE")
    print("==========================================")
    print(f"Total new images added to BDD: {total_images_copied}")
    print(f"Total objects remapped and added: {total_labels_written}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge IDD YOLO data into the BDD label space.")
    parser.add_argument("--idd-dir", type=str, default=str(DEFAULT_IDD_DIR), help="IDD YOLO dataset directory.")
    parser.add_argument("--bdd-dir", type=str, default=str(DEFAULT_BDD_DIR), help="Destination BDD-style dataset.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to merge.")
    args = parser.parse_args()

    merge_datasets(
        Path(args.idd_dir).expanduser().resolve(),
        Path(args.bdd_dir).expanduser().resolve(),
        args.splits,
    )
