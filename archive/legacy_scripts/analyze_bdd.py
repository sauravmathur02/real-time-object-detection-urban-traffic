import os
import argparse
from collections import defaultdict

def analyze_dataset(dataset_path):
    splits = ["train", "val", "test"]

    for split in splits:
        label_path = os.path.join(dataset_path, split)

        if not os.path.exists(label_path):
            continue

        print(f"\n===== {split.upper()} =====")

        class_instance_count = defaultdict(int)
        class_image_count = defaultdict(int)
        total_images = 0
        background_images = 0

        for file in os.listdir(label_path):
            if not file.endswith(".txt"):
                continue

            total_images += 1
            file_path = os.path.join(label_path, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                background_images += 1
                continue

            classes_in_image = set()

            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                class_instance_count[cls] += 1
                classes_in_image.add(cls)

            for cls in classes_in_image:
                class_image_count[cls] += 1

        print(f"Total images: {total_images}")
        print(f"Background images: {background_images}")
        print("\nClass-wise stats:")

        for cls in sorted(class_instance_count.keys()):
            print(f"Class {cls} -> "
                  f"Images: {class_image_count[cls]} | "
                  f"Instances: {class_instance_count[cls]}")

def main():
    parser = argparse.ArgumentParser(description="Analyze BDD100K dataset labels")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the BDD100K labels directory")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist.")
        return

    analyze_dataset(args.dataset_path)

if __name__ == "__main__":
    main()