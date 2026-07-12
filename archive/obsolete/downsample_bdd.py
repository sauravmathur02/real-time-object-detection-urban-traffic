import os
import shutil
import argparse

def downsample_dataset(images_dir, labels_dir, new_images_dir, new_labels_dir):
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)

    copied = 0
    missing = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
            
        base = os.path.splitext(label_file)[0]
        image_file = base + ".jpg"

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(new_images_dir, image_file))
            shutil.copy(label_path, os.path.join(new_labels_dir, label_file))
            copied += 1
        else:
            missing += 1

    print(f"Copied pairs: {copied}")
    print(f"Missing images: {missing}")

def main():
    parser = argparse.ArgumentParser(description="Downsample or copy matching image-label pairs for BDD100K dataset")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to original images directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to original labels directory")
    parser.add_argument("--new_images_dir", type=str, required=True, help="Path to new downsampled images directory")
    parser.add_argument("--new_labels_dir", type=str, required=True, help="Path to new downsampled labels directory")

    args = parser.parse_args()

    downsample_dataset(args.images_dir, args.labels_dir, args.new_images_dir, args.new_labels_dir)

if __name__ == "__main__":
    main()