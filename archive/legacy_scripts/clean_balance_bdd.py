import os
import shutil
import argparse

def balance_dataset(images_dir, labels_dir, new_images_dir, new_labels_dir):
    if os.path.exists(new_images_dir):
        shutil.rmtree(new_images_dir)
    if os.path.exists(new_labels_dir):
        shutil.rmtree(new_labels_dir)

    os.makedirs(new_images_dir)
    os.makedirs(new_labels_dir)

    print("Old balanced folders cleared and new ones created.")

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
            shutil.copy2(image_path, os.path.join(new_images_dir, image_file))
            shutil.copy2(label_path, os.path.join(new_labels_dir, label_file))
            copied += 1
        else:
            missing += 1

    print("\n====== DONE ======")
    print(f"Copied pairs : {copied}")
    print(f"Missing images: {missing}")

    print("\nVerification:")
    print(f"Images in balanced: {len(os.listdir(new_images_dir))}")
    print(f"Labels in balanced: {len(os.listdir(new_labels_dir))}")

def main():
    parser = argparse.ArgumentParser(description="Clean and balance BDD100K dataset by strict 1:1 image-label matching")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to original images directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to original labels directory")
    parser.add_argument("--new_images_dir", type=str, required=True, help="Path to new balanced images directory")
    parser.add_argument("--new_labels_dir", type=str, required=True, help="Path to new balanced labels directory")

    args = parser.parse_args()

    balance_dataset(args.images_dir, args.labels_dir, args.new_images_dir, args.new_labels_dir)

if __name__ == "__main__":
    main()