import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data1" / "auto" / "auto"
DEFAULT_FALLBACK_IMAGES_DIR = PROJECT_ROOT / "data1" / "auto"
DEFAULT_XML_DIR = PROJECT_ROOT / "data1" / "Annotations" / "Annotations"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data1" / "auto_yolo"


def convert_voc_to_yolo(xml_path, img_width, img_height, class_id=9):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        if name not in ["autorickshaw", "auto"]:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        w = xmax - xmin
        h = ymax - ymin
        x_center = xmin + w / 2.0
        y_center = ymin + h / 2.0

        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_labels


def find_images(images_dir, fallback_images_dir):
    patterns = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for pattern in patterns:
        images.extend(sorted(images_dir.glob(pattern)))
    if images:
        return images, images_dir

    if fallback_images_dir != images_dir:
        fallback_images = []
        for pattern in patterns:
            fallback_images.extend(sorted(fallback_images_dir.glob(pattern)))
        if fallback_images:
            return fallback_images, fallback_images_dir

    return [], images_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert VOC/XML auto-rickshaw annotations into YOLO format."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing auto images.",
    )
    parser.add_argument(
        "--fallback-images-dir",
        type=str,
        default=str(DEFAULT_FALLBACK_IMAGES_DIR),
        help="Fallback image directory if --images-dir has no matches.",
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default=str(DEFAULT_XML_DIR),
        help="Directory containing VOC/XML annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=9,
        help="YOLO class id to assign to auto-rickshaw labels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    fallback_images_dir = Path(args.fallback_images_dir).expanduser().resolve()
    xml_dir = Path(args.xml_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("Finding images...")
    images, resolved_images_dir = find_images(images_dir, fallback_images_dir)
    if not images:
        print(f"No images found in {images_dir} or {fallback_images_dir}")
        return

    print(f"Found {len(images)} images in: {resolved_images_dir}")

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(images)
    num_val = int(len(images) * args.val_split)
    val_images = set(images[:num_val])

    processed = 0
    missing_xml = 0

    for img_path in tqdm(images, desc="Converting Auto Dataset"):
        basename = img_path.name
        name_no_ext = img_path.stem

        xml_path = xml_dir / f"{name_no_ext}.xml"
        if not xml_path.exists():
            missing_xml += 1
            print(f"Warning: Missing XML for {basename}")
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            img_width = float(size.find("width").text)
            img_height = float(size.find("height").text)
        except Exception as exc:
            print(f"Error reading size from {xml_path}: {exc}")
            continue

        yolo_lines = convert_voc_to_yolo(xml_path, img_width, img_height, class_id=args.class_id)
        if not yolo_lines:
            continue

        split = "val" if img_path in val_images else "train"
        dest_img = output_dir / "images" / split / basename
        dest_label = output_dir / "labels" / split / f"{name_no_ext}.txt"

        shutil.copy2(img_path, dest_img)
        with open(dest_label, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(yolo_lines))

        processed += 1

    print("\n--- Summary ---")
    print(f"Total processed successfully: {processed}")
    print(f"Missing XMLs: {missing_xml}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
