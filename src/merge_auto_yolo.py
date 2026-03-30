import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_BASE = PROJECT_ROOT / "data1" / "auto_yolo"
DEFAULT_DST_BASE = PROJECT_ROOT / "data1" / "bdd_balanced"


def merge_datasets(src_base, dst_base, splits):
    transferred = 0

    for split in splits:
        src_images_dir = src_base / "images" / split
        src_images = sorted(
            list(src_images_dir.glob("*.jpg"))
            + list(src_images_dir.glob("*.jpeg"))
            + list(src_images_dir.glob("*.png"))
        )

        dst_img_dir = dst_base / "images" / split
        dst_lbl_dir = dst_base / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(src_images, desc=f"Merging {split}"):
            basename = img_path.name
            name_no_ext = img_path.stem
            src_lbl_path = src_base / "labels" / split / f"{name_no_ext}.txt"
            dst_img_path = dst_img_dir / basename
            dst_lbl_path = dst_lbl_dir / f"{name_no_ext}.txt"

            if src_lbl_path.exists():
                shutil.copy2(img_path, dst_img_path)
                shutil.copy2(src_lbl_path, dst_lbl_path)
                transferred += 1

    print(f"Merge Complete! {transferred} files successfully integrated into the balanced dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge converted auto YOLO samples into a target YOLO dataset.")
    parser.add_argument("--src-dir", type=str, default=str(DEFAULT_SRC_BASE), help="Source auto YOLO dataset.")
    parser.add_argument("--dst-dir", type=str, default=str(DEFAULT_DST_BASE), help="Destination YOLO dataset.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to merge.")
    args = parser.parse_args()

    merge_datasets(
        Path(args.src_dir).expanduser().resolve(),
        Path(args.dst_dir).expanduser().resolve(),
        args.splits,
    )
