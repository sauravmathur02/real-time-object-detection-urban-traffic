import json
import os
import argparse
import logging
import shutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def convert_bdd_to_yolo(json_file, images_src_dir, labels_dst_dir, images_dst_dir, mode='train'):
    logging.info(f"Loading JSON annotations from: {json_file}")
    if not os.path.exists(json_file):
        logging.error(f"JSON file not found: {json_file}")
        return 0, 0

    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(labels_dst_dir, exist_ok=True)
    os.makedirs(images_dst_dir, exist_ok=True)
    
    logging.info(f"Starting conversion for {mode} data. Total frames in JSON: {len(data)}")

    img_w, img_h = 1280, 720
    
    processed_count = 0
    skipped_count = 0

    for frame in tqdm(data, desc=f"Processing {mode}"):
        image_name = frame['name']
        src_image_path = os.path.join(images_src_dir, image_name)
        
        # Check if the source image exists
        if not os.path.exists(src_image_path):
            logging.warning(f"Image not found, skipping: {src_image_path}")
            skipped_count += 1
            continue

        # Check if frame has valid labels
        valid_labels = []
        for label in frame.get('labels', []):
            if 'box2d' in label and label.get('category') in BDD_CLASSES:
                valid_labels.append(label)
                
        # Skip if no valid annotations
        if not valid_labels:
            skipped_count += 1
            continue
            
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dst_dir, txt_name)
        dst_image_path = os.path.join(images_dst_dir, image_name)
        
        # Don't overwrite existing labels
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f_txt:
                for label in valid_labels:
                    category = label['category']
                    class_id = BDD_CLASSES.index(category)
                    box = label['box2d']
                    
                    w = box['x2'] - box['x1']
                    h = box['y2'] - box['y1']
                    cx = box['x1'] + (w / 2)
                    cy = box['y1'] + (h / 2)
                    
                    # YOLO: class_id center_x center_y width height (normalized)
                    f_txt.write(f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")
        
        # Copy image if it doesn't exist at destination
        if not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)
            
        processed_count += 1

    # Validate output
    actual_images = len([f for f in os.listdir(images_dst_dir) if f.endswith(('.jpg', '.png'))])
    actual_labels = len([f for f in os.listdir(labels_dst_dir) if f.endswith('.txt')])
    
    logging.info(f"[{mode.upper()}] Conversion Summary:")
    logging.info(f" - Successfully processed and matched: {processed_count}")
    logging.info(f" - Skipped (missing image or no annotations): {skipped_count}")
    logging.info(f" - Images in destination: {actual_images}")
    logging.info(f" - Labels in destination: {actual_labels}")
    
    if actual_images != actual_labels:
        logging.error(f"Mismatch detected! Images: {actual_images}, Labels: {actual_labels}")
    else:
        logging.info("Validation passed: 1:1 image-label matching confirmed.")
        
    return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K JSON labels to YOLO format with image coping")
    parser.add_argument("--json_train", type=str, required=True, help="Path to BDD100K train JSON labels")
    parser.add_argument("--json_val", type=str, required=True, help="Path to BDD100K val JSON labels")
    parser.add_argument("--images_train", type=str, required=True, help="Path to raw BDD100K train images")
    parser.add_argument("--images_val", type=str, required=True, help="Path to raw BDD100K val images")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for YOLO format (e.g., data1/bdd_yolo)")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    json_train = os.path.abspath(args.json_train)
    json_val = os.path.abspath(args.json_val)
    images_train_src = os.path.abspath(args.images_train)
    images_val_src = os.path.abspath(args.images_val)
    output_dir = os.path.abspath(args.output_dir)
    
    logging.info(f"Output directory established at: {output_dir}")
    
    # Build exact output target structures
    labels_train_dst = os.path.join(output_dir, "labels", "train")
    labels_val_dst = os.path.join(output_dir, "labels", "val")
    images_train_dst = os.path.join(output_dir, "images", "train")
    images_val_dst = os.path.join(output_dir, "images", "val")
    
    total_train, skipped_train = convert_bdd_to_yolo(
        json_train, images_train_src, labels_train_dst, images_train_dst, 'train'
    )
    
    total_val, skipped_val = convert_bdd_to_yolo(
        json_val, images_val_src, labels_val_dst, images_val_dst, 'val'
    )
    
    logging.info("="*40)
    logging.info("FINAL PROCESSING REPORT")
    logging.info("="*40)
    logging.info(f"Total Train Images: {total_train}")
    logging.info(f"Total Val Images: {total_val}")
    logging.info(f"Overall Skipped Images: {skipped_train + skipped_val}")
    logging.info("="*40)

if __name__ == "__main__":
    main()
