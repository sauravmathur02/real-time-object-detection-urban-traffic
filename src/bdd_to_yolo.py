print("BDD to YOLO script started...")
import json
import os
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def convert_bdd_to_yolo(json_file, labels_dir, mode='train'):
    logging.info(f"Loading {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(labels_dir, exist_ok=True)
    logging.info(f"Converting {len(data)} labels to {labels_dir}")

    img_w, img_h = 1280, 720

    for frame in tqdm(data):
        image_name = frame['name']
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, txt_name)
        
        with open(label_path, 'w') as f_txt:
            for label in frame.get('labels', []):
                if 'box2d' not in label:
                    continue
                
                category = label['category']
                if category not in BDD_CLASSES:
                    continue
                
                class_id = BDD_CLASSES.index(category)
                box = label['box2d']
                
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']
                cx = box['x1'] + (w / 2)
                cy = box['y1'] + (h / 2)
                
                f_txt.write(f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")

    logging.info(f"Finished conversion for {mode}.")

def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K JSON labels to YOLO format")
    parser.add_argument("--json_train", type=str, help="Path to BDD100K train JSON labels")
    parser.add_argument("--json_val", type=str, help="Path to BDD100K val JSON labels")
    parser.add_argument("--output_dir", type=str, help="Base output directory for YOLO formatted labels")
    parser.add_argument("--label_dir", type=str, help="Specific directory to output labels")
    
    args = parser.parse_args()
    
    if args.json_train:
        target_dir = args.label_dir or os.path.join(args.output_dir, 'labels', 'train')
        convert_bdd_to_yolo(args.json_train, target_dir, 'train')
    
    if args.json_val:
        target_dir = args.label_dir or os.path.join(args.output_dir, 'labels', 'val')
        convert_bdd_to_yolo(args.json_val, target_dir, 'val')

if __name__ == "__main__":
    main()
