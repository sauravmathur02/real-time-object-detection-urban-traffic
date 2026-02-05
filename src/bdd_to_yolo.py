import json
import os
import argparse
from tqdm import tqdm
import shutil

# BDD100K Class Mapping to YOLOv8 (COCO) or Custom
# BDD100K classes: "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
# We will map them to a custom ID mapping for training, or map to COCO if desired.
# Here we define a standard BDD mapping for a custom trained model.

BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def convert_bdd_to_yolo(json_file, output_dir, mode='train'):
    """
    Converts BDD100K json labels to YOLO format .txt files.
    """
    
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Make sure output directory exists
    labels_dir = os.path.join(output_dir, 'labels', mode)
    images_dir = os.path.join(output_dir, 'images', mode)
    
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Converting {len(data)} labels to {labels_dir}...")

    for frame in tqdm(data):
        image_name = frame['name']
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, txt_name)
        
        # BDD100K resolution is usually 1280x720
        # Check if intrinsics provided or assume 1280x720 (standard for BDD)
        # Assuming 1280x720 for normalization if not specified
        img_w = 1280
        img_h = 720
        
        with open(label_path, 'w') as f_txt:
            for label in frame.get('labels', []):
                if 'box2d' not in label:
                    continue
                
                category = label['category']
                if category not in BDD_CLASSES:
                    continue
                
                class_id = BDD_CLASSES.index(category)
                box = label['box2d']
                
                # Convert to center_x, center_y, w, h (normalized)
                x1 = box['x1']
                y1 = box['y1']
                x2 = box['x2']
                y2 = box['y2']
                
                w = x2 - x1
                h = y2 - y1
                cx = x1 + (w / 2)
                cy = y1 + (h / 2)
                
                # Normalize
                ncx = cx / img_w
                ncy = cy / img_h
                nw = w / img_w
                nh = h / img_h
                
                f_txt.write(f"{class_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Conversion for {mode} complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLO format")
    parser.add_argument("--json_train", type=str, help="Path to BDD100K train JSON labels")
    parser.add_argument("--json_val", type=str, help="Path to BDD100K val JSON labels")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for YOLO formatted data")
    
    args = parser.parse_args()
    
    if args.json_train:
        convert_bdd_to_yolo(args.json_train, args.output_dir, 'train')
    
    if args.json_val:
        convert_bdd_to_yolo(args.json_val, args.output_dir, 'val')

if __name__ == "__main__":
    main()
