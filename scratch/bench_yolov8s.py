
import os
import sys
import time
import glob
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_dir = Path(r"C:\Repo\object-Detection")
framework = "v8"
model_path = r"C:\Repo\object-Detection\runs\detect\train_yolov8s2\weights\best.pt"
model_name = "YOLOv8s"

val_image_dir = root_dir / "data1" / "bdd_yolo" / "images" / "val"
img_paths = sorted(glob.glob(str(val_image_dir / "*.jpg")))[:200]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_warmup = 10

if framework == "v8":
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.model.to(device).eval().float()
    
    for p in img_paths[:num_warmup]:
        _ = model.predict(p, imgsz=640, device=device, verbose=False, save=False)
        
    prep_times, inf_times, post_times = [], [], []
    for p in tqdm(img_paths, desc=f"Benchmarking {model_name}"):
        res = model.predict(p, imgsz=640, device=device, verbose=False, save=False)
        speed = res[0].speed
        prep_times.append(speed['preprocess'])
        inf_times.append(speed['inference'])
        post_times.append(speed['postprocess'])

elif framework == "v5":
    yolov5_src = root_dir / "frameworks" / "yolov5" / "yolov5_src"
    model = torch.hub.load(str(yolov5_src), 'custom', path=model_path, source='local', device=device)
    model.eval().float()
    
    for p in img_paths[:num_warmup]:
        _ = model(p, size=640)
        
    prep_times, inf_times, post_times = [], [], []
    for p in tqdm(img_paths, desc=f"Benchmarking {model_name}"):
        res = model(p, size=640)
        t = res.t
        prep_times.append(t[0])
        inf_times.append(t[1])
        post_times.append(t[2])

elif framework == "v7":
    yolov7_dir = str(root_dir / "frameworks" / "yolov7")
    if yolov7_dir not in sys.path:
        sys.path.insert(0, yolov7_dir)
        
    import cv2
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression
    
    model = attempt_load(model_path, map_location=device)
    model.eval().float()
    
    dummy = torch.zeros((1, 3, 640, 640), device=device)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy)
            
    prep_times, inf_times, post_times = [], [], []
    for p in tqdm(img_paths, desc=f"Benchmarking {model_name}"):
        t0 = time.perf_counter()
        im0 = cv2.imread(p)
        img = letterbox(im0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        tensor_img = torch.from_numpy(img).to(device).float() / 255.0
        if tensor_img.ndimension() == 3:
            tensor_img = tensor_img.unsqueeze(0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        with torch.no_grad():
            pred = model(tensor_img, augment=False)[0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        _ = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        prep_times.append((t1 - t0) * 1000.0)
        inf_times.append((t2 - t1) * 1000.0)
        post_times.append((t3 - t2) * 1000.0)

avg_prep = float(np.mean(prep_times))
avg_inf = float(np.mean(inf_times))
avg_post = float(np.mean(post_times))
avg_tot = avg_prep + avg_inf + avg_post

out = {
    "model_name": model_name,
    "preprocess_ms": round(avg_prep, 2),
    "inference_ms": round(avg_inf, 2),
    "postprocess_ms": round(avg_post, 2),
    "total_ms": round(avg_tot, 2),
    "fps": round(1000.0 / avg_tot, 2) if avg_tot > 0 else 0.0,
    "inference_fps": round(1000.0 / avg_inf, 2) if avg_inf > 0 else 0.0
}

print("RESULT_JSON:" + json.dumps(out))
