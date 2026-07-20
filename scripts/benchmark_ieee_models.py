import os
import sys
import time
import glob
import json
import subprocess
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

MODELS_CONFIG = [
    {"name": "YOLOv5s", "framework": "v5", "weight_path": str(root_dir / "experiments" / "yolov5" / "train_yolov5s_v1" / "weights" / "best.pt")},
    {"name": "YOLOv5m", "framework": "v5", "weight_path": str(root_dir / "experiments" / "yolov5" / "train_yolov5m" / "weights" / "best.pt")},
    {"name": "YOLOv7",  "framework": "v7", "weight_path": str(root_dir / "experiments" / "yolov7" / "train_yolov7" / "weights" / "best.pt")},
    {"name": "YOLOv8n", "framework": "v8", "weight_path": str(root_dir / "runs" / "detect" / "train_yolov8n" / "weights" / "best.pt")},
    {"name": "YOLOv8s", "framework": "v8", "weight_path": str(root_dir / "runs" / "detect" / "train_yolov8s2" / "weights" / "best.pt")},
    {"name": "YOLOv8m", "framework": "v8", "weight_path": str(root_dir / "runs" / "detect" / "train_yolov8m" / "weights" / "best.pt")},
    {"name": "YOLOv8l", "framework": "v8", "weight_path": str(root_dir / "runs" / "detect" / "train_fast2" / "weights" / "best.pt")},
]

SINGLE_MODEL_BENCHMARK_CODE = """
import os
import sys
import time
import glob
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_dir = Path(r"{root_dir}")
framework = "{framework}"
model_path = r"{weight_path}"
model_name = "{model_name}"

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
    for p in tqdm(img_paths, desc=f"Benchmarking {{model_name}}"):
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
    for p in tqdm(img_paths, desc=f"Benchmarking {{model_name}}"):
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
    for p in tqdm(img_paths, desc=f"Benchmarking {{model_name}}"):
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

out = {{
    "model_name": model_name,
    "preprocess_ms": round(avg_prep, 2),
    "inference_ms": round(avg_inf, 2),
    "postprocess_ms": round(avg_post, 2),
    "total_ms": round(avg_tot, 2),
    "fps": round(1000.0 / avg_tot, 2) if avg_tot > 0 else 0.0,
    "inference_fps": round(1000.0 / avg_inf, 2) if avg_inf > 0 else 0.0
}}

print("RESULT_JSON:" + json.dumps(out))
"""

def format_md_table(rows, headers):
    lines = []
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for r in rows:
        row_line = "| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |"
        lines.append(row_line)
    return "\n".join(lines)

def main():
    print("=" * 65)
    print("       IEEE RESEARCH PAPER STANDARDIZED BENCHMARK SUITE       ")
    print("=" * 65)
    
    python_exe = sys.executable
    results = []
    
    for item in MODELS_CONFIG:
        model_name = item["name"]
        framework = item["framework"]
        weight_path = item["weight_path"]
        
        print(f"\n---> Benchmarking {model_name} [{framework.upper()}]...")
        if not os.path.exists(weight_path):
            print(f"Error: Weight file not found: {weight_path}")
            continue
            
        script_content = SINGLE_MODEL_BENCHMARK_CODE.format(
            root_dir=str(root_dir),
            framework=framework,
            weight_path=weight_path,
            model_name=model_name
        )
        
        temp_script = root_dir / "scratch" / f"bench_{model_name.lower()}.py"
        temp_script.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_script, "w") as f:
            f.write(script_content)
            
        try:
            cmd = [python_exe, str(temp_script)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = proc.stdout
            
            json_line = [line for line in output.splitlines() if line.startswith("RESULT_JSON:")]
            if json_line:
                res_data = json.loads(json_line[0].replace("RESULT_JSON:", ""))
                res_row = {
                    "Model": model_name,
                    "Framework": f"YOLO{framework.upper()}" if framework != "v8" else "Ultralytics YOLOv8",
                    "Preprocess (ms)": res_data["preprocess_ms"],
                    "Inference (ms)": res_data["inference_ms"],
                    "Postprocess (ms)": res_data["postprocess_ms"],
                    "Total (ms)": res_data["total_ms"],
                    "FPS": res_data["fps"],
                    "Inference-only FPS": res_data["inference_fps"],
                    "Weight Path": os.path.relpath(weight_path, root_dir)
                }
                results.append(res_row)
                print(f"  Preprocess:  {res_data['preprocess_ms']:.2f} ms")
                print(f"  Inference:   {res_data['inference_ms']:.2f} ms")
                print(f"  Postprocess: {res_data['postprocess_ms']:.2f} ms")
                print(f"  Total:       {res_data['total_ms']:.2f} ms")
                print(f"  FPS:         {res_data['fps']} (Inference-only: {res_data['inference_fps']})")
            else:
                print(f"Failed to parse results for {model_name}. Output:\n{output}")
        except subprocess.CalledProcessError as e:
            print(f"Error benchmarking {model_name}: {e}\nStderr: {e.stderr}")
        finally:
            if temp_script.exists():
                os.remove(temp_script)

    if not results:
        print("No results generated.")
        return

    eval_dir = root_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    csv_path = eval_dir / "benchmark_ieee_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to CSV: {csv_path}")
    
    md_path = eval_dir / "benchmark_ieee_results.md"
    
    headers_perf = ["Model", "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)", "Total (ms)", "FPS", "Inference-only FPS"]
    headers_weights = ["Model", "Framework", "Weight Path"]
    
    md_content = f"# IEEE Research Paper YOLO Inference Benchmark Results\n\n"
    md_content += f"- **GPU Hardware:** NVIDIA GeForce RTX 4070\n"
    md_content += f"- **Image Resolution:** 640x640\n"
    md_content += f"- **Batch Size:** 1\n"
    md_content += f"- **Precision:** FP32\n"
    md_content += f"- **Sample Size:** 200 validation images evaluated per model\n\n"
    md_content += f"## Performance Breakdown\n\n"
    md_content += format_md_table(results, headers_perf)
    md_content += "\n\n"
    md_content += f"## Trained Weight File Mappings\n\n"
    md_content += format_md_table(results, headers_weights)
    md_content += "\n"
    
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved benchmark markdown report to: {md_path}")

if __name__ == "__main__":
    main()
