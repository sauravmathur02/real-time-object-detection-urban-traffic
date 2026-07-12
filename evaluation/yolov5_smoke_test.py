import sys
import os
from pathlib import Path

# Add YOLOv5 source directory to sys.path
yolov5_dir = Path("frameworks/yolov5/yolov5_src").resolve()
if str(yolov5_dir) not in sys.path:
    sys.path.append(str(yolov5_dir))

print("Testing YOLOv5 imports...")
failures = []

# List of critical imports to verify
imports_to_test = [
    ("git", "import git"),
    ("yaml", "import yaml"),
    ("scipy", "import scipy"),
    ("seaborn", "import seaborn"),
    ("PIL", "from PIL import Image"),
    ("tqdm", "from tqdm import tqdm"),
    ("cv2", "import cv2"),
    ("pandas", "import pandas"),
    ("matplotlib", "import matplotlib"),
    ("psutil", "import psutil"),
    ("requests", "import requests"),
    ("yolov5_models", "from models.experimental import attempt_load"),
    ("yolov5_utils_general", "from utils.general import check_git_status, check_yaml"),
    ("yolov5_dataloaders", "from utils.dataloaders import create_dataloader"),
]

for name, imp_statement in imports_to_test:
    try:
        exec(imp_statement)
        print(f"OK: {name}")
    except Exception as e:
        print(f"FAIL: {name} (Error: {e})")
        failures.append((name, str(e)))

print("\n--- Summary ---")
if failures:
    print(f"Total import failures: {len(failures)}")
    for name, err in failures:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("All YOLOv5 dependency imports passed successfully!")
    sys.exit(0)
