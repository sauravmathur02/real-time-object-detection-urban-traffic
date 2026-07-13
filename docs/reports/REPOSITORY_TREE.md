# Repository Tree

Due to the extreme size of the dataset folders (>440,000 files), this tree emphasizes the structural organization and architectural layers.

```text
C:\Repo\object-Detection
|   .gitignore
|   analyze_bdd.py                 # [DUPLICATE] Duplicate of src/ version
|   clean_balance_bdd.py           # [DUPLICATE] Duplicate of src/ version
|   count_rider_images.py          # [DUPLICATE] Duplicate of src/ version
|   downsample_bdd.py              # [DUPLICATE] Duplicate of src/ version
|   fix_labels.py                  # [DUPLICATE] Duplicate of src/ version
|   IMPLEMENTATION_GUIDE.md        # [IMPORTANT] Setup instructions
|   output.mp4                     
|   oversample_rider.py            # [DUPLICATE] Duplicate of src/ version
|   Project_Validation_Report.md   # [IMPORTANT] Internal audit report
|   Project_Validation_Report.pdf  
|   README.md                      # [IMPORTANT] Main entrypoint documentation
|   remove_train_class.py          # [DUPLICATE] Duplicate of src/ version
|   requirements.txt               # [IMPORTANT] Python dependencies
|   runs_inspection_report.json    # [IMPORTANT] Training history dump
|   yolo26l.pt
|   yolo26n.pt
|   yolov5s.pt
|   yolov5su.pt
|   yolov8l.pt
|   yolov8n.pt
|   
+---configs                        # [UNUSED] Empty or stub configuration folder
+---data1                          # [IMPORTANT] Datasets root
|   |   bdd100k.yaml               # [IMPORTANT] Base YOLO config
|   |   bdd_balanced.yaml          # [IMPORTANT] Merged YOLO config
|   |   bdd_no_train.yaml
|   |   
|   +---Annotations                # XML Annotations for Auto dataset
|   +---auto                       # Raw Auto dataset images
|   +---auto_yolo                  # Converted YOLO format Auto dataset
|   +---bdd100k                    # Raw BDD100K subset
|   +---bdd100k_seg                # Segmentation masks (Unused in bounding box workflow)
|   +---bdd_balanced               # Final merged/balanced dataset
|   +---bdd_yolo                   # Converted YOLO format BDD100K
|   +---hard_examples              # Mined False Negatives / Low confidence frames
|   \---IDDDetectionsYOLODataset   # India Driving Dataset subset
+---evaluation
+---experiments
|   \---yolov5                     # Active YOLOv5 runs
|       +---train_yolov5s_v1       # [IMPORTANT] Active training run
|       \---validation_yolov5s
+---frameworks
+---paper                          # Paper drafts / writing resources
+---runs                           # Ultralytics YOLOv8 runs
|   \---detect                     # YOLOv8 checkpoints and validation figures
+---SampleVideosForDetection       # Inference test videos
+---scratch                        # Temporary working dir
+---scripts                        # Execution orchestrators
+---src                            # [IMPORTANT] Core Source Code
|   |   bdd_to_yolo.py             
|   |   bdd_to_yolo_prod.py        
|   |   collect_hard_examples.py   
|   |   convert_auto_to_yolo.py    
|   |   detector.py                # [IMPORTANT] YOLO execution wrapper
|   |   main.py                    # [IMPORTANT] Inference entry point
|   |   merge_auto_yolo.py         
|   |   merge_idd.py               
|   |   oversample_rider.py        
|   |   predict_webcam.py          
|   |   verify_dataset.py          
|   |   
|   \---dashboard                  # FastAPI Web Application
|           app.py                 # [IMPORTANT] Backend API
|           camera_manager.py      # [IMPORTANT] Threaded inference manager
|           
+---uploads
\---venv_gpu                       # Python Virtual Environment
```

## Inventory Analysis

- **Total Files (Estimated):** > 440,000 files (Majority are JPG/TXT from BDD100k, IDD, and Auto datasets).
- **Important Files:** `src/main.py`, `src/dashboard/app.py`, `src/detector.py`, `data1/bdd_balanced.yaml`, `README.md`.
- **Duplicate Files:** The root directory is heavily cluttered with duplicate preprocessing scripts (e.g., `oversample_rider.py`, `analyze_bdd.py`) that also exist in `src/`. These pose a maintenance and configuration drift risk.
- **Unused Files:** The directory `configs/` and models like `train_yolov8s` exist without usage. `bdd100k_seg` contains segmentation masks that are completely unused by the bounding box workflow.
