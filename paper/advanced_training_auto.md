# Training with the Auto Class

You have successfully mapped and integrated your auto-rickshaw images into the dataset. The configuration `data1/bdd_balanced.yaml` uses a 10-class label space with `auto` included.

## Start training

Run the following command from the project root:

```powershell
yolo detect train model=yolov8l.pt data=data1/bdd_balanced.yaml epochs=50 imgsz=640 batch=4 name=train_auto_v1
```

Once training completes, the best weights will be saved to:

```text
runs/detect/train_auto_v1/weights/best.pt
```

You can then run inference on a traffic video using:

```powershell
python src/main.py --source SampleVideosForDetection/15_minutes_of_heavy_traffic_noise_in_India_14-08-2022_720p.mp4 --model runs/detect/train_auto_v1/weights/best.pt
```
