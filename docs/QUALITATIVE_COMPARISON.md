# Qualitative Comparison Tool

The `scripts/generate_qualitative_figs.py` script is designed to satisfy reviewer requirements by generating side-by-side visual comparisons of different object detection models (e.g., YOLOv5s vs YOLOv8l).

## Purpose
This tool allows researchers to visually evaluate how different architectures perform on identical edge-cases (such as heavy traffic, night scenes, or occluded riders).

## Input Format
The script accepts any directory containing standard images (`.jpg`, `.jpeg`, `.png`, `.bmp`). It recursively scans subdirectories, allowing you to categorize your test images before running the script.

Example input structure:
```text
sample_images/
|-- day/
|   `-- clear_street.jpg
|-- night/
|   `-- dark_alley.jpg
`-- riders/
    `-- scooter.jpg
```

## Output Format
The script generates composite images where the **Original Image** is displayed on the top row, and the predictions from each supplied model are displayed side-by-side on the bottom row.

Outputs are saved mirroring the input directory structure.

Example output structure:
```text
results/qualitative/
|-- day/
|   `-- clear_street_comparison.jpg
|-- night/
|   `-- dark_alley_comparison.jpg
`-- riders/
    `-- scooter_comparison.jpg
```

Each panel includes:
1. A black title bar with the model name.
2. The model's bounding boxes.
3. The detected class names and confidence scores.

## Example Command

To run the qualitative generator, provide the input directory, a list of models in `name=path` format, and the desired output directory.

```bash
python scripts/generate_qualitative_figs.py \
    --images sample_images \
    --models yolov5s=yolov5s.pt yolov8l=yolov8l.pt \
    --output results/qualitative
```

### Adding More Models
The script automatically supports N-number of models. They will dynamically scale horizontally:
```bash
python scripts/generate_qualitative_figs.py \
    --images sample_images \
    --models v5=yolov5s.pt v8n=yolov8n.pt v8l=yolov8l.pt \
    --output results/qualitative
```
