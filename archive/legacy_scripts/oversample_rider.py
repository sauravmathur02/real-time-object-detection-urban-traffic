import os
import shutil

IMG_ROOT = "data1/bdd_no_train/images/train"
LABEL_ROOT = "data1/bdd_no_train/labels/train"

rider_class = 1

rider_files = []

for file in os.listdir(LABEL_ROOT):
    with open(os.path.join(LABEL_ROOT, file), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0 and int(parts[0]) == rider_class:
                rider_files.append(file.replace(".txt", ""))
                break

print("Found rider images:", len(rider_files))

for base in rider_files:
    for i in range(2):  # duplicate twice (3x total)
        new_name = f"{base}_riderdup{i}"

        shutil.copy(
            os.path.join(IMG_ROOT, base + ".jpg"),
            os.path.join(IMG_ROOT, new_name + ".jpg")
        )

        shutil.copy(
            os.path.join(LABEL_ROOT, base + ".txt"),
            os.path.join(LABEL_ROOT, new_name + ".txt")
        )

print("Oversampling completed.")