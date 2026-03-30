import os

label_dir = "data1/bdd_no_train/labels/train"

rider_images = []

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0 and int(parts[0]) == 1:  # rider class
                rider_images.append(file.replace(".txt", ".jpg"))
                break

print("Total rider images:", len(rider_images))