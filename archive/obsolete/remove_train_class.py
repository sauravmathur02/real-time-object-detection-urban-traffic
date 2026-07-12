import os

LABEL_ROOT = "data1/bdd_no_train/labels"
IGNORE_CLASS = 5  # train class id

for split in ["train", "val"]:
    label_dir = os.path.join(LABEL_ROOT, split)

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        new_lines = []

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            class_id = int(parts[0])

            if class_id == IGNORE_CLASS:
                continue

            if class_id > IGNORE_CLASS:
                class_id -= 1

            parts[0] = str(class_id)
            new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("Train class removed and labels remapped successfully.")