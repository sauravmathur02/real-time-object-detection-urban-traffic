import glob

folders = [
    "data1/bdd_balanced/labels/train",
    "data1/bdd_balanced/labels/val"
]

fixed = 0

for folder in folders:
    for file in glob.glob(folder + "/*.txt"):
        new_lines = []

        with open(file) as f:
            for line in f:
                parts = line.split()
                cls = int(parts[0])

                if cls == 5:
                    continue          # remove train class

                if cls > 5:
                    cls -= 1          # shift classes down

                parts[0] = str(cls)
                new_lines.append(" ".join(parts))

        with open(file, "w") as f:
            f.write("\n".join(new_lines))

        fixed += 1

print("Fixed files:", fixed)