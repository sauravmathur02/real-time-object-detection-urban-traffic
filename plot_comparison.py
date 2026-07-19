import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['YOLOv5s', 'YOLOv5m', 'YOLOv7', 'YOLOv8n',
          'YOLOv8s', 'YOLOv8m', 'YOLOv8l']

# Metrics
precision = [0.7065, 0.6331, 0.6401, 0.5880, 0.6285, 0.6452, 0.7140]
recall    = [0.4749, 0.4424, 0.4932, 0.3300, 0.4178, 0.4504, 0.5070]
map50     = [0.5244, 0.4705, 0.5140, 0.3630, 0.4599, 0.4895, 0.5530]
map5095   = [0.3094, 0.3044, 0.3305, 0.2350, 0.3010, 0.3277, 0.3490]

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(10,5))

plt.bar(x-1.5*width, precision, width, label='Precision')
plt.bar(x-0.5*width, recall, width, label='Recall')
plt.bar(x+0.5*width, map50, width, label='mAP@50')
plt.bar(x+1.5*width, map5095, width, label='mAP@50-95')

plt.xticks(x, models, fontsize=10)
plt.ylabel('Score', fontsize=11)
plt.xlabel('Model', fontsize=11)

plt.ylim(0, 0.8)
plt.yticks(np.arange(0, 0.81, 0.1))

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.legend(loc='upper left', frameon=True)

plt.tight_layout()

plt.savefig("comparison_metrics.png", dpi=600, bbox_inches='tight')
plt.savefig("comparison_metrics.pdf", bbox_inches='tight')

plt.show()