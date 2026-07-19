import sys
import platform

with open("scratch/sys_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Python: {sys.version}\n")
    f.write(f"OS: {platform.platform()} ({platform.system()} {platform.release()})\n")
    try:
        import torch
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
        else:
            f.write("CUDA Version: N/A\nGPU Name: N/A\n")
    except Exception as e:
        f.write(f"PyTorch error: {e}\n")

    try:
        import ultralytics
        f.write(f"Ultralytics Version: {ultralytics.__version__}\n")
    except Exception as e:
        f.write(f"Ultralytics error: {e}\n")

print("Wrote sys_info.txt")
