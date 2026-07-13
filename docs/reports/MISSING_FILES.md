# Missing Files

To elevate this repository to a professional, academic, and open-source research standard, the following files should be created and added to the repository:

### 1. Research & Citation
- `CITATION.cff`: Required for academic repositories to define exactly how the project and dataset should be cited by other researchers.
- `LICENSE`: Missing entirely. A standard open-source license (e.g., MIT, Apache 2.0, or GPL-3.0) is necessary.
- `paper/draft.tex` (or equivalent): The actual research manuscript to map claims to code.

### 2. Validation & Evaluation
- `scripts/benchmark.py`: A standardized script to measure inference speed (FPS) and latency (ms) across different YOLO models.
- `scripts/generate_qualitative_figs.py`: A script to programmatically output side-by-side visual comparisons of different models on the same image (Required for reviewer).
- `tests/`: A directory containing unit tests (e.g., `test_detector.py`, `test_bdd_to_yolo.py`) to ensure data conversion logic and model wrappers function correctly.

### 3. Deployment & DevOps
- `Dockerfile`: To containerize the FastAPI Dashboard and YOLO detector, ensuring it works seamlessly on any OS.
- `docker-compose.yml`: For managing environment setups if a database or message queue is ever added.
- `Makefile`: To standardize complex execution commands (e.g., `make train`, `make deploy`, `make evaluate`).
- `.env.example`: To store configuration variables (like camera sources or ports) instead of hardcoding them in `app.py`.

### 4. Configuration
- `configs/models.yaml`: Centralized configuration to track hyperparameters across different runs, rather than relying solely on Ultralytics defaults.
