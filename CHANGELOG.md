# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-12

### Added
- Complete multi-stage YOLOv8l urban traffic detection pipeline.
- Custom dataset processing for BDD100K, IDD, and Datacluster Auto-rickshaw integration.
- Automated hard-example mining and rider class oversampling logic.
- Real-time OpenCV inference core (`src/main.py`) and FastAPI webcam streaming (`src/dashboard/app.py`).
- Benchmark framework for latency profiling (`scripts/benchmark.py`).
- Qualitative side-by-side comparison generator (`scripts/generate_qualitative_figs.py`).
- Automated experiment parsing and CSV generation (`scripts/experiment_manager.py`).
- Extensive academic documentation ensuring paper-to-code traceability.
