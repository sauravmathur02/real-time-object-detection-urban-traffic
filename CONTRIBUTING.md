# Contributing

Thank you for your interest in contributing to our research repository! 

## Environment Setup
1. Clone the repository and install dependencies via `pip install -r requirements.txt`.
2. Do not commit large dataset images or weights into the repository. Add them to `.gitignore`.
3. Datasets must be localized inside `data1/`.

## Coding Style
- Write modular, clean, and typed Python code (PEP 8).
- Ensure all new scripts contain argparse interfaces and proper logging.
- Do not introduce arbitrary dependencies without discussion.

## Commit Message Format
We follow conventional commits for traceability:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `refactor:` for code restructuring

## Pull Request Guidelines
1. Always open an issue first to discuss major changes.
2. Ensure you have tested your code against `src/main.py` and `scripts/benchmark.py`.
3. Include a detailed description of the changes in the PR.
4. If modifying dataset logic, provide the before/after label counts.
