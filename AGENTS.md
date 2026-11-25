# Repository Guidelines

## Project Structure & Module Organization

StyleID provides two Python implementations of training-free image style transfer. The legacy Stable Diffusion 1.4 entry point is `run_styleid.py`; its model code lives under `ldm/`, with configuration in `models/ldm/`. The actively reproducible Diffusers path is under `diffusers_implementation/` and is wrapped for Windows by `run_styleid.ps1`, `run_styleid.bat`, and `run_demo.ps1`. Sample content and style images live in `data/cnt/` and `data/sty/`; qualitative examples are in `data_vis/`. Evaluation scripts are in `evaluation/`, reusable helpers in `util/`, and documentation assets in `asset/`. Treat `.cache/`, `downloads/`, `results/`, `output/`, and `precomputed_feats/` as generated or local-only data.

## Build, Test, and Development Commands

- `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1` creates `.venv`, installs the CUDA-enabled Diffusers stack, and verifies GPU access.
- `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1` runs the included sample pair and writes `results/demo/stylized_image.jpg`.
- `.\.venv\Scripts\python.exe diffusers_implementation\run_styleid_diffusers.py --cnt_fn data\cnt\00.png --sty_fn data\sty\00.png` runs the maintained implementation directly.
- `conda env create -f environment.yaml` prepares the legacy environment; then run `python run_styleid.py --cnt data/cnt --sty data/sty` after supplying the SD 1.4 checkpoint.
- `python util/copy_inputs.py --cnt data/cnt --sty data/sty` prepares duplicated inputs for evaluation.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for constants. Keep CLI flags lowercase with underscores, matching existing options such as `--save_intermediates`. Group standard-library, third-party, and local imports, and avoid wildcard imports in new code. No formatter or linter is configured, so keep changes PEP 8-compatible and focused.

## Testing Guidelines

There is currently no automated test suite or coverage threshold. Validate changes with a small Diffusers smoke run (low `--ddim_steps` where appropriate), confirm the output image is readable, and exercise both CPU/GPU or precision branches when modifying device logic. For metric changes, run the relevant script in `evaluation/` and record the command used.

## Commit & Pull Request Guidelines

This checkout has no commit history from which to infer a convention. Use concise, imperative subjects, for example `Fix CPU precision fallback`, and keep unrelated changes separate. Pull requests should explain the behavior change, list validation commands and hardware/model details, link relevant issues, and include before/after images for output-affecting changes. Do not commit model weights, caches, generated results, or secrets.
