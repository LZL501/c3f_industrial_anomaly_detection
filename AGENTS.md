# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `c3f/`. Dataset loading and pseudo-anomaly synthesis
are under `c3f/data/`; backbone, C3F memory fusion, decoder, segmentation, and
loss modules are under `c3f/models/`. The plain PyTorch training loop and metric
implementations are in `c3f/engine.py` and `c3f/metrics.py`. Use `tools/train.py`
and `tools/eval.py` as entry points. Dataset templates are in `configs/`, helper
launchers are in `scripts/`, and deterministic checks are in `tests/`. Do not
commit `data/`, `runs/`, `artifacts/`, checkpoints, or generated foreground
previews.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install runtime and test dependencies.
- `pytest -q`: run model, metric, foreground, and checkpoint tests.
- `ruff check c3f tools tests && ruff format --check c3f tools tests`: lint and verify formatting.
- `python tools/train.py --config configs/mvtec.yaml --data-root data/MVTec-AD --texture-root data/dtd/images --category bottle`: train one MVTec category.
- `python tools/eval.py --config configs/mvtec.yaml --checkpoint runs/mvtec_c3f_bottle/best.pth --data-root data/MVTec-AD --category bottle`: evaluate a checkpoint.
- `tensorboard --logdir runs`: inspect scalar and image logs.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, type hints for public interfaces, and
`snake_case` for functions/variables. Classes use `PascalCase`; configuration
keys and filenames use lowercase `snake_case`. Keep paths configurable rather
than embedding machine-specific locations. Prefer small, explicit modules over
framework wrappers.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_*.py`. Add focused tests for tensor
shapes, loss formulas, metric behavior, foreground constraints, and checkpoint
compatibility. Model changes also require a 256x256 forward smoke test; training
changes require at least one generator and discriminator backward pass.

## Commit & Pull Request Guidelines

Write short imperative commit subjects, for example `Align C3F training losses`.
Pull requests must explain behavioral impact, list validation commands, and
report affected datasets or checkpoints. Include foreground or anomaly previews
when data synthesis changes, and metrics when evaluation behavior changes.

## Security & Configuration

Never commit credentials, proxy addresses, private dataset paths, or machine
allocations. Supply them through environment variables or command-line
overrides. Foreground masks are required by default; do not disable validation
without documenting why pseudo anomalies may leave the target object.
