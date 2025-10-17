# Repository Guidelines

## Project Structure & Module Organization
- `kohya_gui/` hosts the Gradio UI components and tab logic; new features usually belong here alongside shared helpers in `common_gui.py`.
- `sd-scripts/` mirrors the upstream training backend; update it via the submodule workflow and avoid direct edits unless coordinated.
- Launcher scripts (`gui.sh`, `gui-uv.sh`, `gui.bat`, `gui.ps1`) wrap environment detection; keep them aligned when adding flags.
- Reference material lives in `docs/`, configuration samples in `config_files/`, and reusable automation in `scripts/`. GPU test artefacts and fixtures are under `test/`.

## Build, Test, and Development Commands
- `uv sync` installs dependencies declared in `pyproject.toml` using the bundled `uv.lock`.
- `./gui-uv.sh --listen 0.0.0.0 --server_port 7860` runs the GUI with the project-managed virtual environment.
- `uv run python kohya_gui.py --headless --config ./config\ example.toml` exercises the CLI pathway without launching a browser.
- `uv run pytest kohya_gui` runs focused module tests; prefix with `CUDA_VISIBLE_DEVICES=` to pin GPUs during heavier flows.

## Coding Style & Naming Conventions
- Target Python 3.10–3.11, use 4-space indentation, and favour explicit imports over star imports.
- Follow PEP 8 naming (`snake_case` for modules/functions, `PascalCase` for classes) and reuse existing enums/constants in `class_*` modules.
- Run `uv run ruff check kohya_gui` before sending a review; use `uv run ruff format` for quick whitespace fixes.

## Testing Guidelines
- Prefer lightweight `pytest` modules colocated with the feature under `kohya_gui/tests` (create the directory if absent) and mirror the naming `test_<feature>.py`.
- Use sample assets from `test/` to avoid embedding large binaries in git; document any additional datasets in PR notes.
- Record GPU, CUDA, and driver details when validating training routines; include command transcripts for reproducibility.

## Commit & Pull Request Guidelines
- Keep commits small, descriptive, and imperative; the history commonly follows `type: summary` patterns such as `chore: update sd-scripts submodule`.
- Reference related issues with `Fixes #<id>` when applicable and describe behavioural changes plus rollback steps in the PR body.
- Attach before/after screenshots or CLI output for UI or logging adjustments, and call out any new environment variables or config keys introduced.

## Security & Configuration Tips
- Never commit `.env`, `config.toml`, or credential-bearing files; sample defaults belong in `config example.toml`.
- Validate third-party weights or models in `models/` via checksums, and document download sources in the PR for traceability.
