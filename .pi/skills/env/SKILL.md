---
name: env
description: Use this repository's Python virtual environment before installing, running, or testing Python.
---

# Python environment

Use this virtual environment for all Python commands.

## Environment

- Directory: `<repo-root>\.venv`
- Manifest: `pyproject.toml` + `uv.lock`
- Base + dev dependencies are installed. Optional extras (`neural`, `r`, `impute`) are installed only on request.
- The project package is installed editable. Do not reinstall it if it is already editable.
- uv is the only environment and package manager for this repository.

## Activate

From the repository root:

PowerShell: `.\.venv\Scripts\Activate.ps1`

cmd: `call .venv\Scripts\activate.bat`

Or run without activating: `uv run <command>`. `uv run` uses `.venv` and synchronizes it with the manifest.

## Testing

    uv run pytest
    uv run pytest tests/test_main.py::test_failing_case

## Linting and pre-commit (only if present in the repository)

- **ruff**: `uv run ruff check .` and `uv run ruff check --fix .`
- **pre-commit** (`.pre-commit-config.yaml`): install with `uv pip install pre-commit --python .venv\Scripts\python.exe`. One-off over all files: `uv run pre-commit run --all-files`. Git hooks: `uv run pre-commit install`.
- **prek** (pre-commit alternative): `uv run prek install`. One-off: `uv run prek run --all-files`.

## Updating dependencies

Update the manifest first, then the environment:

- **`pyproject.toml` + `uv`**: use `uv add` (or edit `pyproject.toml` directly), then `uv lock` (if a lockfile exists) and `uv sync`.
- **pip-tools**: edit the `requirements.in` files, then regenerate the frozen lock with `uv run pip-compile requirements.in -o requirements.txt` (repeat per `.in` file), then `uv pip install -r requirements.txt -r requirements-dev.txt --python .venv\Scripts\python.exe`.
- **`requirements.txt` only**: edit it, then `uv pip install -r requirements.txt -r requirements-dev.txt --python .venv\Scripts\python.exe`.

When changing the environment in any way, enforce the eMudhra security advisory:

- uv must be at least 0.13.6, verified with `uv --version`.
- If certifi appears anywhere in the dependency tree, it is pinned at `certifi>=2026.10.19` in the manifest and lockfile.
- pip, if present in `.venv`, must be at least 26.2.2. Upgrade it only with the SHA-256-verified wheel, never a plain `pip install --upgrade pip`:

        .\.venv\Scripts\python.exe -I -m pip install 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'

- If any of this cannot be enforced, ask the user.
