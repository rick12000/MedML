---
name: project-venv-setup
description: >
  Use when setting up a Python virtual environment for a new or existing project on Windows.
  Handles tool selection (uv vs conda), venv creation, dependency installation, conflict resolution, and optional `.cursor/rules/env.mdc` for future agents.
  Do NOT use for non-Python environments or when the user only wants dependency advice without creating an environment.
---

# Project Virtual Environment Setup

## Workflow

1. Change directory to the repository root (the project workspace).

2. On Windows, use the current user's profile when checking tools or installing `uv` (e.g. `C:\Users\<username>`). Run project commands from the repository root unless the install step requires the user profile.

3. Detect available tooling (from a shell in the user profile or repo root):
   - `uv --version`
   - `conda --version`

4. Select the backend:
   - If `uv` is available → use **uv**.
   - Else if `conda` is available → use **conda**.
   - Else install **uv** for the current user (official Windows install script or equivalent), verify `uv --version`, then use **uv**.

5. Choose an environment name from project context: repository folder name, `pyproject.toml` `name`, or package name in `setup.py`. Use a short, lowercase, hyphenated identifier (e.g. `my-app`). For **uv**, default location is `.venv` in the repo root unless the project already documents another path. For **conda**, create a named env with that identifier.

6. Create the environment:
   - **uv**: `uv venv` (or `uv venv <path>` if the repo already standardizes a path).
   - **conda**: `conda create -n <name> python=<version>` — pick a Python version from `pyproject.toml`, `.python-version`, `runtime.txt`, or CI config; otherwise use a recent stable 3.x.

7. Install project dependencies (activate or use tool-native invocations so packages land in the new env):
   - If `pyproject.toml` exists: prefer editable install — **uv**: `uv pip install -e .` (or `uv sync` when the project uses uv lockfiles); **conda** (env active): `pip install -e .`.
   - Else if `setup.py` exists: `pip install -e .` (or `uv pip install -e .` under uv).
   - Else if `requirements.txt` exists: **uv**: `uv pip install -r requirements.txt`; **conda**: `pip install -r requirements.txt`.
   - Else: inform the user the env is created but no dependency manifest was found; do not invent packages.

8. If install fails due to version conflicts or resolver errors, resolve recursively:
   - Read the error; adjust conflicting pins in `pyproject.toml`, `requirements.txt`, or `setup.py` / `setup.cfg` as appropriate.
   - Retry install after each coherent change.
   - Stop after repeated failure with a short summary of blockers; do not loop indefinitely.
   - When a retry succeeds, persist the resolution in the same manifest files you changed (updated pins, added bounds, split optional deps if the project structure supports it).

9. Tell the user the environment is ready. Include:
   - Environment name and path (uv: `.venv`; conda: env name).
   - **Activation** (Windows):
     - uv / venv: `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.\.venv\Scripts\activate.bat` (cmd).
     - conda: `conda activate <name>`.
   - **Future dependencies**: editable project — `uv pip install -e .` or `pip install -e .`; requirements file — `uv pip install -r requirements.txt` or `pip install -r requirements.txt`; prefer editable installs for local packages so code changes apply without reinstalling the whole tree.

10. If `.cursor/` exists at the repository root, create or update `.cursor/rules/env.mdc` with `alwaysApply: true`. Document for future agents:
    - Which backend (uv vs conda), env name, and venv path.
    - Exact install/update commands for this repo (editable when `pyproject.toml` or `setup.py` is present).
    - Preference: use editable installs for the project package; add new deps to the canonical manifest (`pyproject.toml` or `requirements.txt`), then reinstall/sync — avoid ad-hoc global pip into the wrong interpreter.
    - Tell the user that `env.mdc` was created or updated.

## Rules

- Prefer **uv** over **conda** when both are available.
- Never install dependencies into the system Python when a project env is the goal.
- Match manifest style already used in the repo; do not introduce a second competing dependency format without reason.
- Editable install (`-e`) is the default for installable local packages so agents and developers do not need a full reinstall after every code change.
- Run install and verification commands in the shell; do not only describe them.

## `env.mdc` template

Use this structure (fill in project-specific values):

```markdown
---
description: Python environment and dependency workflow for this repository
alwaysApply: true
---

# Python environment

- Tool: uv | conda
- Env name / path: ...
- Activate (Windows): ...

## Install and update

- ...
- Prefer editable install for this package when developing locally.
- Add or change dependencies in [pyproject.toml | requirements.txt], then run the install command above.
```
