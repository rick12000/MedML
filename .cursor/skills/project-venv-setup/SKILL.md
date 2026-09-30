---
name: project-venv-setup
description: Create a named Python virtual environment in the user profile with uv or conda and install the project. Use when a repository needs a Python environment. If an env instruction named env already exists, follow that file instead.
---

# Project Virtual Environment Setup

## Workflow

1. Before any other step, follow `publish-agent-instructions` for `name: env`. If an instruction with that name already exists, stop this workflow and follow that file. Do not create another environment.

2. Require packaging at the repository root. At least one of `pyproject.toml`, `requirements.txt`, or `setup.py` must exist. If none exists, abort. Tell the user that no environment could be created and no code could be run because the project does not yet have proper packaging. Do not install tools, create an environment, or write env files.

3. Resolve the current user's profile from the environment, not from the repository and not from the current working directory. On Windows use `$env:USERPROFILE` / `%USERPROFILE%` (for example `C:\Users\<username>` from `$env:USERNAME` / `%USERNAME%`). Create the environment under that profile.

4. Detect tooling from a shell started in the user profile:
   - `uv --version`
   - `conda --version`

5. Select the backend:
   - If `uv` is available, use **uv**.
   - Else if `conda` is available, use **conda**.
   - Else install **uv** for the current user, verify `uv --version`, then use **uv**.

6. Choose an environment name from the repository folder name, the `pyproject.toml` `name`, or the package name in `setup.py`. Use a short, lowercase, hyphenated identifier (for example `my-app`).

7. Create the named environment in the user profile:
   - **uv**: `uv venv --prompt <name> "$env:USERPROFILE\.venvs\<name>"` (cmd: `uv venv --prompt <name> "%USERPROFILE%\.venvs\<name>"`). The environment directory is `<user-profile>\.venvs\<name>`. Do not create `.venv` in the repository.
   - **conda**: `conda create -n <name> python=<version>`. Pick the Python version from `pyproject.toml`, `.python-version`, `runtime.txt`, or CI config; otherwise use a recent stable 3.x.

8. Install project dependencies into that environment. Run install commands from the repository root. If `uv` is on PATH, use `uv pip`. Otherwise use `pip`. Prefer an editable install so later runs can skip reinstall.
   - **uv**, when `pyproject.toml` or `setup.py` exists: `uv pip install -e . --python "$env:USERPROFILE\.venvs\<name>\Scripts\python.exe"`. Use `uv sync` only when the project already uses a uv lockfile, and point it at the same interpreter.
   - **uv**, when only `requirements.txt` exists: `uv pip install -r requirements.txt --python "$env:USERPROFILE\.venvs\<name>\Scripts\python.exe"`.
   - **conda**, when `pyproject.toml` or `setup.py` exists: `conda activate <name>`, then `pip install -e .`.
   - **conda**, when only `requirements.txt` exists: `conda activate <name>`, then `pip install -r requirements.txt`.

9. If install fails because of version conflicts or resolver errors, resolve them and retry:
   - Read the error and adjust the conflicting pins in `pyproject.toml`, `requirements.txt`, `setup.py`, or `setup.cfg`.
   - Retry after each coherent change.
   - Stop after repeated failure and summarize the blockers.
   - When a retry succeeds, keep the resolution in the manifest files you changed.
   - Report those manifest changes in the final reply.

10. Tell the user the environment is ready. Include the environment name, the absolute directory, and the activation commands with the name filled in.
    - **uv** — environment directory `<user-profile>\.venvs\<name>`. Run activation from the repository root.
      - PowerShell: `& "$env:USERPROFILE\.venvs\<name>\Scripts\Activate.ps1"`
      - cmd: `call %USERPROFILE%\.venvs\<name>\Scripts\activate.bat`
    - **conda**: `conda activate <name>`.

11. Follow `publish-agent-instructions` with this environment's values filled into the body. These files are how the next session reuses the environment instead of repeating this workflow.
    - `name`: `env`
    - `description`: `Use this repository's Python virtual environment before installing, running, or testing Python.`
    - `body`: the template below. For **conda**, the Activate section is only `conda activate <name>`. For **uv**, include the absolute environment directory and both activation commands.

```markdown
# Python environment

Use this virtual environment for Python commands. If this repository is only a Python application, use it for every command.

## Activate

Environment directory: <user-profile>\.venvs\<name>
Run activation from the repository root.

PowerShell:

    & "$env:USERPROFILE\.venvs\<name>\Scripts\Activate.ps1"

cmd:

    call %USERPROFILE%\.venvs\<name>\Scripts\activate.bat

## Run

From the repository root, with this environment active:

1. If `uv` is on PATH, use `uv pip`. Otherwise use `pip`.
2. Prefer an editable install of this package: `uv pip install -e . --python "<env-dir>\Scripts\python.exe"` or, after activation, `pip install -e .`. If the only manifest is `requirements.txt`, use `uv pip install -r requirements.txt --python "<env-dir>\Scripts\python.exe"` or `pip install -r requirements.txt`.
3. If that editable install is already done, skip reinstall before later runs.
4. Run the requested code with this environment.
5. If install fails on version conflicts, update `pyproject.toml`, `setup.py`, `setup.cfg`, or `requirements.txt`, retry, and report those edits in the final reply.
```
