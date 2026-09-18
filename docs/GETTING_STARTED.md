## Development Workflow

### Local Development

This project uses a **shared** virtualenv at `~/.venvs/rl` (Python 3.10,
managed with [`uv`](https://docs.astral.sh/uv/)) rather than a
per-project one — it already has the heavy RL dependencies (`ray`,
`torch`, `gymnasium`) installed, since those are the same across RL
projects and slow to reinstall. Only this project's own package needs
installing into it.

**One-time setup** (already done once for this repo, repeat only if the
venv is recreated or dependencies change):
```bash
# Dev tools (pytest, black, flake8, mypy) -- not part of the RL deps above
uv pip install --python ~/.venvs/rl/bin/python "pytest>=7.0.0" "pytest-cov>=4.0.0" \
    "black>=23.0.0" "flake8>=5.0.0" "mypy>=0.990"

# Editable install of THIS project -- makes `import slap_mapd_coupling`
# work from anywhere without PYTHONPATH tricks, and is what lets an
# editor's language server (Pylance etc.) resolve the imports too, once
# the editor is pointed at this interpreter (see "Editor setup" below).
cd slap-mapd-coupling
uv pip install --python ~/.venvs/rl/bin/python -e .
```

**Day to day**, either activate the venv once per shell:
```bash
source ~/.venvs/rl/bin/activate
cd slap-mapd-coupling

pytest                        # run tests
python examples/00_build_instance.py   # run an example
slap-mapd --help              # the project's own CLI
black src tests examples      # format
flake8 src tests examples     # lint
mypy src                      # type check
deactivate                    # when done
```

or call the venv's binaries directly without activating, which is more
scriptable and what CI should do:
```bash
~/.venvs/rl/bin/python -m pytest
~/.venvs/rl/bin/python examples/00_build_instance.py
~/.venvs/rl/bin/slap-mapd --help
```

If a change is made to `pyproject.toml`'s `dependencies` (new package,
version bump), re-run the editable install step above so `~/.venvs/rl`
picks it up — it will not update itself automatically.

#### Editor setup (fixes "Import could not be resolved")

That error means the editor's Python extension is pointed at a
different interpreter than `~/.venvs/rl/bin/python` (or `-e .` was never
run). This repo's `.vscode/settings.json` already sets
`python.defaultInterpreterPath` to `~/.venvs/rl/bin/python` for VS Code;
if the error persists, explicitly select it: **Ctrl+Shift+P → "Python:
Select Interpreter" → Enter interpreter path... →**
`~/.venvs/rl/bin/python`, then reload the window. For another editor,
point its equivalent "Python interpreter" setting at the same path.
Note `.vscode/` is gitignored here (a personal setting, not shared via
git) — if this repo ever gets a second contributor, either remove that
line from `.gitignore` or have them create the same file locally.


### Continuous Integration (GitHub Actions)
- Run tests on every push
- Check code formatting & linting
- Build Docker image
- Generate coverage report



### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 5.0.4
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
```