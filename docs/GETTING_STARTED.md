## Development Workflow

### Local Development
```bash
# Clone and set up
git clone <repo>
cd hierarchical-robot-rl
conda env create -f environment.yml
conda activate robot-rl

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests examples

# Lint
flake8 src tests examples

# Type check
mypy src
```


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