# ML Project Coding Rules

Generic standards for Python ML projects. Copy to CLAUDE.md or project root.

---

## Project Structure

```
project_name/
├── src/                  # Core library code (importable modules)
│   ├── model/            # Model architectures
│   ├── data/             # Data loading and preprocessing
│   ├── training/         # Training loops, optimizers, schedulers
│   └── utils/            # Shared utilities
├── scripts/              # Standalone executables (not importable)
├── notebooks/            # Exploration and analysis only
├── tests/                # Unit and integration tests
├── configs/              # Config files (JSON, YAML, or Python)
├── requirements.txt      # Pinned dependencies
└── README.md
```

**Rules:**
- `src/` contains reusable library code — no side effects, no direct I/O
- `scripts/` contains executables — CLI args, file I/O, orchestration
- `notebooks/` for exploration only — no production logic lives here
- Never import from `scripts/` in library code
- No circular imports between submodules

---

## Naming Conventions

| Target | Convention | Example |
|--------|-----------|---------|
| Files/modules | `snake_case.py` | `data_loader.py` |
| Classes | `PascalCase` | `EnergyModel`, `DataPipeline` |
| Functions/methods | `snake_case()` | `load_checkpoint()`, `get_optimizer()` |
| Variables | `snake_case` | `batch_size`, `num_epochs` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_ATOMS`, `DEFAULT_CUTOFF` |
| Type aliases | `PascalCase` | `Array`, `ConfigDict`, `PyTree` |
| Private helpers | `_leading_underscore()` | `_normalize_features()` |
| Registry dicts | `UPPER_SNAKE_CASE` | `NONLINEARITY`, `OPTIMIZERS` |

---

## Python Style

### Imports

Order: stdlib → third-party → local. Blank line between groups.

```python
import os
import json
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from . import utils
from .model import backbone
```

- Never use wildcard imports (`from x import *`)
- Use explicit relative imports within a package
- Group ML framework imports (jax, torch, tf) together

### Type Hints

Always annotate function signatures. Define module-level type aliases for repeated types.

```python
Array = np.ndarray
ConfigType = dict[str, Any]

def train_step(
    params: PyTree,
    batch: dict[str, Array],
    learning_rate: float,
) -> Tuple[PyTree, float]:
    ...
```

- Use `Optional[T]` not `T | None` for Python < 3.10 compat
- Use `Tuple`, `List`, `Dict` from `typing` for Python < 3.9 compat
- Never annotate `self`
- Return type annotation required on all public functions

### Constants and Registries

Use module-level dicts as registries for swappable components:

```python
ACTIVATIONS: dict[str, Callable] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "swish": nn.silu,
}

def get_activation(name: str) -> Callable:
    if name not in ACTIVATIONS:
        raise ValueError(f'Activation "{name}" not found. Options: {list(ACTIVATIONS)}')
    return ACTIVATIONS[name]
```

---

## Module Structure

Consistent ordering within every file:

1. Copyright / license header (if applicable)
2. Module docstring
3. Imports
4. Module-level type aliases
5. Module-level constants and registries
6. Helper functions (prefixed `_` if private)
7. Classes
8. Public factory / construction functions

---

## Functions and Classes

### Functions

- Single responsibility — one function does one thing
- Prefer pure functions (no hidden state mutations) in library code
- Factory functions return constructed objects from config:

```python
def model_from_config(cfg: ConfigDict) -> nn.Module:
    return MyModel(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )
```

### Classes

- Use dataclasses or framework module classes (Flax `nn.Module`, PyTorch `nn.Module`)
- Declare all fields at class level with type annotations
- `__call__` or `forward` is the primary computation method

```python
@dataclass
class TrainerConfig:
    learning_rate: float
    num_epochs: int
    batch_size: int
    schedule: str = "cosine"
```

### Docstrings

Required for: all public classes, complex functions, non-obvious behavior.
Not required for: simple property accessors, obvious one-liners.

Format:
```python
def compute_loss(predictions: Array, targets: Array, reduction: str = "mean") -> float:
    """Compute MSE loss between predictions and targets.

    Args:
        predictions: Model output, shape (batch, output_dim).
        targets: Ground truth, same shape as predictions.
        reduction: One of "mean", "sum", "none".

    Returns:
        Scalar loss value (or array if reduction="none").
    """
```

No docstring needed when the function name + signature is self-explanatory.

---

## Error Handling

- Raise `ValueError` for bad input values
- Raise `TypeError` for wrong types
- Raise `FileNotFoundError` for missing files
- Use descriptive messages including the bad value and valid options

```python
# Good
raise ValueError(f'Schedule "{cfg.schedule}" not supported. Choose from: {VALID_SCHEDULES}')

# Bad
raise ValueError("Invalid schedule")
```

- No bare `except:` — always catch specific exceptions
- No silent failures — if something goes wrong, raise
- Assertions for internal invariants only (not user input validation)

---

## Config Management

Use structured config objects, not raw dicts or argparse Namespace.

- Access via dot notation: `cfg.learning_rate` not `cfg["learning_rate"]`
- Provide `default_config()` factory for each configurable component
- Document every config field in the default config

```python
def default_training_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 1e-3
    config.num_epochs = 100
    config.batch_size = 32
    config.schedule = "cosine"       # Options: "constant", "cosine", "linear"
    config.weight_decay = 1e-4
    return config
```

---

## ML-Specific Rules

### Model Definitions

- All hyperparameters declared as class fields (not hardcoded in `forward`/`__call__`)
- No magic numbers inside model logic — define as named constants or config fields
- Models are stateless classes — state lives in params/weights outside the class

### Data

- Data loading is always lazy (return iterators/generators, not loaded tensors)
- Separate: raw data loading → preprocessing → batching → augmentation
- All data transforms are pure functions: `transform(data) -> data`
- Validate data shapes and dtypes at pipeline boundaries, not inside model code

### Checkpointing

- Save: model params + optimizer state + epoch/step + config
- Load: verify config matches before restoring params
- One checkpoint per experiment run, uniquely named by timestamp or run ID

### Experiments

- Every run logs: config, metrics per epoch, final metrics
- Reproducibility: log random seeds, library versions, git commit hash
- Metrics go to structured logs (not print statements)

---

## Logging

Use Python `logging` module in library code. Use `print()` only in scripts.

```python
import logging
logger = logging.getLogger(__name__)

# Library code
logger.info("Starting training step %d", step)
logger.warning("Checkpoint not found, training from scratch")

# Scripts only
print(f"Saved results to {output_path}")
```

---

## Scripts

- All scripts use `argparse` or `absl.flags` for CLI args — no hardcoded paths
- Entry point always guarded:

```python
def main(argv):
    ...

if __name__ == "__main__":
    main()
```

- Scripts validate inputs early and fail fast with clear messages
- No business logic in scripts — orchestrate library functions only

---

## Testing

- Test files mirror source structure: `tests/model/test_backbone.py` for `src/model/backbone.py`
- Test function names: `test_<what>_<condition>()`
- Fixtures in `conftest.py` at the relevant directory level
- Test real behavior — avoid mocking core logic
- At minimum: one test per public function for happy path + one for invalid input

```python
def test_get_activation_valid_name():
    fn = get_activation("relu")
    assert fn(np.array([-1.0, 1.0])).tolist() == [0.0, 1.0]

def test_get_activation_invalid_name_raises():
    with pytest.raises(ValueError, match="not found"):
        get_activation("unknown")
```

---

## Dependencies

- Pin all dependencies with exact versions in `requirements.txt`
- Separate `requirements-dev.txt` for linting, testing, notebooks
- No version ranges in production requirements — reproducibility matters
- Document why unusual pins exist (compatibility constraints) in comments

---

## Git Workflow

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- Branch names: `<type>/<short-description>` — e.g. `feat/add-attention-layer`
- One logical change per PR
- No force-push to main/master
- Squash noisy WIP commits before merging

---

## Code Quality Checklist

Before merging:
- [ ] All public functions have type annotations
- [ ] All public classes/complex functions have docstrings
- [ ] No hardcoded paths, magic numbers, or placeholder strings
- [ ] Error messages include the bad value and valid alternatives
- [ ] New functions have at least one test
- [ ] Config fields documented in `default_config()`
- [ ] No unused imports
