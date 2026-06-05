"""Root conftest.py — pytest path and shared fixtures."""

import sys
from pathlib import Path

# Make the project root importable so `import dmpsp` works without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent))
