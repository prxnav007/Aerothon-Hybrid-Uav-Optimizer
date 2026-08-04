"""Pytest bootstrap: put the repository root on ``sys.path``.

Without this, ``tests/`` is not a package and pytest prepends only the tests
directory, so ``import src.models...`` fails. Keeping it here (rather than in
each test module) means every current and future test file can import the
project the same way.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
