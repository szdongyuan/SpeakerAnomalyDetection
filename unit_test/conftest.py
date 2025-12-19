import os
import sys


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Test environment stability:
# - Numba cache: required for importing librosa under pytest --import-mode=importlib on some setups.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_PROJECT_ROOT, ".numba_cache"))
# - Qt headless backend: required for PyQt tests in CI / sandboxed environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
