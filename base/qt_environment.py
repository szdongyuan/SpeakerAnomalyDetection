import os
import sys
from importlib.util import find_spec
from pathlib import Path


def _candidate_qt_plugin_paths():
    pyqt5_spec = find_spec("PyQt5")
    if pyqt5_spec and pyqt5_spec.submodule_search_locations:
        for location in pyqt5_spec.submodule_search_locations:
            yield Path(location) / "Qt5" / "plugins"

    prefixes = [sys.prefix]
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix and base_prefix not in prefixes:
        prefixes.append(base_prefix)

    for prefix in prefixes:
        yield Path(prefix) / "Lib" / "site-packages" / "PyQt5" / "Qt5" / "plugins"


def ensure_qt_platform_plugin_path():
    for qt_plugins_path in _candidate_qt_plugin_paths():
        qt_platforms_path = qt_plugins_path / "platforms"
        if qt_platforms_path.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins_path))
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms_path))
            return str(qt_platforms_path)
    return None
