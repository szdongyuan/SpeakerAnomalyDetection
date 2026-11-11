from configparser import ConfigParser, Error as IniError
from pathlib import Path

from PyQt5.QtCore import QIODevice, QSaveFile

from consts.running_consts import DEFAULT_DIR


def _preference_path(path):
    if path is not None:
        return Path(path)
    return Path(DEFAULT_DIR) / "ui" / "ui_config" / "workbench_toolbar_preferences.ini"


def load_mode(logger, *, path=None):
    path = _preference_path(path)
    config = ConfigParser(interpolation=None, default_section="")
    try:
        with path.open(encoding="utf-8") as stream:
            config.read_file(stream)
    except FileNotFoundError:
        return "mark"
    except (OSError, UnicodeError, IniError) as error:
        logger.warning("Cannot read workbench toolbar preference %s (%s)", path, error)
        return "mark"

    mode = config.get("General", "mode", fallback="mark")
    if mode in ("test", "mark"):
        return mode
    logger.warning("Invalid workbench toolbar mode in %s", path)
    return "mark"


def save_mode(mode, logger, *, path=None):
    path = _preference_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.warning("Cannot prepare workbench toolbar preference %s (%s)", path, error)
        return

    payload = f"[General]\nmode={mode}\n".encode("ascii")
    output = QSaveFile(str(path))
    output.setDirectWriteFallback(False)
    if not output.open(QIODevice.WriteOnly):
        logger.warning("Cannot open workbench toolbar preference %s (%s)", path, output.errorString())
        return
    if output.write(payload) != len(payload):
        output.cancelWriting()
        logger.warning("Cannot write workbench toolbar preference %s (%s)", path, output.errorString())
        return
    if not output.commit():
        logger.warning("Cannot commit workbench toolbar preference %s (%s)", path, output.errorString())
