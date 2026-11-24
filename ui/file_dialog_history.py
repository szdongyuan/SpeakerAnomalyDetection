"""Per-entry directory history; dialogs and business file handling stay with callers."""

import json
import logging
import os
from pathlib import Path
import tempfile

from PyQt5.QtCore import QDir, QStandardPaths

from consts.running_consts import DEFAULT_DIR


REFERENCE_AUDIO_SELECT = "reference_audio_select"
FFT_BACKGROUND_AUDIO_SELECT = "fft_background_audio_select"
LOUDNESS_BACKGROUND_AUDIO_SELECT = "loudness_background_audio_select"
AUDIO_CLIP_SOURCE_SELECT = "audio_clip_source_select"
AUDIO_CLIP_SAVE = "audio_clip_save"
QUEUE_IMPORT = "queue_import"
QUEUE_NEW = "queue_new"
QUEUE_FIRST_SAVE = "queue_first_save"
QUEUE_SAVE_AS = "queue_save_as"
PRODUCT_PROJECT_IMPORT = "product_project_import"
PRODUCT_PROGRAM_IMPORT = "product_program_import"
THRESHOLD_CSV_IMPORT = "threshold_csv_import"
THRESHOLD_CSV_EXPORT = "threshold_csv_export"
REPORT_PROJECT_SELECT = "report_project_select"
REPORT_PDF_SAVE = "report_pdf_save"
AUDIO_ARCHIVE_EXPORT = "audio_archive_export"

ENTRY_KEYS = frozenset({
    REFERENCE_AUDIO_SELECT, FFT_BACKGROUND_AUDIO_SELECT,
    LOUDNESS_BACKGROUND_AUDIO_SELECT, AUDIO_CLIP_SOURCE_SELECT, AUDIO_CLIP_SAVE,
    QUEUE_IMPORT, QUEUE_NEW, QUEUE_FIRST_SAVE, QUEUE_SAVE_AS,
    PRODUCT_PROJECT_IMPORT, PRODUCT_PROGRAM_IMPORT,
    THRESHOLD_CSV_IMPORT, THRESHOLD_CSV_EXPORT,
    REPORT_PROJECT_SELECT, REPORT_PDF_SAVE, AUDIO_ARCHIVE_EXPORT,
})

_logger = logging.getLogger("core")


def _history_path() -> Path:
    return Path(DEFAULT_DIR) / "ui" / "ui_config" / "file_dialog_history.json"


def _validate_entry(entry_key: str) -> None:
    if entry_key not in ENTRY_KEYS:
        raise ValueError(f"Unknown file dialog history entry: {entry_key}")


def _load_directories() -> dict:
    try:
        with _history_path().open(encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _logger.warning("Unable to read file dialog history: %s", error)
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("last_directories"), dict):
        _logger.warning("Invalid file dialog history structure")
        return {}
    return data["last_directories"]


def _usable_directory(value) -> str:
    # JSON paths are an external boundary; never resolve relative history via cwd.
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        return ""
    try:
        return os.path.normpath(value) if os.path.isdir(value) else ""
    except (OSError, ValueError):
        return ""


def get_initial_directory(entry_key: str, fallback: str) -> str:
    """Resolve history, caller's directory, Documents, then the user home."""
    _validate_entry(entry_key)
    for candidate in (_load_directories().get(entry_key), fallback):
        directory = _usable_directory(candidate)
        if directory:
            return directory
    documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return _usable_directory(documents) or os.path.normpath(QDir.homePath())


def remember_directory(entry_key: str, directory: str) -> bool:
    """Remember a confirmed directory without making persistence a business gate."""
    _validate_entry(entry_key)
    directory = _usable_directory(directory)
    if not directory:
        _logger.warning("Cannot remember unavailable directory for %s", entry_key)
        return False
    directories = _load_directories()
    if directories.get(entry_key) == directory:
        return True
    directories[entry_key] = directory
    target = _history_path()
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"last_directories": directories}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return True
    except OSError as error:
        _logger.warning("Unable to save file dialog history for %s: %s", entry_key, error)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                _logger.warning("Unable to remove file dialog history temporary file: %s", error)
