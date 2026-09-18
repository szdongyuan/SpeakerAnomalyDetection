"""Versioned, atomic storage for the last unfinished product-test round."""

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from consts.running_consts import DEFAULT_DIR


def product_test_progress_path():
    # Frozen applications may load modules from a temporary extraction directory.
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(DEFAULT_DIR)
    return root / "ui" / "ui_config" / "product_test_progress.json"


def _encode_metadata(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported progress metadata: {type(value).__name__}")


class ProductTestProgressStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else product_test_progress_path()

    def load(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("测试进度文件版本不支持")
        progress = payload["progress"]
        if progress is not None and not isinstance(progress, dict):
            raise ValueError("测试进度内容格式错误")
        return progress

    def save(self, progress):
        data = json.dumps(
            {"version": 1, "progress": progress},
            ensure_ascii=False, indent=2, default=_encode_metadata,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def clear(self):
        # A tombstone also clears corrupt/unsupported old snapshots atomically.
        self.save(None)
