"""Atomic progress storage with read support for older versioned snapshots."""

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from consts.running_consts import DEFAULT_DIR


PROGRESS_RESULTS = ("OK", "NG", "not_labeled")
PROGRESS_CHANNEL_RESULTS = PROGRESS_RESULTS + ("结果不完整",)
PROGRESS_ANALYSIS_FAILURES = ("分析失败", "结果不完整")


def group_progress_analysis_files(audio_path, files, raw_csv_files):
    """Compact registered paths without discovering or adding deletion targets."""
    if not audio_path or audio_path not in files or not raw_csv_files <= files:
        raise ValueError("本轮音频或原始 CSV 与文件归属清单不一致")
    groups = {
        "csv": {"directory": "", "files": []},
        "images": {"directory": "", "files": []},
    }
    for filename in sorted(files - raw_csv_files - {audio_path}):
        path = Path(filename)
        if path.suffix.lower() == ".csv":
            kind = "csv"
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            kind = "images"
        else:
            raise ValueError("本轮文件清单包含无法分组的文件")
        group = groups[kind]
        directory = path.parent.as_posix()
        if group["directory"] and group["directory"] != directory:
            raise ValueError(f"本次录音的 {kind} 文件分布在多个目录，无法保存为单一目录")
        group["directory"] = directory
        group["files"].append(path.name)
    return groups


def expand_progress_analysis_files(groups):
    """Expand only named result files; a directory is never a deletion target."""
    if not isinstance(groups, dict) or set(groups) != {"csv", "images"}:
        raise ValueError("分析结果文件分组格式错误")
    files = set()
    for kind, suffixes in (("csv", {".csv"}), ("images", {".png", ".jpg", ".jpeg"})):
        group = groups[kind]
        if not isinstance(group, dict) or set(group) != {"directory", "files"}:
            raise ValueError("分析结果目录和文件名清单格式错误")
        directory, names = group["directory"], group["files"]
        if (not isinstance(directory, str) or not isinstance(names, list)
                or bool(directory) != bool(names)):
            raise ValueError("分析结果目录与文件名清单不一致")
        for name in names:
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or any(char in name for char in '/\\:\x00')
                    or Path(name).suffix.lower() not in suffixes):
                raise ValueError("分析结果必须登记对应类型的单个文件名")
            files.add(os.path.normpath(os.path.join(directory, name)))
    return files


def compact_progress_result(value):
    """Normalize legacy/runtime labels without accepting arbitrary corrupt text."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ("ok", "ng"):
            return label.upper()
        if label in ("not_labeled", "not labeled", "none", "null", "-", "待判定", "未判定"):
            return "not_labeled"
    raise ValueError("工况或通道判定无效")


def compact_progress_channels(items):
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("通道判定格式错误")
    # Preserve channel failures separately from the product's OK/NG verdict.
    return [
        {
            "raw_channel": item["raw_channel"],
            "result": (
                "结果不完整" if item["result"] == "结果不完整"
                else compact_progress_result(item["result"])
            ),
        }
        for item in items
    ]


def _migrate_v1_progress(state):
    """Read only the fields needed to resume; analysis history stays on disk."""
    completed = state["completed"]
    if (not isinstance(completed, list)
            or not all(isinstance(key, str) for key in completed)
            or len(set(completed)) != len(completed)):
        raise ValueError("已完成工况格式错误")
    for field in ("records", "rows", "results", "counted_labels"):
        if not isinstance(state[field], dict):
            raise ValueError(f"测试进度字段格式错误：{field}")
    if set(state["records"]) != set(completed) or set(state["rows"]) != set(completed):
        raise ValueError("工况结果与完成集合不一致")
    conditions = {}
    for key in completed:
        record, row = state["records"][key], state["rows"][key]
        if not isinstance(record, dict) or not isinstance(row, dict):
            raise ValueError("工况记录格式错误")
        info = record.get("recorded_signal_info", {})
        if not isinstance(info, dict):
            raise ValueError("录音信息格式错误")
        label = (state["results"].get(key) or info.get("labels")
                 or record.get("result_label") or row.get("result"))
        conditions[key] = {
            "result": compact_progress_result(label),
            "channel_results": compact_progress_channels(row["channel_results"]),
        }
        if row.get("result") in PROGRESS_ANALYSIS_FAILURES:
            conditions[key]["analysis_status"] = row["result"]
    result = {key: state[key] for key in (
        "signature", "identity", "group_id", "next_key", "selected_key", "channels",
        "serial_port_index", "waiting_port_idle", "owned_files",
    )}
    counted = state["counted_labels"].get(state["group_id"])
    result.update(
        config_file=None, config_name=None, completed_conditions=conditions,
        counted_result=compact_progress_result(counted) if counted is not None else None,
    )
    return result


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
        self.loaded_version = None

    def load(self):
        self.loaded_version = None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict):
            raise ValueError("测试进度文件格式错误")
        version = payload.get("version")
        if "version" in payload and (type(version) is not int or version not in (1, 2, 3)):
            raise ValueError("测试进度文件版本不支持")
        if "progress" not in payload:
            raise ValueError("测试进度内容缺失")
        progress = payload["progress"]
        if progress is not None and not isinstance(progress, dict):
            raise ValueError("测试进度内容格式错误")
        if progress is not None and version == 1:
            progress = _migrate_v1_progress(progress)
        # Normalize legacy names once at the file boundary; runtime uses only
        # the descriptive names. Reading does not rewrite the saved file.
        completed = progress.get("completed_conditions", {}) if progress is not None else {}
        if isinstance(completed, dict):
            for condition in completed.values():
                if not isinstance(condition, dict):
                    continue  # The resume validator reports malformed conditions.
                for old, new in (
                    ("channels", "channel_results"),
                    ("column_channels", "analysis_column_channels"),
                ):
                    if old in condition:
                        if new in condition and condition[new] != condition[old]:
                            raise ValueError(f"测试进度通道新旧字段不一致：{old}/{new}")
                        condition[new] = condition.pop(old)
        self.loaded_version = version
        return progress

    def save(self, progress):
        data = json.dumps(
            {"progress": progress},
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
