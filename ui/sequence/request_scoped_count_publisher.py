"""Qt-free durable count publishers for request-owned recording workers."""
from __future__ import annotations

from datetime import datetime
import os

from base.atomic_count_store import update_json, update_shared_counts
from consts.running_consts import DEFAULT_DIR


def normalize_count_label(label):
    lowered = str(label or "").strip().lower()
    if lowered == "ok":
        return "OK"
    if lowered == "ng":
        return "NG"
    if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
        return "not_labeled"
    raise ValueError(f"unsupported count label: {label}")


def default_mark_result_path():
    return os.path.normpath(DEFAULT_DIR + "ui/ui_config/mark_result.json")


def default_shared_result_path():
    date_text = datetime.now().strftime("%Y-%m-%d")
    return os.path.normpath(
        DEFAULT_DIR + f"log/test_result_log/{date_text}.dat")


def increment_mark_result(label, *, path=None):
    """Durably increment only mark_result.json; never touch QWidget state."""
    normalized = normalize_count_label(label)
    path = str(path or default_mark_result_path())
    key = {"OK": "ok", "NG": "ng", "not_labeled": "not_labels"}[normalized]
    def mutate(data):
        data = dict(data)
        data["total"] = int(data.get("total", 0) or 0) + 1
        data[key] = int(data.get(key, 0) or 0) + 1
        data["datatime"] = datetime.now().strftime("%Y-%m-%d")
        return data
    update_json(path, mutate, default={
        "total": 0, "ok": 0, "ng": 0, "not_labels": 0, "datatime": ""})
    return normalized


def _read_shared_counts(path):
    counts = {"total": 0, "ok": 0, "ng": 0, "not_labels": 0}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            if separator and key.strip() in counts:
                counts[key.strip()] = int(value.strip() or 0)
    return counts


def increment_shared_result(label, *, path=None):
    """Durably increment only the shared test-count file."""
    normalized = normalize_count_label(label)
    path = str(path or default_shared_result_path())
    key = {"OK": "ok", "NG": "ng", "not_labeled": "not_labels"}[normalized]
    date_text = datetime.now().strftime("%Y-%m-%d")
    def mutate(counts):
        counts = dict(counts)
        counts["total"] += 1
        counts[key] += 1
        return counts
    update_shared_counts(path, mutate, date_text=date_text)
    return normalized


def relabel_mark_result(old_label, new_label, *, path=None):
    old = normalize_count_label(old_label)
    new = normalize_count_label(new_label)
    path = str(path or default_mark_result_path())
    keys = {"OK": "ok", "NG": "ng", "not_labeled": "not_labels"}
    def mutate(data):
        data = dict(data)
        if old != new:
            data[keys[old]] = max(0, int(data.get(keys[old], 0) or 0) - 1)
            data[keys[new]] = int(data.get(keys[new], 0) or 0) + 1
        data["total"] = max(int(data.get("total", 0) or 0), sum(
            int(data.get(key, 0) or 0) for key in ("ok", "ng", "not_labels")))
        data["datatime"] = datetime.now().strftime("%Y-%m-%d")
        return data
    update_json(path, mutate, default={
        "total": 0, "ok": 0, "ng": 0, "not_labels": 0, "datatime": ""})


def relabel_shared_result(old_label, new_label, *, path=None):
    old = normalize_count_label(old_label)
    new = normalize_count_label(new_label)
    path = str(path or default_shared_result_path())
    keys = {"OK": "ok", "NG": "ng", "not_labeled": "not_labels"}
    def mutate(counts):
        counts = dict(counts)
        if old != new:
            counts[keys[old]] = max(0, counts[keys[old]] - 1)
            counts[keys[new]] += 1
        return counts
    update_shared_counts(
        path, mutate, date_text=datetime.now().strftime("%Y-%m-%d"))
