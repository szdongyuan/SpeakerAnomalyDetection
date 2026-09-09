"""Process-wide serialized, crash-safe updates for count files."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import weakref


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS = weakref.WeakValueDictionary()


def _path_lock(path):
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def path_lock_registry_size():
    with _LOCKS_GUARD:
        return len(_PATH_LOCKS)


def _sync_directory(directory):
    flags = getattr(os, "O_RDONLY", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Windows commonly does not permit directory fsync.  The file itself
        # has already been flushed before replace.
        pass
    finally:
        os.close(descriptor)


def _replace_with_retry(source, target, *, attempts=10):
    """Tolerate brief Windows sharing locks while keeping failure bounded."""
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(.005 * (attempt + 1))


def atomic_update_text(path, *, load, mutate, dump, encoding="utf-8"):
    """Lock a path, transform its current value, and atomically replace it."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target) or os.curdir
    os.makedirs(directory, exist_ok=True)
    with _path_lock(target):
        if os.path.exists(target):
            with open(target, "r", encoding=encoding) as stream:
                current = load(stream.read())
        else:
            current = load("")
        updated = mutate(current)
        text = dump(updated)
        descriptor, temporary = tempfile.mkstemp(
            dir=directory, prefix=f".{os.path.basename(target)}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_with_retry(temporary, target)
            _sync_directory(directory)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return updated


def update_json(path, mutate, *, default=None):
    seed = dict(default or {})

    def load(text):
        return json.loads(text) if text.strip() else dict(seed)

    return atomic_update_text(
        path, load=load, mutate=mutate,
        dump=lambda value: json.dumps(value, indent=4, ensure_ascii=False) + "\n")


def parse_shared_counts(text):
    result = {"total": 0, "ok": 0, "ng": 0, "not_labels": 0}
    for line in str(text or "").splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in result:
            result[key] = max(0, int(value.strip() or 0))
    return result


def dump_shared_counts(counts, *, date_text):
    total = max(int(counts.get("total", 0) or 0),
                sum(int(counts.get(key, 0) or 0)
                    for key in ("ok", "ng", "not_labels")))
    ok = max(0, int(counts.get("ok", 0) or 0))
    percent = round(ok / total * 100, 2) if total else 0
    return "".join((
        f"total: {total}\n", f"ok: {ok}\n",
        f"ng: {max(0, int(counts.get('ng', 0) or 0))}\n",
        f"not_labels: {max(0, int(counts.get('not_labels', 0) or 0))}\n",
        f"ok_percent: {percent}%\n", f"datatime: {date_text}\n"))


def update_shared_counts(path, mutate, *, date_text):
    return atomic_update_text(
        path, load=parse_shared_counts, mutate=mutate,
        dump=lambda counts: dump_shared_counts(counts, date_text=date_text))
