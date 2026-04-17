from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


_RESERVED_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


@dataclass(frozen=True)
class MesWriteResult:
    ok: bool
    message: str


def normalize_mes_file_base(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        return os.path.expandvars(os.path.expanduser(raw))
    except Exception:
        return raw


def select_mes_export_config(analysis_config: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(analysis_config, dict):
        return None, None

    display_sequence = analysis_config.get("display_sequence")
    if not isinstance(display_sequence, list):
        return None, None

    for item_name in display_sequence:
        if not isinstance(item_name, str):
            continue
        item_cfg = analysis_config.get(item_name)
        if not isinstance(item_cfg, dict):
            continue
        if item_cfg.get("type") != "Excel":
            continue
        if not item_cfg.get("enabled", True):
            continue
        if item_cfg.get("save_mes_enabled") is not True:
            continue
        return item_name, item_cfg

    return None, None


def resolve_mes_output_path(cfg: dict[str, Any]) -> str:
    base = normalize_mes_file_base((cfg or {}).get("mes_file_base"))
    file_name = str((cfg or {}).get("mes_file_name") or "")
    _root, ext = os.path.splitext(file_name)
    final_name = file_name if ext else f"{file_name}.txt"
    return os.path.join(base, final_name)


def build_mes_payload(sn: str, label: str) -> str:
    return f"{sn},{label}"


def _is_valid_mes_file_name(text: str) -> bool:
    raw = "" if text is None else str(text)
    trimmed = raw.strip()
    if not trimmed:
        return False
    if raw[-1] in {" ", "."}:
        return False
    if any(ord(ch) < 32 for ch in raw):
        return False

    drive, _tail = os.path.splitdrive(raw)
    if drive:
        return False
    if trimmed in {".", ".."}:
        return False
    if any(ch in raw for ch in '<>:"/\\|?*'):
        return False
    if os.path.basename(raw) != raw:
        return False
    if any(sep in raw for sep in ("/", "\\")):
        return False

    normalized_basename = trimmed.split(".", 1)[0].rstrip(" .").upper()
    if normalized_basename in _RESERVED_DEVICE_NAMES:
        return False

    return True


def _validate_mes_runtime_config(cfg: dict[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(cfg, dict):
        return False, "invalid mes config: config must be a dict"

    base = normalize_mes_file_base(cfg.get("mes_file_base"))
    file_name = str(cfg.get("mes_file_name") or "")

    if not base:
        return False, "invalid mes config: mes_file_base is required"
    if not file_name.strip():
        return False, "invalid mes config: mes_file_name is required"
    if not _is_valid_mes_file_name(file_name):
        return False, "invalid mes config: mes_file_name is malformed"
    if not os.path.isdir(base):
        os.makedirs(base, exist_ok=True)

    return True, ""


def write_mes_result(cfg: dict[str, Any] | None, *, sn: str, label: str, logger) -> MesWriteResult:
    ok, message = _validate_mes_runtime_config(cfg)
    if not ok:
        return MesWriteResult(ok=False, message=message)

    path = resolve_mes_output_path(cfg or {})
    payload = build_mes_payload(sn, label)

    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(payload)
    except OSError as exc:
        return MesWriteResult(ok=False, message=str(exc))

    if logger is not None:
        logger.info(f"mes_write_ok path={path} payload={payload}")

    return MesWriteResult(ok=True, message=path)


__all__ = [
    "MesWriteResult",
    "build_mes_payload",
    "normalize_mes_file_base",
    "resolve_mes_output_path",
    "select_mes_export_config",
    "write_mes_result",
]
