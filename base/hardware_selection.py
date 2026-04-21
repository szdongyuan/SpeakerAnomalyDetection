"""Persist & restore the operator's last hardware choice.

Stores ``(api_name, speaker_name, speaker_channels, mic_name, mic_channels)``
in ``configs/hardware_selection.json`` and re-applies it at startup. Each
side falls back independently when its saved device cannot be matched
against the currently enumerated hardware:

* missing mic     -> OS default mic + ``[0]`` (single-channel, safe on
  any sound card; matches the post-PaError-9998 hand-fix in
  ``main_window.__init__`` so a corrupt/missing JSON yields the same
  behaviour as a fresh install).
* missing speaker -> OS default speaker + all of its channels.

Matching uses ``(api_name, device_name)`` rather than the raw
``sounddevice`` index, because indices shuffle whenever the user
plugs/unplugs anything. ``configs/hardware_selection.json`` is treated
as opportunistic state, never as required input -- any I/O or parse
error silently degrades to the all-default path.
"""

import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from base.sound_device_manager import SoundDeviceManager
from consts import error_code, model_consts


# Tests monkeypatch this and write into ``tmp_path`` so a developer's
# real ``configs/hardware_selection.json`` is never touched.
_HARDWARE_SELECTION_PATH = (
    model_consts.DEFAULT_DIR + "configs/hardware_selection.json"
)


_io_lock = threading.Lock()


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def _read_saved_selection(path: str) -> Optional[Dict[str, Any]]:
    """Return the parsed JSON dict, or ``None`` for any failure path."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> bool:
    """Write ``payload`` to ``path`` via tmp+rename. Returns ``True`` on success.

    Crash mid-write only ever leaves the previous file intact. A failed
    write is swallowed and reported as ``False`` -- losing the latest
    hardware choice is strictly less harmful than aborting startup or
    raising during dialog acceptance.
    """
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".hardware_selection_", suffix=".json.tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int_channels(raw: Any, max_channels: int) -> List[int]:
    """Drop non-int / out-of-range entries; preserve order, dedupe."""
    if not isinstance(raw, (list, tuple)) or max_channels <= 0:
        return []
    seen = set()
    cleaned: List[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 0 <= value < max_channels and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned


def _os_default_device(kind: str) -> Optional[Dict[str, Any]]:
    """Return the OS default mic/speaker dict, or ``None`` if unavailable."""
    try:
        code, dev = SoundDeviceManager().get_default_device(kind, refresh=False)
    except Exception:
        return None
    if code != error_code.OK:
        return None
    return dev


def _enumerate_devices() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Defensive wrapper around ``get_device_info``: never raises."""
    try:
        return SoundDeviceManager.get_device_info()
    except Exception:
        return {}


def _find_device_by_name(
    devices_by_api: Dict[str, Dict[str, List[Dict[str, Any]]]],
    api_name: Optional[str],
    name: Optional[str],
    kind: str,
) -> Optional[Dict[str, Any]]:
    """Look up ``name`` under ``api_name`` and the input/output bucket."""
    if not api_name or not name:
        return None
    bucket = "input" if kind == "mic" else "output"
    api_block = devices_by_api.get(api_name)
    if not isinstance(api_block, dict):
        return None
    for dev in api_block.get(bucket, []) or []:
        if isinstance(dev, dict) and dev.get("name") == name:
            return dev
    return None


def _max_channels(device: Optional[Dict[str, Any]], kind: str) -> int:
    if not device:
        return 0
    key = "max_input_channels" if kind == "mic" else "max_output_channels"
    try:
        return max(0, int(device.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _apply_default_device(
    mic: Optional[Dict[str, Any]], speaker: Optional[Dict[str, Any]]
) -> None:
    """Sync ``sd.default.device`` with the resolved indices (best-effort)."""
    try:
        if not mic or not speaker:
            return
        mic_idx = int(mic.get("index"))
        spk_idx = int(speaker.get("index"))
    except (TypeError, ValueError):
        return
    try:
        SoundDeviceManager.change_default_device(mic_idx, spk_idx)
    except Exception:
        # PortAudio occasionally rejects the assignment; the dialog can
        # still recover via on_hardware_window_init when the user opens
        # the hardware menu.
        pass


def _resolve_api_name(
    saved: Optional[Dict[str, Any]],
    devices_by_api: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Optional[str]:
    if not isinstance(saved, dict):
        return None
    api_name = saved.get("api_name")
    if not isinstance(api_name, str):
        return None
    if api_name not in devices_by_api:
        return None
    return api_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def restore_or_default() -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[int],
    List[int],
]:
    """Resolve last hardware choice.

    Returns ``(mic_device, speaker_device, mic_channels, speaker_channels)``
    in the same shape ``main_window`` already consumes. Each side falls
    back independently:

    * mic fallback     -> OS default mic + ``[0]``
    * speaker fallback -> OS default speaker + all available channels

    Side-effect: calls :meth:`SoundDeviceManager.change_default_device`
    so PortAudio's process-wide default tracks the resolved indices.
    """
    try:
        SoundDeviceManager.refresh_available_device()
    except Exception:
        pass

    saved = _read_saved_selection(_HARDWARE_SELECTION_PATH)
    devices_by_api = _enumerate_devices()
    api_name = _resolve_api_name(saved, devices_by_api)

    mic_device = _find_device_by_name(
        devices_by_api,
        api_name,
        (saved or {}).get("mic_name") if isinstance(saved, dict) else None,
        "mic",
    )
    if mic_device is not None:
        max_in = _max_channels(mic_device, "mic")
        mic_channels = _safe_int_channels(saved.get("mic_channels"), max_in)
        if not mic_channels and max_in > 0:
            # Saved channel list is empty or fully out-of-range under the
            # current device -- fall back to In1 rather than recording
            # zero channels (which would crash the streaming path).
            mic_channels = [0]
    else:
        mic_device = _os_default_device("mic")
        max_in = _max_channels(mic_device, "mic")
        mic_channels = [0] if max_in > 0 else []

    speaker_device = _find_device_by_name(
        devices_by_api,
        api_name,
        (saved or {}).get("speaker_name") if isinstance(saved, dict) else None,
        "speaker",
    )
    if speaker_device is not None:
        max_out = _max_channels(speaker_device, "speaker")
        # The hardware dialog currently forces speaker_channels to ``[]``
        # on every OK; an empty saved list is therefore the *normal*
        # case, not a corruption indicator. Truncate but do not back-fill.
        speaker_channels = _safe_int_channels(
            saved.get("speaker_channels"), max_out
        )
    else:
        speaker_device = _os_default_device("speaker")
        max_out = _max_channels(speaker_device, "speaker")
        speaker_channels = list(range(max_out))

    _apply_default_device(mic_device, speaker_device)

    return mic_device, speaker_device, mic_channels, speaker_channels


def _build_payload(
    mic_device: Optional[Dict[str, Any]],
    speaker_device: Optional[Dict[str, Any]],
    mic_channels: Optional[List[int]],
    speaker_channels: Optional[List[int]],
    devices_by_api: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the JSON payload, or ``None`` if there is nothing worth saving.

    Resolves the host-API name from one of the two devices' ``hostapi``
    index -- the hardware dialog never returns ``api_name`` on its own,
    so we have to look it up here.
    """
    if not mic_device and not speaker_device:
        return None
    if devices_by_api is None:
        devices_by_api = _enumerate_devices()

    api_name = None
    try:
        hostapi_idx = None
        for dev in (speaker_device, mic_device):
            if isinstance(dev, dict) and dev.get("hostapi") is not None:
                hostapi_idx = int(dev.get("hostapi"))
                break
        if hostapi_idx is not None:
            info = SoundDeviceManager.get_api_info(hostapi_idx)
            if isinstance(info, dict):
                api_name = info.get("name")
    except Exception:
        api_name = None

    payload: Dict[str, Any] = {
        "api_name": api_name,
        "speaker_name": (speaker_device or {}).get("name") if speaker_device else None,
        "speaker_channels": [
            int(c) for c in (speaker_channels or [])
            if isinstance(c, int) and not isinstance(c, bool)
        ],
        "mic_name": (mic_device or {}).get("name") if mic_device else None,
        "mic_channels": [
            int(c) for c in (mic_channels or [])
            if isinstance(c, int) and not isinstance(c, bool)
        ],
    }
    return payload


def save_if_changed(
    mic_device: Optional[Dict[str, Any]],
    speaker_device: Optional[Dict[str, Any]],
    mic_channels: Optional[List[int]],
    speaker_channels: Optional[List[int]],
) -> bool:
    """Persist the current selection only if it differs from the JSON on disk.

    Returns ``True`` when a write was performed. ``False`` covers both
    "no change" and "write failed"; callers that only want to react to
    success should check the return value, but errors here must never
    propagate (this is wired into the dialog's accept path).
    """
    payload = _build_payload(
        mic_device, speaker_device, mic_channels, speaker_channels
    )
    if payload is None:
        return False
    with _io_lock:
        existing = _read_saved_selection(_HARDWARE_SELECTION_PATH)
        if existing == payload:
            return False
        return _atomic_write_json(_HARDWARE_SELECTION_PATH, payload)
