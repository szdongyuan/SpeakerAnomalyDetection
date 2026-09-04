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
plugs/unplugs anything. Legacy JSON remains opportunistic. Explicit VE input
records instead retain their stable identity and ordered routes as unavailable
until discovery confirms them. Their rate lives only in the profile store.
"""

import json
import os
import re
from copy import deepcopy
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from base.sound_device_manager import SoundDeviceManager
from base.ve3668n_input import normalize_machine_id, validate_device_snapshot, validate_physical_channels
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
    """Read opportunistic legacy JSON, retaining recognizable broken VE state."""
    if not path or not os.path.isfile(path):
        return None
    raw = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        data = json.loads(raw)
    except (OSError, ValueError):
        # Opportunistic legacy recovery must not erase a recognizable explicit
        # VE choice, even if the document was truncated during an external edit.
        if re.search(r'"input_selection"\s*:|"backend"\s*:\s*"vkinging"', raw):
            return {"input_selection": None}
        return None
    if isinstance(data, dict) and "input_selection" in data:
        return data  # The canonical position is authoritative, even if invalid.
    misplaced = _invalid_root_ve_selection(data)
    if isinstance(data, dict):
        return {**data, **misplaced} if misplaced else data
    return misplaced


def _invalid_root_ve_selection(data):
    """Retain provenance, but never trust a selection in a noncanonical layout."""
    pending, records = [data], []
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if "input_selection" in value:
                records.append(value["input_selection"])
            elif value.get("backend") == "vkinging":
                records.append(value)
            else:
                pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and re.search(r'"input_selection"\s*:|"backend"\s*:\s*"vkinging"', value):
            records.append(None)
    if not records:
        return None
    # A single structured record can retain diagnostic identity/routes. Multiple
    # candidates or text markers provide provenance only, not a chosen device.
    return {"input_selection": records[0] if len(records) == 1 else None,
            "input_selection_error": "VE hardware input_selection must be at the top level of a JSON object"}


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
        spk_idx = int(speaker.get("index"))
        if not is_ve_input(mic):
            mic_idx = int(mic.get("index"))
    except (TypeError, ValueError):
        return
    if is_ve_input(mic):
        # This only assigns the ordinary output slot; it cannot resolve or
        # replace the input default and needs no PortAudio exception wrapper.
        SoundDeviceManager.change_default_output_device(spk_idx)
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


def is_ve_input(device):
    return isinstance(device, dict) and device.get("backend") == "vkinging"


def unavailable_ve_selection(record):
    """Retain explicit identity/routes even when the saved record is corrupt."""
    raw = record if isinstance(record, dict) else {}
    device = {
        "backend": "vkinging", "model": "VE3668N",
        "machine_id": raw.get("machine_id"), "name": "VE3668N",
        "address": None, "physical_channels": raw.get("physical_channels", []),
        "max_input_channels": 8, "input_config": None, "available": False,
        "diagnostic": "正在检查 VE3668N；确认设备和配置前不可用",
    }
    try:
        if (set(raw) != {"schema_version", "backend", "machine_id", "physical_channels"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
                or raw["backend"] != "vkinging"):
            raise ValueError("input_selection schema_version/backend/fields invalid")
        device["machine_id"] = normalize_machine_id(raw["machine_id"])
        device["physical_channels"] = list(validate_physical_channels(raw["physical_channels"]))
    except ValueError as exc:
        device["selection_error"] = str(exc)
        device["diagnostic"] = f"VE 输入选择损坏，不可用；请重新选择设备和通道：{exc}"
    return device


def resolve_ve_input(device, channels, discovered_devices, *, profile_store,
                     calibration_store, diagnostic=""):
    """Reconcile a retained choice with this discovery generation and its profile.

    Diagnostics belong to unavailable UI descriptors, never verified snapshots.
    This does not enumerate native hardware or write profile defaults.
    """
    unavailable = deepcopy(device)
    unavailable.update(available=False, input_config=None)
    try:
        if device.get("selection_error"):
            raise ValueError(device["selection_error"])
        machine_id = normalize_machine_id(device.get("machine_id"))
        channels = validate_physical_channels(channels)
        matches = [item for item in discovered_devices if item.get("machine_id") == machine_id]
        if len(matches) != 1:
            raise ValueError(f"MachineId {machine_id} 不可用；{diagnostic or '请连接同一设备后刷新'}")
        current = validate_device_snapshot(matches[0])
        if not current["available"] or not set(channels).issubset(current["physical_channels"]):
            raise ValueError(f"MachineId {machine_id} 所选 AIN 通道不可用")
        current["input_config"] = profile_store.load(current, calibration_store)
        return current
    except (ValueError, OSError) as exc:
        # Config/identity boundary: retain the exact choice, never substitute a
        # profile default or another input when validation or storage fails.
        unavailable["diagnostic"] = f"VE3668N 不可用：{exc}"
        return unavailable


def restore_or_default(*, path=None, soundcard_only=False, apply_defaults=True) -> Tuple[
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

    Explicit VE choices never use the mic fallback or mutate the input default;
    their ordinary output is applied independently.
    ``soundcard_only`` is for a user-requested switch back to the retained legacy
    configuration (including its shared input/output API), not an automatic
    fallback. ``path`` supports isolated instance use.
    ``apply_defaults=False`` resolves a dialog draft without applying audio
    defaults; existing startup callers retain the default applying behavior.
    """
    try:
        SoundDeviceManager.refresh_available_device()
    except Exception:
        pass

    saved = _read_saved_selection(str(path) if path is not None else _HARDWARE_SELECTION_PATH)
    if (soundcard_only and isinstance(saved, dict) and "input_selection" in saved
            and isinstance(saved.get("soundcard_selection"), dict)):
        saved = saved["soundcard_selection"]
    devices_by_api = _enumerate_devices()
    api_name = _resolve_api_name(saved, devices_by_api)

    mic_device = _find_device_by_name(
        devices_by_api,
        api_name,
        (saved or {}).get("mic_name") if isinstance(saved, dict) else None,
        "mic",
    )
    if not soundcard_only and isinstance(saved, dict) and "input_selection" in saved:
        mic_device = unavailable_ve_selection(saved["input_selection"])
        if saved.get("input_selection_error"):
            mic_device["selection_error"] = saved["input_selection_error"]
            mic_device["diagnostic"] = f"VE 输入选择损坏，不可用：{saved['input_selection_error']}"
        raw_channels = mic_device["physical_channels"]
        mic_channels = list(raw_channels) if isinstance(raw_channels, (list, tuple)) else []
    elif mic_device is not None:
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

    if apply_defaults:
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
    *, path=None, strict=False,
) -> bool:
    """Persist the current selection only if it differs from the JSON on disk.

    Returns ``True`` when a write was performed. ``False`` covers both
    "no change" and "write failed" for existing callers. ``strict=True`` raises
    on failure (False then means unchanged); it is used when explicitly leaving
    VE. VE profile + selection acceptance uses save_ve_selection instead.
    """
    path = str(path) if path is not None else _HARDWARE_SELECTION_PATH
    with _io_lock:
        existing = _read_saved_selection(path)
        if is_ve_input(mic_device):
            try:
                payload = _ve_selection_payload(mic_device, speaker_device, mic_channels,
                    speaker_channels, existing or {})
            except ValueError:
                if strict:
                    raise
                return False
        else:
            payload = _build_payload(mic_device, speaker_device, mic_channels, speaker_channels)
        if payload is None:
            return False
        if existing == payload:
            return False
        written = _atomic_write_json(path, payload)
        if not written and strict:
            raise OSError(f"Could not save hardware selection: {path}")
        return written


def _ve_selection_payload(device, speaker, channels, speaker_channels, existing, *,
                          api_name=None, legacy_mic_device=None, legacy_mic_channels=None,
                          legacy_soundcard_selection=None):
    current = validate_device_snapshot(device)
    channels = validate_physical_channels(channels)
    if not current["available"] or not set(channels).issubset(current["physical_channels"]):
        raise ValueError("VE input or selected physical_channels unavailable")
    payload = _build_payload(None, speaker, [], speaker_channels) or {
        "api_name": None, "speaker_name": None, "speaker_channels": [],
    }
    # Ordinary soundcards use one API for both input and output. Retain that
    # bounded context separately from VE's independently chosen ordinary output.
    legacy = legacy_soundcard_selection
    if legacy is None:
        legacy = existing.get("soundcard_selection")
    if not isinstance(legacy, dict):
        legacy = existing if existing.get("mic_name") else {}
    legacy = {key: deepcopy(legacy[key]) for key in (
        "api_name", "speaker_name", "speaker_channels", "mic_name", "mic_channels") if key in legacy}
    if legacy_mic_device:
        mic_payload = _build_payload(legacy_mic_device, None, legacy_mic_channels, [])
        legacy.update({key: mic_payload[key] for key in ("api_name", "mic_name", "mic_channels")})
    if legacy:
        payload["soundcard_selection"] = legacy
    payload.update(
        api_name=api_name if api_name is not None else (payload.get("api_name") or existing.get("api_name")),
        mic_name=legacy.get("mic_name"),
        mic_channels=deepcopy(legacy.get("mic_channels", [])),
        input_selection={"schema_version": 1, "backend": "vkinging",
                         "machine_id": current["machine_id"], "physical_channels": list(channels)},
    )
    return payload


def _restore_profile_bytes(path, previous):
    """Compensate only this dialog's profile write if selection publication fails."""
    if previous is None:
        path.unlink()
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_ve_selection(mic_device, speaker_device, mic_channels, speaker_channels, *,
                      profile_store, calibration_store, path=None, sample_rate=None,
                      api_name=None, legacy_mic_device=None, legacy_mic_channels=None,
                      legacy_soundcard_selection=None):
    """Strict VE OK boundary: return the committed snapshot, or raise.

    Unlike save_if_changed, an unchanged selection is success. Profile and
    selection are separate files: publish selection last, compensating the
    profile bytes on failure. Calibration is never rolled back; observed
    condition invalidations must remain sticky. Call only on explicit OK with
    one instance owner per store and after the recording busy guard.
    ``legacy_soundcard_selection`` optionally supplies the dialog's unsaved
    five-field ordinary selection; it never includes VE configuration or fs.
    """
    current = validate_device_snapshot(mic_device)
    channels = validate_physical_channels(mic_channels)
    if not current["available"] or not set(channels).issubset(current["physical_channels"]):
        raise ValueError("VE input or selected physical_channels unavailable")
    rate = current["input_config"]["sample_rate"] if sample_rate is None else sample_rate
    path = str(path) if path is not None else _HARDWARE_SELECTION_PATH
    with _io_lock:
        existing = _read_saved_selection(path) or {}
        payload = _ve_selection_payload(current, speaker_device, channels, speaker_channels,
            existing, api_name=api_name, legacy_mic_device=legacy_mic_device,
            legacy_mic_channels=legacy_mic_channels,
            legacy_soundcard_selection=legacy_soundcard_selection)
        try:
            previous = profile_store.path.read_bytes()
        except FileNotFoundError:
            previous = None
        config = profile_store.set_sample_rate(current, rate, calibration_store)
        if existing != payload and not _atomic_write_json(path, payload):
            error = OSError(f"Could not save VE hardware selection: {path}")
            try:
                _restore_profile_bytes(profile_store.path, previous)
            except OSError as rollback_error:
                # The UI remains uncommitted; explicitly diagnose an uncertain
                # disk profile instead of pretending cross-file atomicity.
                raise OSError(f"{error}; profile rollback failed: {rollback_error}") from rollback_error
            raise error
        return {**current, "input_config": config}
