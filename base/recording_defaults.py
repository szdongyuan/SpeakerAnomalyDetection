"""Device-type recording defaults, independent of UI and recording hardware."""

from collections.abc import Mapping
import json
import logging
import math
import os
from pathlib import Path
import tempfile

from base.recording_preview_config import validate_recording_preview_time_mode
from base.ve3668n_input import validate_range_index, validate_sample_rate
from consts.recording_preview_consts import RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
from consts.running_consts import DEFAULT_DIR
from consts.ve3668n_consts import VE_BACKEND, VE_RANGE_INDEX_CONFIG_KEY


def recording_profile_key(mic):
    if not mic:
        return None
    return "vkinging" if mic.get("backend") == VE_BACKEND else "soundcard"


def _validate_profile_key(profile_key):
    if profile_key not in ("soundcard", "vkinging"):
        raise ValueError("profile_key must be soundcard or vkinging")


def validated_recording_profile(profile_key, detail):
    """Validate present fields and return their device-specific projection."""
    _validate_profile_key(profile_key)
    if not isinstance(detail, Mapping):
        raise ValueError(f"{profile_key} profile must be a mapping")
    fields = ("total_time", "sample_rate", "startup_trim_ms",
              "use_streaming_recording", RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY)
    if profile_key == "vkinging":
        fields += (VE_RANGE_INDEX_CONFIG_KEY,)
    profile = {key: detail[key] for key in fields if key in detail}
    if "total_time" in profile:
        value = profile["total_time"]
        if type(value) not in (int, float):
            raise ValueError("total_time must be a finite positive number")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError("total_time must be a finite positive number") from exc
        if not finite or value <= 0:
            raise ValueError("total_time must be a finite positive number")
    if "sample_rate" in profile:
        value = profile["sample_rate"]
        if profile_key == "vkinging":
            validate_sample_rate(value)
        elif type(value) is not int or value not in (44100, 48000):
            raise ValueError("sample_rate must be integer 44100 or 48000 for soundcard")
    if "startup_trim_ms" in profile:
        value = profile["startup_trim_ms"]
        if type(value) is not int or value < 0:
            raise ValueError("startup_trim_ms must be a nonnegative integer")
    if "use_streaming_recording" in profile:
        if type(profile["use_streaming_recording"]) is not bool:
            raise ValueError("use_streaming_recording must be a boolean")
    if RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY in profile:
        validate_recording_preview_time_mode(profile[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY])
    if VE_RANGE_INDEX_CONFIG_KEY in profile:
        validate_range_index(profile[VE_RANGE_INDEX_CONFIG_KEY])
    return profile


class RecordingDefaultsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path():
        return Path(DEFAULT_DIR) / "ui/ui_config/recording_default_config.json"

    def _read_envelope(self):
        try:
            with self.path.open(encoding="utf-8") as stream:
                envelope = json.load(stream)
        except FileNotFoundError:
            return {"schema_version": 1, "profiles": {}}
        if not isinstance(envelope, dict):
            raise ValueError("recording defaults must be an object")
        version = envelope.get("schema_version")
        if type(version) is not int or version != 1:
            raise ValueError("schema_version must be integer 1")
        if not isinstance(envelope.get("profiles"), dict):
            raise ValueError("profiles must be an object")
        return envelope

    def load(self, profile_key):
        _validate_profile_key(profile_key)
        envelope = self._read_envelope()
        return validated_recording_profile(profile_key, envelope["profiles"].get(profile_key, {}))

    def save(self, profile_key, detail):
        """Atomically replace this profile while preserving other parsed data."""
        profile = validated_recording_profile(profile_key, detail)
        envelope = self._read_envelope()
        envelope["profiles"][profile_key] = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                # Other profiles may already contain nonfinite numbers. Preserve
                # their parsed values; only the newly saved profile is validated.
                json.dump(envelope, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to clean up recording defaults temporary file %s: %s",
                        temp_path, exc,
                    )
