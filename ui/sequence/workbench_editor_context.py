"""Independent queue data used by configuration editing and VE acquisition setup."""
import os

from base import load_config
from base.hardware_analysis_channel_sync import update_sequence_analysis_channels
from base.input_calibration_range import resolve_test_full_scale_peak_volts
from base.load_config import LoadUiConfig
from base.recording_channel_selection import (
    RecordingChannelSelectionError, canonicalize_recording_input_channels,
)
from base.sequence_queue_references import queue_path_key
from base.vkinging_recording_config import resolve_ve_recording_device
from consts import error_code


def _is_independent_analysis_queue(data):
    """Validate the complete queue required for editor channel synchronization."""
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    first = data[0].get("seq1")
    if not isinstance(first, dict):
        return False
    for group in data:
        if not isinstance(group, dict):
            return False
        for sequence in group.values():
            if not isinstance(sequence, dict):
                return False
            acquisition = sequence.get("acq", {})
            analysis = sequence.get("analysis_list", {})
            if (not isinstance(acquisition, dict) or not isinstance(acquisition.get("detail", {}), dict)
                    or not isinstance(analysis, dict)):
                return False
            if any(isinstance(item, dict) and item.get("type") is not None
                   and not isinstance(item["type"], str) for item in analysis.values()):
                return False
    if "detail" not in first.get("acq", {}):
        return False
    analysis = first.get("analysis_list", {})
    order = analysis.get("display_sequence", [])
    return isinstance(order, list) and all(
        isinstance(name, str) and isinstance(analysis.get(name), dict)
        and isinstance(analysis[name].get("type"), str)
        for name in order
    )


def _is_sequence_config_path(path):
    if not path or not isinstance(path, (str, bytes, os.PathLike)):
        return False
    load_code, data = LoadUiConfig.load_sequence_config_from_json(path)
    return load_code == error_code.OK and _is_independent_analysis_queue(data)


def resolve_editor_ve_device(queue_data, device, profile_store, calibration_store):
    """Use the loaded queue, consulting shared stores only for a missing rate."""
    detail = queue_data[0]["seq1"]["acq"]["detail"] if queue_data else {}
    fallback = None
    if "sample_rate" not in detail:
        if profile_store is None or calibration_store is None:
            raise ValueError("VE recording requires the shared profile and calibration stores")
        fallback = profile_store.load(device, calibration_store)
    return resolve_ve_recording_device(device, detail, fallback_profile=fallback)


class WorkbenchEditorContext:
    """Own an independent queue snapshot without writing its file or registry."""

    def __init__(self, selected_path, queue_data, logger):
        self.selected_path = selected_path
        self.queue_data = queue_data
        self._logger = logger

    @classmethod
    def load(cls, logger):
        registry_dir = os.path.dirname(load_config.SEQUENCE_CONFIG_REGISTRY_PATH)
        registry = {
            name: os.path.normpath(os.path.join(registry_dir, path))
            if isinstance(path, str) and path else path
            for name, path in LoadUiConfig._load_sequence_config_registry().items()
        }
        if "using_config_path" in registry and registry["using_config_path"] is None:
            return cls(None, [], logger)

        selected = registry.get("using_config_path")
        if not _is_sequence_config_path(selected):
            selected = None
            user_keys = sorted(name for name in registry if name not in ("using_config_path", "默认配置"))
            for name in user_keys:
                path = registry[name]
                if isinstance(path, str) and os.path.exists(path) and _is_sequence_config_path(path):
                    selected = path
                    break
            default = registry.get("默认配置")
            if selected is None and _is_sequence_config_path(default):
                selected = default

        load_code, data = LoadUiConfig.load_sequence_config_from_json(selected)
        if load_code != error_code.OK or not _is_independent_analysis_queue(data):
            return cls(None, [], logger)
        return cls(selected, data, logger)

    def update_channels(self, channels):
        if not self.queue_data:
            return True
        if not _is_independent_analysis_queue(self.queue_data):
            self._logger.warning(
                "Independent analysis queue cannot update channels: invalid runtime structure in %s",
                self.selected_path,
            )
            return False
        try:
            channels = canonicalize_recording_input_channels(channels)
        except RecordingChannelSelectionError as error:
            self._logger.warning("Independent analysis channel update failed: %s", error)
            return False
        update_sequence_analysis_channels(self.queue_data, channels)
        return True

    def configuration_deleted(self, target):
        if (target.kind == "queue" and self.selected_path
                and queue_path_key(self.selected_path, os.path.dirname(target.path))
                == queue_path_key(target.path, os.path.dirname(target.path))):
            self.selected_path = None
            self.queue_data = []

    def resolve_ve_device(self, device, profile_store, calibration_store):
        return resolve_editor_ve_device(self.queue_data, device, profile_store, calibration_store)

    def test_full_scale_peak_volts(self):
        return resolve_test_full_scale_peak_volts(self.queue_data)
