"""Synchronize recorded analysis selections after an explicit hardware change."""

import os
from copy import deepcopy

from base.load_config import LoadUiConfig
from base.product_test_config_refresh import build_product_test_refresh_snapshot
from base.recording_channel_selection import canonicalize_recording_input_channels


def recording_input_selection_key(device, channels):
    """Ignore transient discovery/profile fields and output-only changes."""
    device = device or {}
    if device.get("backend") == "vkinging":
        identity = ("vkinging", device.get("machine_id"))
    else:
        identity = (
            "sounddevice", device.get("name"), device.get("hostapi"), device.get("index")
        )
    return identity, tuple(sorted(set(channels)))


def update_sequence_analysis_channels(data, channels):
    """Update only recorded channel fields, retaining all other queue settings."""
    for group in data:
        for sequence in group.values():
            mode = str(sequence.get("acq", {}).get("mode") or "RECORD_ONLY")
            if mode.strip().upper() != "RECORD_ONLY":
                continue
            for item in sequence.get("analysis_list", {}).values():
                if isinstance(item, dict) and item.get("type") in {
                    "SPL", "Spec", "FBA", "FFT", "LP", "LOUD",
                }:
                    item["analysis_channel"] = channels[0]
                    item["analysis_channels"] = list(channels)


def synchronize_product_analysis_channels(manager, active_file, channels):
    """Save only referenced recording queues, then return their fresh snapshot.

    Each queue write is atomic. A write failure is surfaced to the caller, which
    must block stale runtime configuration until synchronization is retried.
    """
    channels = list(canonicalize_recording_input_channels(channels))
    snapshot = build_product_test_refresh_snapshot(manager, active_file)
    updates = {}
    for queue in snapshot.queue_catalog.values():
        path = os.path.normcase(os.path.realpath(queue["path"]))
        if path in updates:
            continue
        data = deepcopy(queue["data"])
        update_sequence_analysis_channels(data, channels)
        if data != queue["data"]:
            updates[path] = data

    for path, data in updates.items():
        if not LoadUiConfig.save_sequence_config_to_json(data, path):
            raise OSError(f"分析通道同步失败，测试队列未保存：{path}")
    if updates:
        snapshot = build_product_test_refresh_snapshot(manager, active_file)
    return snapshot
