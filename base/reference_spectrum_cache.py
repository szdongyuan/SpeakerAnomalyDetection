"""
Reference-data persistence helpers for reference spectrum comparison.

User-facing terminology should prefer "reference data". Internally, this module
manages the JSON payload used to cache generated reference spectra.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from base.core_algorithm.response.reference_spectrum_analyzer import (
    ReferenceSpectrumChannelResult,
    ReferenceSpectrumParams,
)


REFERENCE_DATA_READY = "ready"
REFERENCE_DATA_NOT_GENERATED = "not_generated"
REFERENCE_DATA_OUTDATED = "outdated"


def _normalize_path(file_path: str) -> str:
    return os.path.abspath(os.fspath(file_path)).replace("\\", "/")


def _resolve_channel_label(channel_index: int, channel_labels: Optional[dict]) -> str:
    if isinstance(channel_labels, dict):
        if channel_index in channel_labels:
            return str(channel_labels[channel_index])
        str_key = str(channel_index)
        if str_key in channel_labels:
            return str(channel_labels[str_key])
    return f"CH{int(channel_index) + 1}"


def _extract_shared_frequency_axis(channel_results: list[ReferenceSpectrumChannelResult]) -> np.ndarray:
    if not channel_results:
        raise ValueError("channel_results cannot be empty")
    base_freq_hz = np.asarray(channel_results[0].frequencies_hz, dtype=np.float64)
    for result in channel_results[1:]:
        current_freq_hz = np.asarray(result.frequencies_hz, dtype=np.float64)
        if base_freq_hz.shape != current_freq_hz.shape or not np.allclose(
            base_freq_hz, current_freq_hz, equal_nan=True
        ):
            raise ValueError("All channel results must share the same frequency axis")
    return base_freq_hz


def build_source_file_info(file_path: str) -> dict:
    if not file_path:
        raise ValueError("file_path cannot be empty")
    normalized_path = _normalize_path(file_path)
    if not os.path.exists(normalized_path):
        raise ValueError(f"Reference source file does not exist: {normalized_path}")
    stat = os.stat(normalized_path)
    return {
        "path": normalized_path,
        "size_bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def build_analysis_params_payload(params: ReferenceSpectrumParams) -> dict:
    if not isinstance(params, ReferenceSpectrumParams):
        raise ValueError("params must be ReferenceSpectrumParams")
    return {
        "spectrum_method": "welch",
        "window": str(params.window),
        "nperseg": int(params.nperseg),
        "overlap_ratio": float(params.overlap_ratio),
        "smoothing": int(params.smoothing),
    }


def build_reference_data_payload(
    *,
    reference_source_path: str,
    sample_rate: int,
    channel_results: list[ReferenceSpectrumChannelResult],
    params: ReferenceSpectrumParams,
    channel_labels: Optional[dict] = None,
    frame_count: Optional[int] = None,
) -> dict:
    sample_rate_value = int(sample_rate)
    if sample_rate_value <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    source_file_info = build_source_file_info(reference_source_path)
    shared_freq_hz = _extract_shared_frequency_axis(channel_results)

    channel_payloads = []
    for result in channel_results:
        channel_payloads.append(
            {
                "channel_index": int(result.channel_index),
                "label": _resolve_channel_label(int(result.channel_index), channel_labels),
                "spectrum_db": np.asarray(result.spectrum_db, dtype=np.float64).tolist(),
            }
        )

    duration_sec = None
    if frame_count is not None:
        frame_count_value = int(frame_count)
        if frame_count_value < 0:
            raise ValueError(f"frame_count must be >= 0, got {frame_count}")
        duration_sec = float(frame_count_value / sample_rate_value)

    return {
        "schema_version": 1,
        "source_file": source_file_info,
        "audio_meta": {
            "sample_rate": sample_rate_value,
            "channel_count": int(len(channel_results)),
            "duration_sec": duration_sec,
        },
        "analysis_params": build_analysis_params_payload(params),
        "freq_axis_hz": shared_freq_hz.tolist(),
        "channels": channel_payloads,
    }


def save_reference_data(cache_path: str, payload: dict) -> bool:
    if not cache_path:
        return False
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_reference_data(cache_path: str) -> Optional[dict]:
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def extract_reference_channel_results(payload: dict) -> list[ReferenceSpectrumChannelResult]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    freq_axis_hz = np.asarray(payload.get("freq_axis_hz") or [], dtype=np.float64)
    channels = payload.get("channels") or []
    if freq_axis_hz.size == 0:
        raise ValueError("Reference data is missing freq_axis_hz")
    if not isinstance(channels, list) or not channels:
        raise ValueError("Reference data is missing channel spectra")

    results = []
    for channel in channels:
        if not isinstance(channel, dict):
            raise ValueError("Invalid channel payload in reference data")
        channel_index = int(channel.get("channel_index", 0))
        spectrum_db = np.asarray(channel.get("spectrum_db") or [], dtype=np.float64)
        if spectrum_db.shape != freq_axis_hz.shape:
            raise ValueError("Reference channel spectrum shape does not match freq_axis_hz")
        results.append(
            ReferenceSpectrumChannelResult(
                channel_index=channel_index,
                frequencies_hz=np.asarray(freq_axis_hz, dtype=np.float64),
                spectrum_db=spectrum_db,
            )
        )
    return results


def is_reference_data_outdated(
    payload: dict,
    *,
    reference_source_path: str,
    params: ReferenceSpectrumParams,
) -> bool:
    if not isinstance(payload, dict):
        return True

    try:
        current_source = build_source_file_info(reference_source_path)
    except Exception:
        return True

    payload_source = payload.get("source_file")
    payload_params = payload.get("analysis_params")
    if not isinstance(payload_source, dict) or not isinstance(payload_params, dict):
        return True

    if str(payload_source.get("path") or "") != str(current_source["path"]):
        return True
    if int(payload_source.get("size_bytes", -1)) != int(current_source["size_bytes"]):
        return True
    if int(payload_source.get("mtime", -1)) != int(current_source["mtime"]):
        return True

    current_params = build_analysis_params_payload(params)
    for key in ("spectrum_method", "window", "nperseg", "overlap_ratio", "smoothing"):
        if payload_params.get(key) != current_params.get(key):
            return True
    return False


def get_reference_data_state(
    *,
    reference_source_path: str,
    reference_data_path: str,
    params: ReferenceSpectrumParams,
    current_sample_rate: Optional[int] = None,
    current_channel_count: Optional[int] = None,
) -> str:
    if not reference_source_path or not reference_data_path:
        return REFERENCE_DATA_NOT_GENERATED
    if not os.path.exists(reference_source_path):
        return REFERENCE_DATA_NOT_GENERATED

    payload = load_reference_data(reference_data_path)
    if payload is None:
        return REFERENCE_DATA_NOT_GENERATED

    if is_reference_data_outdated(
        payload,
        reference_source_path=reference_source_path,
        params=params,
    ):
        return REFERENCE_DATA_OUTDATED

    audio_meta = payload.get("audio_meta") or {}
    if current_sample_rate is not None and int(audio_meta.get("sample_rate", -1)) != int(current_sample_rate):
        return REFERENCE_DATA_OUTDATED
    if current_channel_count is not None and int(audio_meta.get("channel_count", -1)) != int(current_channel_count):
        return REFERENCE_DATA_OUTDATED
    return REFERENCE_DATA_READY
