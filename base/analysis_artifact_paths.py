"""Filesystem-only paths for recordings and derived analysis artifacts.

The database keeps its existing WAV record and label fields.  Images, CSV files
and PDF reports are related to a recording by the shared WAV stem, not by a new
database identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re


ARTIFACT_DIRECTORY_NAMES = {
    "wav": "wav",
    "images": "images",
    "csv": "csv",
}
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_INVALID_COMPONENT_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_path_component(value, *, max_length=80):
    """Return a readable and collision-resistant Windows path component."""
    original = str(value or "").strip()
    sanitized = _INVALID_COMPONENT_PATTERN.sub("_", original)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._")
    if not sanitized:
        raise ValueError("path component is empty after sanitization")
    if sanitized.upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"_{sanitized}"
    if len(sanitized) <= max_length:
        return sanitized
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{sanitized[:prefix_length]}_{digest}"


def format_channel_name(raw_channel, channel_labels=None):
    """Return the operator-facing channel name without changing its identity."""
    if type(raw_channel) is not int or raw_channel < 0:
        raise ValueError("raw_channel must be a non-negative integer")

    channel_name = f"CH{raw_channel + 1}"
    normalized_labels = {
        str(channel or "").strip().upper(): str(label or "").strip()
        for channel, label in dict(channel_labels or {}).items()
    }
    label = normalized_labels.get(channel_name, "")
    return f"{channel_name}({label})" if label else channel_name


@dataclass(frozen=True)
class AnalysisStorageContext:
    result_root_directory: str
    project_name: str
    product_model: str
    sample_number: str
    test_round: int
    port_name: str
    condition_name: str
    recording_started_at: datetime

    def __post_init__(self):
        required_text = {
            "result_root_directory": self.result_root_directory,
            "project_name": self.project_name,
            "product_model": self.product_model,
            "sample_number": self.sample_number,
            "port_name": self.port_name,
            "condition_name": self.condition_name,
        }
        for field_name, value in required_text.items():
            if not str(value or "").strip():
                raise ValueError(f"{field_name} is required")
        root = Path(self.result_root_directory).expanduser()
        if not root.is_absolute():
            raise ValueError("result_root_directory must be an absolute path")
        if type(self.test_round) is not int or not 1 <= self.test_round <= 9999:
            raise ValueError("test_round must be between 1 and 9999")
        if not isinstance(self.recording_started_at, datetime):
            raise TypeError("recording_started_at must be a datetime")

    def to_metadata(self):
        """Return plain values suitable for in-process task construction."""
        return {
            "result_root_directory": str(self.result_root_directory),
            "project_name": str(self.project_name),
            "product_model": str(self.product_model),
            "sample_number": str(self.sample_number),
            "test_round": int(self.test_round),
            "port_name": str(self.port_name),
            "condition_name": str(self.condition_name),
            "recording_started_at": self.recording_started_at.isoformat(
                timespec="milliseconds"
            ),
        }


def storage_context_from_metadata(metadata):
    values = dict(metadata or {})
    started_at = values.get("recording_started_at")
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at)
    return AnalysisStorageContext(
        result_root_directory=values.get("result_root_directory", ""),
        project_name=values.get("project_name", ""),
        product_model=values.get("product_model", ""),
        sample_number=values.get("sample_number", ""),
        test_round=values.get("test_round", 0),
        port_name=values.get("port_name", ""),
        condition_name=values.get("condition_name", ""),
        recording_started_at=started_at,
    )


def build_recording_stem(context: AnalysisStorageContext, *, max_length=220):
    """Build 项目_型号_样本_端口_轮次_档位_时间戳."""
    timestamp = context.recording_started_at.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    components = (
        context.project_name,
        context.product_model,
        context.sample_number,
        context.port_name,
        f"R{context.test_round:04d}",
        context.condition_name,
        timestamp,
    )
    stem = "_".join(sanitize_path_component(value) for value in components)
    return _shorten_component(stem, max_length=max_length)


def get_model_directory(context: AnalysisStorageContext):
    root = Path(context.result_root_directory).expanduser().resolve()
    path = (
        root
        / sanitize_path_component(context.project_name)
        / sanitize_path_component(context.product_model)
    )
    _validate_containment(path, root)
    return path


def get_sample_directory(context: AnalysisStorageContext):
    path = get_model_directory(context) / sanitize_path_component(
        context.sample_number
    )
    _validate_containment(path, context.result_root_directory)
    return path


def get_artifact_directory(context: AnalysisStorageContext, artifact_kind):
    directory_name = ARTIFACT_DIRECTORY_NAMES.get(artifact_kind)
    if directory_name is None:
        raise ValueError(f"unsupported artifact kind: {artifact_kind}")
    path = get_sample_directory(context) / directory_name
    _validate_containment(path, context.result_root_directory)
    return path


def build_wav_path(context: AnalysisStorageContext, *, artifact_stem=""):
    stem = (
        sanitize_path_component(artifact_stem, max_length=220)
        if artifact_stem
        else build_recording_stem(context)
    )
    path = get_artifact_directory(context, "wav") / f"{stem}.wav"
    _validate_containment(path, context.result_root_directory)
    return path


def build_wav_image_directory(context, wav_stem):
    directory_name = sanitize_path_component(wav_stem, max_length=220)
    path = get_artifact_directory(context, "images") / directory_name
    _validate_containment(path, context.result_root_directory)
    return path


def build_channel_image_path(
    context,
    wav_stem,
    analysis_item_name,
    raw_channel,
    channel_labels=None,
):
    channel_name = sanitize_path_component(
        format_channel_name(raw_channel, channel_labels),
        max_length=60,
    )
    filename = _shorten_component(
        f"{sanitize_path_component(analysis_item_name, max_length=100)}_"
        f"{channel_name}",
        max_length=180,
    )
    return build_wav_image_directory(context, wav_stem) / f"{filename}.png"


def build_wav_csv_directory(context, wav_stem):
    directory_name = sanitize_path_component(wav_stem, max_length=220)
    path = get_artifact_directory(context, "csv") / directory_name
    _validate_containment(path, context.result_root_directory)
    return path


def build_csv_path(context, wav_stem, analysis_item_name, config_item_name):
    filename = _shorten_component(
        "_".join(
            (
                sanitize_path_component(analysis_item_name, max_length=100),
                sanitize_path_component(config_item_name, max_length=100),
            )
        ),
        max_length=220,
    )
    path = build_wav_csv_directory(context, wav_stem) / f"{filename}.csv"
    _validate_containment(path, context.result_root_directory)
    return path


def build_pdf_path(context: AnalysisStorageContext, *, generated_at=None):
    generated_at = generated_at or datetime.now()
    timestamp = generated_at.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    stem = _shorten_component(
        "_".join(
            (
                sanitize_path_component(context.project_name),
                sanitize_path_component(context.product_model),
                sanitize_path_component(context.sample_number),
                f"R{context.test_round:04d}",
                timestamp,
            )
        ),
        max_length=220,
    )
    directory = get_model_directory(context) / "pdf"
    _validate_containment(directory, context.result_root_directory)
    return directory / f"{stem}.pdf"


def next_available_path(path):
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    for index in range(1, 10000):
        numbered = candidate.with_name(
            f"{candidate.stem}_{index:02d}{candidate.suffix}"
        )
        if not numbered.exists():
            return numbered
    raise FileExistsError(f"no available file name for {candidate}")


def _shorten_component(value, *, max_length):
    text = str(value)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text[:max_length - len(digest) - 1]}_{digest}"


def _validate_containment(path, root):
    absolute_root = os.path.normcase(os.path.abspath(os.fspath(root)))
    absolute_path = os.path.normcase(os.path.abspath(os.fspath(path)))
    if os.path.commonpath((absolute_root, absolute_path)) != absolute_root:
        raise ValueError("artifact path escapes result_root_directory")
