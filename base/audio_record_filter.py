"""Read-only recording-name parsing and filtering of existing database rows."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import ntpath
from pathlib import PureWindowsPath
import re

from consts.running_consts import DEFAULT_DIR


_RECORDING_TAIL_RE = re.compile(
    r"^(?P<port>.+)_R(?P<round>[0-9]{4})_(?P<condition>.+)_"
    r"(?P<timestamp>[0-9]{8}-[0-9]{6}(?:-[0-9]{3})?)"
    r"(?:_SN-(?P<serial_number>.+?))?"
    r"(?:_(?P<collision>[0-9]{2}))?$"
)


class UnknownFilterValue(Enum):
    UNKNOWN = "unknown"


# Distinct from both an unrestricted field (None) and a real name '未识别'.
UNKNOWN = UnknownFilterValue.UNKNOWN

METADATA_FILTER_FIELDS = {
    "select_project": "project_key",
    "select_sample_number": "sample_number",
    "select_test_round": "test_round",
    "select_port": "port_name",
    "select_condition": "condition_name",
}


@dataclass(frozen=True)
class AudioFilterMetadata:
    project_key: str | None = None
    project_name: str | None = None
    sample_number: str | None = None
    test_round: int | None = None
    port_name: str | None = None
    condition_name: str | None = None


def parse_recording_stem(stem: str, project: str, model: str, sample: str):
    """Shared historical naming contract used by reports and audio filters."""
    prefix = f"{project}_{model}_{sample}_"
    if not stem.startswith(prefix):
        return {}, "WAV 文件名与项目/型号/样本目录不一致"
    tail = stem[len(prefix):]
    match = _RECORDING_TAIL_RE.fullmatch(tail)
    if match is None:
        return {}, "WAV 文件名缺少可解析的端口、轮次、档位或时间"
    try:
        timestamp = match.group("timestamp")
        timestamp_format = (
            "%Y%m%d-%H%M%S-%f" if timestamp.count("-") == 2 else "%Y%m%d-%H%M%S"
        )
        recorded_at = datetime.strptime(timestamp, timestamp_format)
    except ValueError:
        return {}, "WAV 文件名中的录制时间无效"
    return {
        "port": match.group("port"),
        "condition": match.group("condition"),
        "test_round": int(match.group("round")),
        "recorded_at": recorded_at,
    }, ""


def parse_audio_filter_metadata(file_path, *, application_root=DEFAULT_DIR):
    """Parse a Windows database path lexically, without testing file existence."""
    raw_path = str(file_path or "").strip()
    if not raw_path:
        return AudioFilterMetadata()
    if not ntpath.isabs(raw_path):
        raw_path = ntpath.join(str(application_root), raw_path)
    path = PureWindowsPath(ntpath.normpath(raw_path))
    if path.suffix.lower() != ".wav" or path.parent.name.lower() != "wav":
        return AudioFilterMetadata()
    sample_dirs = [path.parent.parent]
    if sample_dirs[0].name.lower() == "audio":
        # Prefix validation distinguishes the new layout from a legacy sample
        # whose actual name is 'audio'. Neither branch accesses the filesystem.
        sample_dirs.insert(0, sample_dirs[0].parent)
    for sample_dir in sample_dirs:
        model_dir = sample_dir.parent
        project_dir = model_dir.parent
        if not all((sample_dir.name, model_dir.name, project_dir.name)):
            continue
        parsed, error = parse_recording_stem(
            path.stem, project_dir.name, model_dir.name, sample_dir.name,
        )
        if not error:
            return AudioFilterMetadata(
                project_key=ntpath.normcase(str(project_dir)),
                project_name=project_dir.name,
                sample_number=sample_dir.name,
                test_round=parsed["test_round"],
                port_name=parsed["port"],
                condition_name=parsed["condition"],
            )
    return AudioFilterMetadata()


def matches_filter_value(actual, selected):
    if selected is None:
        return True
    if selected is UNKNOWN:
        return actual is None
    return actual == selected


def natural_sort_key(value):
    return tuple(
        int(part) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def filter_audio_records(rows, metadata_by_id, filters):
    """Return original rows; field comparisons never search unrelated columns."""
    metadata_filters = [
        (field, filters[key]) for key, field in METADATA_FILTER_FIELDS.items()
        if filters.get(key) is not None
    ]
    output = []
    for row in rows:
        metadata = metadata_by_id[row[0]]
        if any(not matches_filter_value(getattr(metadata, field), value)
               for field, value in metadata_filters):
            continue
        if filters.get("select_product_model") is not None and row[2] != filters["select_product_model"]:
            continue
        if "select_sample_rate" in filters and row[3] not in filters["select_sample_rate"]:
            continue
        if "select_labels" in filters and row[5] not in filters["select_labels"]:
            continue
        if filters.get("select_record_date") is not None and row[4] != filters["select_record_date"]:
            continue
        output.append(row)
    return output
