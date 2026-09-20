"""Read existing artifacts of one recording; never execute analysis or write data."""

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path

from base.analysis_artifact_names import (
    CHANNEL_ARTIFACT_RE, CHANNEL_IMAGE_RE, parse_csv_artifact,
    split_segment_channel,
)
from base.audio_record_filter import natural_sort_key, parse_audio_filter_metadata
from base.spl_csv_schema import resolve_overall_spl_csv_columns
from consts.running_consts import DEFAULT_DIR


@dataclass(frozen=True)
class SavedImage:
    path: str
    channel_id: str = ""
    channel_label: str = ""


@dataclass(frozen=True)
class SavedAnalysisItem:
    key: str
    name: str
    images: tuple = ()
    csv_files: tuple = ()
    channels: tuple = ()


@dataclass(frozen=True)
class SavedRecordingResults:
    wav_path: str
    directories: tuple = ()
    items: tuple = ()
    issues: tuple = ()


@dataclass(frozen=True)
class SavedScalar:
    channel_id: str
    channel_label: str
    segment_label: str
    metric: str
    value: str
    unit: str
    judgement: str


@dataclass(frozen=True)
class ScalarReadResult:
    values: tuple = ()
    issues: tuple = ()


def _check_cancel(cancel_requested):
    if cancel_requested and cancel_requested():
        raise InterruptedError("结果读取已停止")


def _channel(value):
    segment, text = split_segment_channel(value)
    match = CHANNEL_ARTIFACT_RE.match(str(text or value or "").strip())
    if match is None:
        return segment, "", ""
    return segment, f"CH{match.group('channel')}", match.group("label") or ""


def item_channels(item, scalars=()):
    """Return physical channels; conflicting saved labels must not be guessed."""
    labels = {}
    entries = list(item.channels)
    entries.extend((v.channel_id, v.channel_label) for v in scalars)
    for channel, label in entries:
        labels.setdefault(channel, set())
        if label:
            labels[channel].add(label)
    result, issues = [], []
    for channel in sorted(labels, key=natural_sort_key):
        names = labels[channel]
        label = next(iter(names)) if len(names) == 1 else ""
        if len(names) > 1:
            issues.append(f"{channel} 的已保存通道名称不一致，按通道编号显示")
        result.append((channel, f"{channel}({label})" if label else channel or "未记录通道"))
    return tuple(result), tuple(issues)


def discover_recording_results(wav_path, *, application_root=DEFAULT_DIR,
                               cancel_requested=None):
    wav = Path(os.path.abspath(Path(application_root) / wav_path))
    metadata = parse_audio_filter_metadata(str(wav))
    if metadata.project_key is None:
        return SavedRecordingResults(str(wav), issues=("无法定位这条录音的分析结果目录",))
    relative = wav.relative_to(Path(metadata.project_key)).parts
    project = wav.parents[len(relative) - 1]
    sample = project / relative[0] / relative[1]
    states, directories, issues = {}, [], []
    for kind, directory in (
        ("分析图片", sample / "images" / wav.stem),
        ("分析数据", sample / "csv" / wav.stem),
    ):
        _check_cancel(cancel_requested)
        try:
            paths = sorted(directory.iterdir(), key=lambda p: natural_sort_key(p.name))
        except FileNotFoundError:
            continue
        except OSError as error:
            issues.append(f"无法读取分析结果目录：{kind}（{error}）")
            continue
        directories.append((kind, str(directory)))
        for path in paths:
            _check_cancel(cancel_requested)
            if not path.is_file():
                continue
            if kind == "分析图片" and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                match = CHANNEL_IMAGE_RE.fullmatch(path.stem)
                key = match.group("key") if match else f"image:{path.name}"
                name = match.group("key") if match else path.name
                state = states.setdefault(key, {"name": name, "images": [], "csv": [], "channels": []})
                channel = f"CH{match.group('channel')}" if match else ""
                label = (match.group("label") or "") if match else ""
                state["images"].append(SavedImage(str(path), channel, label))
                state["channels"].append((channel, label))
            elif kind == "分析数据" and path.suffix.lower() == ".csv":
                parsed = parse_csv_artifact(path)
                if parsed is None:
                    if path.name.lower() != "channel_mapping.csv":
                        issues.append(f"暂不支持查看的结果文件：{path.name}")
                    continue
                key, _, role = parsed
                state = states.setdefault(key, {"name": key, "images": [], "csv": [], "channels": []})
                state["csv"].append((role, str(path)))
                # Curve channels are encoded in the header. Do not read curve rows.
                if role != "总体声压级":
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as stream:
                            header = next(csv.reader(stream), [])
                        for value in header:
                            _, channel, label = _channel(value)
                            if channel:
                                state["channels"].append((channel, label))
                    except (OSError, UnicodeError, csv.Error) as error:
                        issues.append(f"CSV 表头读取失败：{path.name}（{error}）")
            elif path.name.lower() != "channel_mapping.csv":
                issues.append(f"暂不支持查看的结果文件：{path.name}")
    items = tuple(
        SavedAnalysisItem(key, state["name"], tuple(state["images"]),
                          tuple(state["csv"]), tuple(state["channels"]))
        for key, state in sorted(states.items(), key=lambda pair: natural_sort_key(pair[1]["name"]))
    )
    return SavedRecordingResults(str(wav), tuple(directories), items, tuple(issues))


def _saved_number(value):
    text = str(value or "").strip()
    try:
        valid = math.isfinite(float(text))
    except ValueError:
        valid = False
    return text if valid else "数值不可用"


def read_item_scalars(item, *, cancel_requested=None):
    values, issues = [], []
    for role, filename in item.csv_files:
        if role != "总体声压级":
            continue
        _check_cancel(cancel_requested)
        try:
            with open(filename, encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if "通道" not in (reader.fieldnames or ()):
                    raise ValueError("缺少通道列")
                columns, unit = resolve_overall_spl_csv_columns(reader.fieldnames)
                column = columns[0]
                for row in reader:
                    _check_cancel(cancel_requested)
                    segment, channel, label = _channel(row.get("通道"))
                    if not channel:
                        issues.append(f"未识别数值通道：{Path(filename).name}")
                        continue
                    values.append(SavedScalar(
                        channel, label, segment, role, _saved_number(row.get(column)),
                        unit, str(row.get("result") or "").strip(),
                    ))
        except InterruptedError:
            raise
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            issues.append(f"数值读取失败：{Path(filename).name}（{error}）")
    return ScalarReadResult(tuple(values), tuple(dict.fromkeys(issues)))
