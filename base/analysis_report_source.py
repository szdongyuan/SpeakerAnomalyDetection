"""Read-only project index used by the manual analysis-report exporter."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import os
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence

from base.analysis_artifact_names import (
    CHANNEL_IMAGE_RE as _CHANNEL_IMAGE_RE,
    CHANNEL_ARTIFACT_RE as _CHANNEL_ARTIFACT_RE,
    CHANNEL_NAME_RE as _CHANNEL_NAME_RE,
    NATURAL_PART_RE as _NATURAL_PART_RE,
    parse_csv_artifact as _parse_csv_artifact,
    infer_analysis_type as _infer_analysis_type,
    split_segment_channel,
)
from base.audio_record_filter import parse_recording_stem as _parse_recording_stem
from consts import model_consts
from consts.running_consts import DEFAULT_DIR


_SUPPORTED_TYPES = frozenset({"SPL", "Spec", "FBA", "FFT"})


@dataclass(frozen=True, order=True)
class AnalysisItemIdentity:
    """Stable configured-item identity; type only selects the renderer."""

    key: str
    analysis_type: str

    def __post_init__(self):
        key = str(self.key or "").strip()
        analysis_type = _normalize_analysis_type(self.analysis_type)
        if not key:
            raise ValueError("analysis item key is required")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "analysis_type", analysis_type)


@dataclass(frozen=True)
class CandidateAnalysisItem:
    identity: AnalysisItemIdentity
    csv_files: tuple[tuple[str, str], ...] = ()
    image_files: tuple[tuple[int, str], ...] = ()

    @property
    def display_name(self) -> str:
        return self.identity.key

    @property
    def has_values(self) -> bool:
        return bool(self.csv_files)

    @property
    def has_scalar_values(self) -> bool:
        return any(
            role == "总体声压级" for role, _ in self.csv_files)

    @property
    def has_charts(self) -> bool:
        return bool(self.image_files)

    def csv_path(self, role: str) -> str:
        for stored_role, path in self.csv_files:
            if stored_role == role:
                return path
        return ""


@dataclass(frozen=True)
class ReportCandidate:
    wav_path: str
    project: str
    model: str
    sample: str
    port: str = ""
    condition: str = ""
    test_round: int | None = None
    recorded_at: datetime | None = None
    label: str = ""
    database_status: str = "not_found"
    analysis_items: tuple[CandidateAnalysisItem, ...] = ()
    channel_labels: tuple[tuple[str, str], ...] = ()
    channel_mapping_source: str = "分析结果文件"
    issues: tuple[str, ...] = ()

    @property
    def round_text(self) -> str:
        return f"R{self.test_round:04d}" if self.test_round is not None else "—"

    @property
    def recorded_at_text(self) -> str:
        if self.recorded_at is None:
            return "—"
        return self.recorded_at.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def result_text(self) -> str:
        normalized = str(self.label or "").strip().upper()
        if normalized in {"OK", "NG"}:
            return normalized
        return "无判定结果"

    @property
    def data_status(self) -> str:
        return "结果不完整" if self.issues else "完整"

    def analysis_item(self, identity: AnalysisItemIdentity) -> CandidateAnalysisItem | None:
        for item in self.analysis_items:
            if item.identity == identity:
                return item
        return None

    def channel_label_map(self) -> dict[str, str]:
        return dict(self.channel_labels)


@dataclass(frozen=True)
class ProjectReportIndex:
    project_directory: str
    project_name: str
    candidates: tuple[ReportCandidate, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportAnalysisItemOption:
    identity: AnalysisItemIdentity
    available_count: int
    selected_wav_count: int
    has_values: bool
    has_scalar_values: bool
    has_charts: bool

    @property
    def display_name(self) -> str:
        return self.identity.key

    @property
    def coverage_text(self) -> str:
        return f"{self.available_count}/{self.selected_wav_count}"


def scan_project(
    project_directory: str,
    *,
    database_path: str | None = None,
    cancel_requested=None,
) -> ProjectReportIndex:
    """Scan one `<project>/<model>/<sample>/wav` tree without modifying it."""

    root = Path(str(project_directory or "").strip()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("所选项目目录不存在")

    candidates = []
    for wav_path in _iter_project_wavs(root):
        _raise_if_cancelled(cancel_requested)
        candidate = _build_candidate(root, wav_path)
        candidates.append(candidate)

    candidates.sort(key=_candidate_sort_key)
    db_path = str(database_path or model_consts.DATABASE_PATH)
    db_records, database_warning = _query_database_records(
        db_path,
        [candidate.wav_path for candidate in candidates],
        cancel_requested=cancel_requested,
    )
    enriched = []
    for candidate in candidates:
        _raise_if_cancelled(cancel_requested)
        record = db_records.get(_canonical_path(candidate.wav_path))
        issues = list(candidate.issues)
        if database_warning:
            database_status = "unavailable"
            issues.append(database_warning)
            label = ""
        elif record is None:
            database_status = "not_found"
            issues.append("数据库中未找到该 WAV 的判定记录")
            label = ""
        else:
            database_status = "matched"
            label = str(record.get("labels") or "")
        enriched.append(
            replace(
                candidate,
                label=label,
                database_status=database_status,
                issues=tuple(dict.fromkeys(issues)),
            )
        )

    warnings = (database_warning,) if database_warning else ()
    return ProjectReportIndex(
        project_directory=str(root),
        project_name=root.name,
        candidates=tuple(enriched),
        warnings=warnings,
    )


def filter_candidates(
    candidates: Sequence[ReportCandidate],
    filters: Mapping[str, object] | None = None,
) -> list[ReportCandidate]:
    """Apply exact in-memory filters; an explicitly empty multi-select matches none."""

    filters = dict(filters or {})
    multi_fields = {
        "models": "model",
        "ports": "port",
        "conditions": "condition",
        "rounds": "round_text",
    }
    allowed = {}
    for filter_name in multi_fields:
        if filter_name in filters:
            allowed[filter_name] = {
                str(value).strip()
                for value in (filters.get(filter_name) or ())
                if str(value).strip()
            }

    sample_numbers = None
    if "sample_numbers" in filters:
        sample_numbers = {
            str(value).strip().casefold()
            for value in (filters.get("sample_numbers") or ())
            if str(value).strip()
        }
    sample_keyword = str(filters.get("sample_keyword") or "").strip().casefold()
    result_filter = str(filters.get("result") or "all").strip()
    data_status_filter = str(filters.get("data_status") or "all").strip()

    output = []
    for candidate in candidates:
        rejected = False
        for filter_name, attribute in multi_fields.items():
            if filter_name not in allowed:
                continue
            candidate_value = str(getattr(candidate, attribute)) or "—"
            if candidate_value not in allowed[filter_name]:
                rejected = True
                break
        if rejected:
            continue
        if sample_numbers is not None and candidate.sample.casefold() not in sample_numbers:
            continue
        if sample_keyword and sample_keyword not in candidate.sample.casefold():
            continue
        if result_filter != "all" and candidate.result_text != result_filter:
            continue
        if data_status_filter != "all" and candidate.data_status != data_status_filter:
            continue
        output.append(candidate)
    return output


def catalog_analysis_items(
    candidates: Iterable[ReportCandidate],
) -> list[ReportAnalysisItemOption]:
    selected = list(candidates)
    by_identity: dict[AnalysisItemIdentity, dict[str, object]] = {}
    for candidate in selected:
        for item in candidate.analysis_items:
            state = by_identity.setdefault(
                item.identity,
                {
                    "count": 0,
                    "values": False,
                    "scalar_values": False,
                    "charts": False,
                },
            )
            state["count"] = int(state["count"]) + 1
            state["values"] = bool(state["values"] or item.has_values)
            state["scalar_values"] = bool(
                state["scalar_values"] or item.has_scalar_values
            )
            state["charts"] = bool(state["charts"] or item.has_charts)
    return [
        ReportAnalysisItemOption(
            identity,
            int(state["count"]),
            len(selected),
            bool(state["values"]),
            bool(state["scalar_values"]),
            bool(state["charts"]),
        )
        for identity, state in sorted(
            by_identity.items(),
            key=lambda pair: (
                _natural_key(pair[0].key),
                pair[0].analysis_type.casefold(),
            ),
        )
    ]


def default_report_path(project_directory: str, *, now: datetime | None = None) -> str:
    project = Path(str(project_directory or "").strip()).expanduser().resolve()
    output_directory = Path(DEFAULT_DIR).expanduser().resolve() / "audio_data"
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return str(
        output_directory / f"{project.name}_声学测试报告_{timestamp}.pdf"
    )


def _iter_project_wavs(root: Path):
    for model_entry in _iter_directories(root):
        for sample_entry in _iter_directories(Path(model_entry.path)):
            sample_directory = Path(sample_entry.path)
            # Existing recordings retain their original paths and DB identities.
            for wav_directory in (
                sample_directory / "audio" / "wav",
                sample_directory / "wav",
            ):
                try:
                    entries = list(os.scandir(wav_directory))
                except OSError:
                    continue
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(".wav"):
                        yield Path(entry.path).resolve()


def _iter_directories(path: Path):
    try:
        entries = list(os.scandir(path))
    except OSError:
        return ()
    return sorted(
        (entry for entry in entries if entry.is_dir()),
        key=lambda entry: _natural_key(entry.name),
    )


def _build_candidate(
    root: Path,
    wav_path: Path,
) -> ReportCandidate:
    relative = wav_path.relative_to(root)
    model, sample = relative.parts[0], relative.parts[1]
    parsed, parse_error = _parse_recording_stem(
        wav_path.stem,
        root.name,
        model,
        sample,
    )
    csv_directory = root / model / sample / "csv" / wav_path.stem
    image_directory = root / model / sample / "images" / wav_path.stem
    analysis_items = _discover_analysis_items(csv_directory, image_directory)
    channel_labels, mapping_source, channel_issues = _load_candidate_channel_labels(
        csv_directory,
        image_directory,
    )
    issues = []
    if parse_error:
        issues.append(parse_error)
    if not analysis_items:
        issues.append("未找到可导出的分析项产物")
    issues.extend(channel_issues)
    return ReportCandidate(
        wav_path=str(wav_path),
        project=root.name,
        model=model,
        sample=sample,
        port=str(parsed.get("port") or ""),
        condition=str(parsed.get("condition") or ""),
        test_round=parsed.get("test_round"),
        recorded_at=parsed.get("recorded_at"),
        analysis_items=analysis_items,
        channel_labels=channel_labels,
        channel_mapping_source=mapping_source,
        issues=tuple(issues),
    )


def _discover_analysis_items(
    csv_directory: Path,
    image_directory: Path,
) -> tuple[CandidateAnalysisItem, ...]:
    discovered: dict[str, dict[str, object]] = {}
    if csv_directory.is_dir():
        for path in sorted(csv_directory.glob("*.csv"), key=lambda item: _natural_key(item.name)):
            if path.name.casefold() == "channel_mapping.csv":
                continue
            parsed = _parse_csv_artifact(path)
            if parsed is None:
                continue
            key, analysis_type, role = parsed
            state = discovered.setdefault(key, {"types": set(), "csv": [], "images": []})
            state["types"].add(analysis_type)
            state["csv"].append((role, str(path.resolve())))
    if image_directory.is_dir():
        for path in sorted(image_directory.glob("*.png"), key=lambda item: _natural_key(item.name)):
            match = _CHANNEL_IMAGE_RE.fullmatch(path.stem)
            if match is None:
                continue
            key = match.group("key")
            state = discovered.setdefault(key, {"types": set(), "csv": [], "images": []})
            inferred = _infer_analysis_type(key)
            if inferred:
                state["types"].add(inferred)
            state["images"].append((int(match.group("channel")), str(path.resolve())))

    items = []
    for key, state in discovered.items():
        known_types = {value for value in state["types"] if value}
        if len(known_types) == 1:
            analysis_type = next(iter(known_types))
        elif not known_types and state["images"] and not state["csv"]:
            # Spec only persists its rendered channel images. Other supported
            # types have a type-specific CSV role, so an image-only item is a
            # historical Spec result even when its configured name is custom.
            analysis_type = "Spec"
        else:
            analysis_type = "未知"
        items.append(
            CandidateAnalysisItem(
                AnalysisItemIdentity(key, analysis_type),
                tuple(state["csv"]),
                tuple(state["images"]),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (
                _natural_key(item.identity.key),
                item.identity.analysis_type.casefold(),
            ),
        )
    )


def _normalize_analysis_type(value: object) -> str:
    text = str(value or "").strip()
    for supported in _SUPPORTED_TYPES:
        if text.casefold() == supported.casefold():
            return supported
    return "未知"


def _load_candidate_channel_labels(
    csv_directory: Path,
    image_directory: Path,
):
    labels = {}
    issues = []
    if csv_directory.is_dir():
        csv_paths = sorted(
            (
                path
                for path in csv_directory.glob("*.csv")
                if path.name.casefold() != "channel_mapping.csv"
            ),
            key=lambda item: _natural_key(item.name),
        )
        for path in csv_paths:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = csv.reader(stream)
                    header = next(rows, [])
                    for value in header:
                        _add_artifact_channel_label(
                            labels,
                            value,
                            issues,
                            path.name,
                        )
                    try:
                        channel_index = header.index("通道")
                    except ValueError:
                        continue
                    for row in rows:
                        if channel_index < len(row):
                            _add_artifact_channel_label(
                                labels,
                                row[channel_index],
                                issues,
                                path.name,
                            )
            except (OSError, csv.Error, UnicodeError):
                continue

    if image_directory.is_dir():
        for path in sorted(
            image_directory.glob("*.png"),
            key=lambda item: _natural_key(item.name),
        ):
            match = _CHANNEL_IMAGE_RE.fullmatch(path.stem)
            if match is None:
                continue
            _add_channel_label(
                labels,
                f"CH{match.group('channel')}",
                match.group("label") or "",
                issues,
                path.name,
            )

    normalized = _normalized_channel_labels(labels)
    source = "分析结果文件" if normalized else "分析结果文件（未记录通道信息）"
    return normalized, source, tuple(dict.fromkeys(issues))


def _add_artifact_channel_label(labels, value, issues, source_name):
    text = str(value or "").strip()
    if text.startswith(("输出负载", "时间")):
        _, text = split_segment_channel(text)
    match = _CHANNEL_ARTIFACT_RE.match(text)
    if match is None:
        return
    _add_channel_label(
        labels,
        f"CH{match.group('channel')}",
        match.group("label") or "",
        issues,
        source_name,
    )


def _add_channel_label(labels, channel_name, label, issues, source_name):
    channel = str(channel_name).upper()
    text = str(label or "").strip()
    if channel not in labels or (not labels[channel] and text):
        labels[channel] = text
        return
    existing = labels[channel]
    if text and existing and text != existing:
        issues.append(
            f"{channel} 的已保存 Label 不一致：{existing} / {text}（{source_name}）"
        )


def _normalized_channel_labels(labels: Mapping[str, object]):
    normalized = []
    for channel_name, label in dict(labels or {}).items():
        name = str(channel_name or "").strip().upper()
        match = _CHANNEL_NAME_RE.fullmatch(name)
        if match is None:
            continue
        text = str(label or "").strip() or "未配置"
        normalized.append((name, text))
    return tuple(sorted(normalized, key=lambda pair: int(pair[0][2:])))


def _query_database_records(
    database_path: str,
    wav_paths: Sequence[str],
    *,
    cancel_requested=None,
):
    if not wav_paths:
        return {}, ""
    path = Path(str(database_path or "")).expanduser().resolve()
    if not path.is_file():
        return {}, "数据库文件不存在"

    query_values = []
    for wav_path in wav_paths:
        query_values.extend(_database_path_variants(wav_path))
    query_values = list(dict.fromkeys(query_values))
    records = {}
    try:
        uri = path.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            for offset in range(0, len(query_values), 400):
                _raise_if_cancelled(cancel_requested)
                batch = query_values[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT file_path, product_model, record_date, labels "
                    f"FROM audio_data_table WHERE file_path IN ({placeholders})",
                    batch,
                ).fetchall()
                for file_path, product_model, record_date, labels in rows:
                    records[_canonical_db_value(file_path)] = {
                        "product_model": product_model,
                        "record_date": record_date,
                        "labels": labels,
                    }
    except (OSError, sqlite3.Error) as error:
        return {}, f"数据库判定不可读取：{error}"
    return records, ""


def _database_path_variants(wav_path: str):
    absolute = os.path.abspath(os.path.normpath(wav_path))
    variants = [absolute.replace("\\", "/"), absolute]
    application_root = os.path.abspath(os.path.normpath(DEFAULT_DIR))
    try:
        if os.path.commonpath((absolute, application_root)) == application_root:
            relative = os.path.relpath(absolute, application_root)
            variants.extend((relative.replace("\\", "/"), relative))
    except ValueError:
        pass
    return tuple(dict.fromkeys(variants))


def _canonical_db_value(value: object) -> str:
    text = str(value or "").strip()
    path = text if os.path.isabs(text) else os.path.join(DEFAULT_DIR, text)
    return _canonical_path(path)


def _canonical_path(value: object) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(value or ""))))


def _candidate_sort_key(candidate: ReportCandidate):
    return (
        _natural_key(candidate.model),
        _natural_key(candidate.sample),
        _natural_key(candidate.port),
        _natural_key(candidate.condition),
        candidate.test_round if candidate.test_round is not None else 10000,
        candidate.recorded_at or datetime.max,
        candidate.wav_path.casefold(),
    )


def _natural_key(value: object):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART_RE.split(str(value or ""))
    )


def _raise_if_cancelled(cancel_requested):
    if callable(cancel_requested) and cancel_requested():
        raise InterruptedError("项目扫描已取消")
