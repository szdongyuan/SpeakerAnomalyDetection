"""Atomic, filesystem-only CSV export for one analysis item across channels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from base.analysis_artifact_paths import (
    build_csv_path,
    format_channel_name,
)


@dataclass(frozen=True)
class CsvExportRecord:
    config_item_name: str
    ok: bool
    file_path: str = ""
    error_message: str = ""


def export_item_csvs(
    storage_context,
    wav_stem,
    config_key,
    analysis_type,
    config,
    channel_outputs,
    channel_labels=None,
):
    """Export enabled result data; no extra save checkboxes are consulted."""
    ordered = sorted(channel_outputs, key=lambda item: int(item["raw_channel"]))
    if not ordered:
        return ()
    tables = _build_tables(
        analysis_type,
        config or {},
        ordered,
        channel_labels or {},
    )
    records = []
    for config_item_name, header, rows in tables:
        path = build_csv_path(
            storage_context,
            wav_stem,
            config_key,
            config_item_name,
        )
        try:
            _write_csv_atomic(path, header, rows)
        except (OSError, TypeError, ValueError, csv.Error) as error:
            records.append(
                CsvExportRecord(
                    config_item_name,
                    False,
                    error_message=str(error),
                )
            )
        else:
            records.append(
                CsvExportRecord(config_item_name, True, str(path))
            )
    return tuple(records)


def _build_tables(analysis_type, config, outputs, channel_labels):
    if analysis_type == "SPL":
        return _spl_tables(config, outputs, channel_labels)
    if analysis_type == "FBA":
        return (
            _curve_table(
                "频段能量",
                "X轴中心频率Hz",
                "频段声压级_Y轴dB",
                outputs,
                channel_labels,
            ),
        )
    if analysis_type == "FFT":
        return (
            _curve_table(
                "FFT频谱",
                "X轴频率Hz",
                "FFT_Y轴dB",
                outputs,
                channel_labels,
            ),
        )
    if analysis_type == "AI":
        header = ("通道", "模型输出值", "判定阈值", "result")
        rows = []
        for output in outputs:
            metrics = dict(output.get("metrics") or {})
            rows.append(
                (
                    format_channel_name(
                        int(output["raw_channel"]),
                        channel_labels,
                    ),
                    _csv_number(metrics.get("model_output_value")),
                    _csv_number(metrics.get("decision_threshold")),
                    str(output.get("judgement") or ""),
                )
            )
        return (("模型输出", header, rows),)
    return ()


def _spl_tables(config, outputs, channel_labels):
    tables = []
    include_overall = bool(config.get("show_overall_spl", False)) or (
        bool(config.get("limit_checked", False))
        and str(config.get("limit_metric", "overall_spl") or "overall_spl")
        == "overall_spl"
    )
    if include_overall:
        header = (
            "通道",
            "总体声压级dB",
            "总体下限dB",
            "总体上限dB",
            "result",
        )
        overall_judged = bool(config.get("limit_checked", False)) and str(
            config.get("limit_metric", "overall_spl") or "overall_spl"
        ) == "overall_spl"
        rows = []
        for output in outputs:
            metrics = dict(output.get("metrics") or {})
            rows.append(
                (
                    format_channel_name(
                        int(output["raw_channel"]),
                        channel_labels,
                    ),
                    _csv_number(metrics.get("overall_spl")),
                    _csv_number(metrics.get("overall_lower_limit")),
                    _csv_number(metrics.get("overall_upper_limit")),
                    str(output.get("judgement") or "") if overall_judged else "",
                )
            )
        tables.append(("总体声压级", header, rows))
    tables.append(
        _curve_table(
            "实时声压级",
            "X轴时间秒",
            "SPL_Y轴dB",
            outputs,
            channel_labels,
        )
    )
    return tuple(tables)


def _curve_table(
    config_item_name,
    x_name,
    y_name,
    outputs,
    channel_labels,
):
    reference = dict(outputs[0].get("csv_curve") or {})
    x_values = list(reference.get("x") or [])
    lower = list(reference.get("lower") or [])
    upper = list(reference.get("upper") or [])
    header = [x_name]
    for output in outputs:
        header.append(
            f"{format_channel_name(int(output['raw_channel']), channel_labels)}_"
            f"{y_name}"
        )
    header.extend(("下限_Y轴dB", "上限_Y轴dB"))

    curves = []
    for output in outputs:
        curve = dict(output.get("csv_curve") or {})
        channel_x = np.asarray(curve.get("x") or [], dtype=np.float64)
        channel_y = np.asarray(curve.get("y") or [], dtype=np.float64)
        if channel_x.size != channel_y.size:
            raise ValueError("curve x/y lengths differ")
        if channel_x.tolist() != x_values:
            if not x_values or channel_x.size == 0:
                raise ValueError("channel curves do not share an x axis")
            channel_y = np.interp(
                np.asarray(x_values, dtype=np.float64),
                channel_x,
                channel_y,
            )
        curves.append(channel_y.tolist())

    rows = []
    for index, x_value in enumerate(x_values):
        row = [_csv_number(x_value)]
        row.extend(_csv_number(curve[index]) for curve in curves)
        row.append(_csv_number(lower[index]) if lower else "")
        row.append(_csv_number(upper[index]) if upper else "")
        rows.append(tuple(row))
    return config_item_name, tuple(header), rows


def _write_csv_atomic(path, header, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _csv_number(value):
    if value is None or isinstance(value, bool):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.10g}"
