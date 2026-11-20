"""Atomic, filesystem-only CSV export for one analysis item across channels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math

import numpy as np

from base.analysis_artifact_paths import (
    build_csv_path,
    format_channel_name,
)
from base.csv_writer import write_csv_atomic as _write_csv_atomic
from base.spl_csv_schema import overall_spl_csv_columns
from consts.acoustic_analysis.common_consts import SOUND_QUALITY_ANALYSES


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
    *,
    skip_overall=False,
    input_voltage=None,
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
        if skip_overall and config_item_name == "总体声压级":
            continue
        if config_item_name == "总体声压级" and input_voltage is not None:
            header = (*header, "电压")
            rows = [(*row, input_voltage) for row in rows]
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
    if analysis_type in SOUND_QUALITY_ANALYSES:
        spec = SOUND_QUALITY_ANALYSES[analysis_type]
        rows = []
        for output in outputs:
            header, row = sound_quality_summary_csv_row(analysis_type, config, output)
            rows.append((format_channel_name(int(output["raw_channel"]), channel_labels), *row))
        return (
            iter_curve_csv_table(spec["label"] + "时间曲线", "时间(s)", spec["label"] + spec["unit"],
                                  outputs, channel_labels, limit_unit=spec["unit"]),
            (spec["label"] + "汇总", ("通道", *header), rows),
        )
    if analysis_type == "SPL":
        return _spl_tables(config, outputs, channel_labels)
    if analysis_type == "LOUD":
        unit = outputs[0]["metrics"]["unit"]
        curve = iter_curve_csv_table(
            "响度时间曲线", "时间(s)", f"响度_{unit}", outputs, channel_labels,
            limit_unit=unit,
        )
        rows = []
        for output in outputs:
            header, row = loudness_summary_csv_row(config, output)
            rows.append((format_channel_name(int(output["raw_channel"]), channel_labels), *row))
        return curve, ("响度汇总", ("通道", *header), rows)
    if analysis_type == "FBA":
        return (
            iter_curve_csv_table(
                "频段能量",
                "X轴中心频率Hz",
                "频段声压级_Y轴dB",
                outputs,
                channel_labels,
            ),
        )
    if analysis_type == "FFT":
        return (
            iter_curve_csv_table(
                "FFT频谱",
                "X轴频率Hz",
                "FFT_Y轴dB",
                outputs,
                channel_labels,
            ),
        )
    return ()


def sound_quality_summary_csv_row(kind, config, result):
    spec = SOUND_QUALITY_ANALYSES[kind]
    metrics, unit = result["metrics"], spec["unit"]
    enabled = bool(config.get("limit_checked"))
    header = [label + unit for _, label in spec["metrics"]]
    header += ["有效帧数", "总帧数", "判定指标", f"下限{unit}", f"上限{unit}", "判定结果"]
    row = [_csv_number(metrics.get(key)) for key, _ in spec["metrics"]]
    row += [metrics.get("valid_frame_count"), metrics.get("frame_count"),
            config.get("limit_metric", "curve_y") if enabled else "",
            _csv_number(metrics.get("overall_lower_limit")),
            _csv_number(metrics.get("overall_upper_limit")),
            (result.get("judgement") or "判定失败") if enabled else "未启用"]
    return header, row


def loudness_summary_csv_row(config, result):
    """Share summary columns between station and legacy multichannel exports."""
    metrics = result["metrics"]
    unit = metrics["unit"]
    limit_unit = metrics.get("limit_unit", unit)
    header = [f"稳态平均响度{unit}", f"最大瞬态响度{unit}", "判定指标",
              f"下限{limit_unit}", f"上限{limit_unit}", "判定结果"]
    enabled = bool(config.get("limit_checked"))
    judgement = (result.get("judgement") or "判定失败") if enabled else "未启用"
    row = [metrics[f"steady_state_average_{unit}"], metrics[f"max_transient_{unit}"],
           metrics.get("limit_metric") if enabled else None,
           metrics.get("overall_lower_limit"), metrics.get("overall_upper_limit"), judgement]
    return header, row


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
            *overall_spl_csv_columns(config.get("weighting", "Z")),
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
        iter_curve_csv_table(
            "实时声压级",
            "X轴时间秒",
            "SPL_Y轴dB",
            outputs,
            channel_labels,
        )
    )
    return tuple(tables)


def _curve_values(curve, key):
    value = curve.get(key)
    return () if value is None else value


def iter_curve_csv_table(
    config_item_name,
    x_name,
    y_name,
    outputs,
    channel_labels,
    *,
    limit_unit="dB",
):
    """Validate columns once and format rows lazily without copying whole curves."""
    reference = dict(outputs[0].get("csv_curve") or {})
    x_values = _curve_values(reference, "x")
    lower = _curve_values(reference, "lower")
    upper = _curve_values(reference, "upper")
    reference_x = np.asarray(x_values, dtype=np.float64)
    header = [x_name]
    for output in outputs:
        header.append(
            f"{format_channel_name(int(output['raw_channel']), channel_labels)}_"
            f"{y_name}"
        )
    header.extend((f"下限_Y轴{limit_unit}", f"上限_Y轴{limit_unit}"))

    curves = []
    for output in outputs:
        curve = dict(output.get("csv_curve") or {})
        channel_x = np.asarray(_curve_values(curve, "x"), dtype=np.float64)
        channel_y = np.asarray(_curve_values(curve, "y"), dtype=np.float64)
        if channel_x.size != channel_y.size:
            raise ValueError("curve x/y lengths differ")
        if not np.array_equal(channel_x, reference_x):
            if not len(x_values) or channel_x.size == 0:
                raise ValueError("channel curves do not share an x axis")
            channel_y = np.interp(
                reference_x,
                channel_x,
                channel_y,
            )
        curves.append(channel_y)

    def rows():
        for index, x_value in enumerate(x_values):
            row = [_csv_number(x_value)]
            row.extend(_csv_number(curve[index]) for curve in curves)
            row.append(_csv_number(lower[index]) if len(lower) else "")
            row.append(_csv_number(upper[index]) if len(upper) else "")
            yield tuple(row)

    return config_item_name, tuple(header), rows()


def _csv_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.10g}"
