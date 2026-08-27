from __future__ import annotations

import logging
import math
import time
from typing import Any

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice


_NON_JUDGMENT_ANALYSIS_TYPES = {"SPEC"}
_CURVE_JUDGMENT_ANALYSIS_TYPES = {
    "SPL",
    "FFT",
    "FBA",
    "SPLF",
    "FR",
}
_REPORT_PLOT_LAYOUT_SIZE = (1000, 600)
_REPORT_PLOT_EXPORT_WIDTH = 1000
_REPORT_PLOT_SLOW_SECONDS = 0.5
_LOGGER = logging.getLogger(__name__)


def _format_deviation(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _finite_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_finite_values(item))
        return values
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if math.isfinite(number) else []


def _format_measurement(value: Any) -> str:
    values = _finite_values(value)
    if not values:
        return "-"
    return f"{values[0]:.6g}"


def _format_limit_range(value: Any) -> str:
    values = _finite_values(value)
    if not values:
        return "-"
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-12):
        return f"{lower:.6g}"
    return f"{lower:.6g} ~ {upper:.6g}"


def _configured_limit_value(
    item_config: dict[str, Any],
    side: str,
    result: dict[str, Any],
    *,
    preferred_prefixes: tuple[str, ...] | None = None,
) -> str:
    result_key = f"{side}_limits"
    if result_key in result:
        return _format_limit_range(result.get(result_key))
    if not item_config.get("limit_checked", False):
        return "-"

    for prefix in preferred_prefixes or ("constant", "scalar", "curve"):
        enabled_key = f"{prefix}_{side}_enabled"
        value_key = f"{prefix}_{side}_value"
        if item_config.get(enabled_key, False):
            return _format_measurement(item_config.get(value_key))

    limit_data = item_config.get("limit_data")
    if isinstance(limit_data, (list, tuple)) and len(limit_data) >= 3:
        limit_index = 1 if side == "upper" else 2
        return _format_limit_range(limit_data[limit_index])

    segment_keys = (
        f"manual_{side}_segments",
        f"{side}_segments",
    )
    if any(item_config.get(key) for key in segment_keys):
        return "曲线"
    if str(item_config.get("limit_mode") or "").lower() in ("csv", "manual"):
        return "曲线"
    return "-"


def _values_vary(value: Any) -> bool:
    values = _finite_values(value)
    if len(values) < 2:
        return False
    first = values[0]
    return any(
        not math.isclose(first, current, rel_tol=1e-9, abs_tol=1e-12)
        for current in values[1:]
    )


def _curve_limit_display(
    item_config: dict[str, Any],
    side: str,
    result: dict[str, Any],
    *,
    preferred_prefixes: tuple[str, ...] | None = None,
) -> str:
    display_value = _configured_limit_value(
        item_config,
        side,
        result,
        preferred_prefixes=preferred_prefixes,
    )
    if display_value == "-":
        return display_value

    result_limits = result.get(f"{side}_limits")
    if result_limits is not None:
        return "见分析图" if _values_vary(result_limits) else display_value

    limit_data = item_config.get("limit_data")
    if isinstance(limit_data, (list, tuple)) and len(limit_data) >= 3:
        limit_index = 1 if side == "upper" else 2
        if _values_vary(limit_data[limit_index]):
            return "见分析图"

    if any(
        item_config.get(key)
        for key in (f"manual_{side}_segments", f"{side}_segments")
    ):
        return "见分析图"
    if _values_vary(item_config.get(f"curve_{side}_value")):
        return "见分析图"
    return "见分析图" if display_value == "曲线" else display_value


def _analysis_measurement_fields(
    instance: Any,
    item_config: dict[str, Any],
) -> dict[str, str]:
    item_type = str(item_config.get("type") or "").upper()
    result = getattr(instance, "result", {}) or {}
    if not isinstance(result, dict):
        result = {}

    measurement = "-"
    unit = "-"
    limit_prefixes = None
    metric = ""
    if item_type == "SPL":
        values = _finite_values(result.get("overall_spl"))
        if not values:
            signal_values = _finite_values(result.get("signal_spl"))
            values = [max(signal_values)] if signal_values else []
        measurement = _format_measurement(values)
        unit_getter = getattr(instance, "_get_spl_unit", None)
        unit = str(unit_getter() if callable(unit_getter) else "dB SPL")
    elif item_type == "FFT":
        spectrum_values = _finite_values(
            result.get("plot_db", result.get("fft_db"))
        )
        measurement = _format_measurement(
            [max(spectrum_values)] if spectrum_values else []
        )
        weighting = str(
            result.get("weighting")
            or item_config.get("weighting")
            or "Z"
        )
        unit = (
            "dB"
            if str(result.get("display_mode") or "") == "delta"
            else ("dB SPL" if weighting == "Z" else f"dB({weighting}) SPL")
        )
    elif item_type == "FBA":
        weighting = str(
            result.get("weighting")
            or item_config.get("weighting")
            or "Z"
        )
        measurement = _format_measurement(
            result.get("overall_weighted_db", result.get("overall_db"))
        )
        unit = "dB" if weighting == "Z" else f"dB({weighting})"
    elif item_type in ("THD", "PRB"):
        curve_values = _finite_values(result.get("thd"))
        measurement = _format_measurement(
            [max(curve_values)] if curve_values else []
        )
        unit = "phon" if item_type == "PRB" else "%"
    elif item_type == "SPLF":
        curve_values = _finite_values(result.get("spl_db"))
        measurement = _format_measurement(
            [max(curve_values)] if curve_values else []
        )
        unit = "dB SPL"
    elif item_type == "FR":
        curve_values = _finite_values(result.get("fr"))
        measurement = _format_measurement(
            [max(curve_values)] if curve_values else []
        )
        unit = "dB"
    elif item_type == "SPEC":
        unit = "dB"
    elif item_type == "LOUD":
        summary = result.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        advanced_config = item_config.get("advanced") or {}
        if not isinstance(advanced_config, dict):
            advanced_config = {}
        metric = str(
            item_config.get("limit_metric") or "curve_y"
        ).lower()
        limit_prefixes = (
            ("constant", "curve")
            if metric == "curve_y"
            else ("scalar",)
        )
        unit = str(
            item_config.get("curve_limit_unit")
            or advanced_config.get("curve_y_unit")
            or "sone"
        ).lower()
        if metric == "steady_state_average":
            measurement = _format_measurement(
                summary.get(f"steady_state_average_{unit}")
            )
        elif metric == "max_transient":
            measurement = _format_measurement(
                summary.get(f"max_transient_{unit}")
            )
        elif metric == "specific_loudness_summed_exceedance":
            measurement = _format_measurement(
                summary.get("specific_loudness_summed_exceedance")
            )
            unit = "sone"
        else:
            loudness_values = _finite_values(
                result.get(
                    "loudness_level_phon"
                    if unit == "phon"
                    else "loudness_sone"
                )
            )
            measurement = _format_measurement(
                [max(loudness_values)] if loudness_values else []
            )

    uses_curve_judgment = bool(item_config.get("limit_checked", False)) and (
        item_type in _CURVE_JUDGMENT_ANALYSIS_TYPES
        or (item_type == "LOUD" and metric == "curve_y")
    )
    if uses_curve_judgment:
        return {
            "measurement": "曲线判定",
            "lower_limit": _curve_limit_display(
                item_config,
                "lower",
                result,
                preferred_prefixes=limit_prefixes,
            ),
            "upper_limit": _curve_limit_display(
                item_config,
                "upper",
                result,
                preferred_prefixes=limit_prefixes,
            ),
            "unit": unit,
        }

    return {
        "measurement": measurement,
        "lower_limit": _configured_limit_value(
            item_config,
            "lower",
            result,
            preferred_prefixes=limit_prefixes,
        ),
        "upper_limit": _configured_limit_value(
            item_config,
            "upper",
            result,
            preferred_prefixes=limit_prefixes,
        ),
        "unit": unit,
    }


def _analysis_judgment(
    instance: Any,
    item_key: str,
    item_config: dict[str, Any],
    analysis_result_dict: dict[str, Any],
) -> tuple[str, str]:
    result = None
    title_name = str(getattr(instance, "title_name", "") or "")
    for candidate in (title_name, item_key):
        if candidate and candidate in analysis_result_dict:
            result = analysis_result_dict[candidate]
            break

    if isinstance(result, (tuple, list)) and result:
        if result[0] is not None:
            status = "OK" if bool(result[0]) else "NG"
            deviation = _format_deviation(result[1] if len(result) > 1 else None)
            return status, deviation
        return "未产生判定", _format_deviation(result[1] if len(result) > 1 else None)

    if item_config.get("limit_checked") is False:
        return "未启用判定", "-"
    item_type = str(item_config.get("type") or "").upper()
    if item_type in _NON_JUDGMENT_ANALYSIS_TYPES:
        return "仅图表分析", "-"
    return "未产生判定", "-"


def _plot_title(plot_widget: pg.PlotWidget, fallback: str) -> str:
    plot_item = plot_widget.getPlotItem()
    title_label = getattr(plot_item, "titleLabel", None)
    title = str(getattr(title_label, "text", "") or "").strip()
    return title or fallback


def _report_plot_widgets(instance: Any) -> list[pg.PlotWidget]:
    widgets: list[pg.PlotWidget] = []
    explicit_provider = getattr(instance, "get_report_plot_widgets", None)
    if not callable(explicit_provider):
        return widgets
    for widget in explicit_provider() or []:
        if isinstance(widget, pg.PlotWidget):
            widgets.append(widget)

    unique_widgets = []
    seen_ids = set()
    for widget in widgets:
        widget_id = id(widget)
        if widget_id in seen_ids:
            continue
        seen_ids.add(widget_id)
        unique_widgets.append(widget)
    return unique_widgets


def export_plot_widget_png(
    plot_widget: pg.PlotWidget,
    width: int = _REPORT_PLOT_EXPORT_WIDTH,
) -> bytes:
    started_at = time.perf_counter()
    plot_item = plot_widget.getPlotItem()
    title_label = getattr(plot_item, "titleLabel", None)
    title_text = str(getattr(title_label, "text", "") or "")
    title_options = dict(getattr(title_label, "opts", {}) or {})
    title_visible = bool(title_label is not None and title_label.isVisible())
    original_size = plot_widget.size()

    try:
        if title_visible:
            plot_item.setTitle(None)
        plot_widget.resize(*_REPORT_PLOT_LAYOUT_SIZE)
        plot_widget.ensurePolished()
        plot_item.layout.activate()

        exporter = pg.exporters.ImageExporter(plot_item)
        exporter.parameters()["width"] = int(width)
        image = exporter.export(toBytes=True)
    finally:
        plot_widget.resize(original_size)
        if title_visible:
            plot_item.setTitle(title_text, **title_options)
        else:
            plot_item.setTitle(None)
        plot_item.layout.activate()

    if image is None or image.isNull():
        raise RuntimeError("分析图导出结果为空")

    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    if not buffer.open(QIODevice.WriteOnly):
        raise RuntimeError("分析图内存缓冲区创建失败")
    try:
        if not image.save(buffer, "PNG"):
            raise RuntimeError("分析图 PNG 编码失败")
    finally:
        buffer.close()
    png_data = bytes(byte_array)
    elapsed_seconds = time.perf_counter() - started_at
    if elapsed_seconds >= _REPORT_PLOT_SLOW_SECONDS:
        _LOGGER.warning(
            "product_pdf_plot_snapshot_slow: title=%r width=%d elapsed_ms=%.1f",
            title_text,
            int(width),
            elapsed_seconds * 1000.0,
        )
    return png_data


def build_analysis_report_items(
    analysis_instances: list[Any],
    analysis_config: dict[str, Any],
    analysis_result_dict: dict[str, Any],
    preflight_skips: Any = None,
) -> list[dict[str, Any]]:
    report_items = []
    analysis_config = analysis_config if isinstance(analysis_config, dict) else {}
    analysis_result_dict = (
        analysis_result_dict if isinstance(analysis_result_dict, dict) else {}
    )

    for instance in analysis_instances or []:
        item_key = str(getattr(instance, "_sequence_analysis_key", "") or "").strip()
        if not item_key:
            continue
        item_config = analysis_config.get(item_key)
        if not isinstance(item_config, dict):
            item_config = {}

        state = str(
            getattr(instance, "_product_report_analysis_state", "completed")
            or "completed"
        )
        error = str(
            getattr(instance, "_product_report_analysis_error", "") or ""
        ).strip()
        if state == "failed":
            status, deviation = "分析失败", "-"
        else:
            status, deviation = _analysis_judgment(
                instance,
                item_key,
                item_config,
                analysis_result_dict,
            )
        measurement_fields = _analysis_measurement_fields(
            instance,
            item_config,
        )

        images = []
        image_errors = []
        for index, plot_widget in enumerate(_report_plot_widgets(instance)):
            caption = _plot_title(
                plot_widget,
                item_key if index == 0 else f"{item_key} - 图 {index + 1}",
            )
            try:
                png_data = export_plot_widget_png(plot_widget)
            except Exception as exc:
                image_errors.append(f"{caption}: {exc}")
                continue
            images.append({"caption": caption, "png_data": png_data})

        report_items.append(
            {
                "name": item_key,
                "type": str(item_config.get("type") or ""),
                "state": state,
                "status": status,
                "deviation": deviation,
                **measurement_fields,
                "error": error,
                "image_errors": image_errors,
                "images": images,
            }
        )
    reported_keys = {item["name"] for item in report_items}
    if isinstance(preflight_skips, dict):
        skipped_values = preflight_skips.values()
    else:
        skipped_values = preflight_skips or ()
    for skip in skipped_values:
        if isinstance(skip, dict):
            item_key = str(skip.get("item_key") or "").strip()
            item_type = str(skip.get("item_type") or "")
            reason = str(skip.get("reason") or "").strip()
        else:
            item_key = str(getattr(skip, "item_key", "") or "").strip()
            item_type = str(getattr(skip, "item_type", "") or "")
            reason = str(getattr(skip, "reason", "") or "").strip()
        if not item_key or item_key in reported_keys:
            continue
        reported_keys.add(item_key)
        report_items.append(
            {
                "name": item_key,
                "type": item_type,
                "state": "skipped",
                "reason": reason,
                "error": "",
                "image_errors": [],
                "images": [],
            }
        )
    return report_items
