"""Render analysis artifacts without importing Qt or analysis window classes."""

from __future__ import annotations

from io import BytesIO

import numpy as np

from consts.acoustic_analysis.curve_style_consts import (
    LOWER_LIMIT_COLOR,
    MAIN_CURVE_COLOR,
    UPPER_LIMIT_COLOR,
)


def render_analysis_png(plot_payload):
    """Return a PNG byte string for one internal calculation plot payload."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    from matplotlib import pyplot as plt

    payload = dict(plot_payload or {})
    kind = payload.get("kind")
    figure, axis = plt.subplots(figsize=(12, 8), dpi=100)
    try:
        if kind == "curve":
            _render_curve(axis, payload)
        elif kind == "bar":
            _render_bar(axis, payload)
        elif kind == "spectrogram":
            _render_spectrogram(figure, axis, payload)
        elif kind == "values":
            _render_values(axis, payload)
        else:
            raise ValueError(f"不支持的图片类型：{kind}")
        figure.tight_layout()
        output = BytesIO()
        figure.savefig(output, format="png", facecolor="white")
        return output.getvalue()
    finally:
        plt.close(figure)


def _render_curve(axis, payload):
    x_values = _array_value(payload.get("x"))
    y_values = _array_value(payload.get("y"))
    if x_values.size != y_values.size or x_values.size == 0:
        raise ValueError("曲线没有可绘制的数据")
    colors = dict(payload.get("colors") or {})
    main_color = colors.get(MAIN_CURVE_COLOR, "#33C44D")
    axis.plot(x_values, y_values, color=main_color, linewidth=1.2, label="Result")
    baseline = _optional_same_length(payload.get("baseline"), x_values)
    if baseline is not None:
        axis.plot(
            x_values,
            baseline,
            color="#808080",
            linewidth=1.0,
            label="Baseline",
        )
    lower = _optional_same_length(payload.get("lower"), x_values)
    upper = _optional_same_length(payload.get("upper"), x_values)
    if lower is not None:
        axis.plot(
            x_values,
            lower,
            color=colors.get(LOWER_LIMIT_COLOR, "#800080"),
            linestyle="--",
            linewidth=1.0,
            label="Lower limit",
        )
    if upper is not None:
        axis.plot(
            x_values,
            upper,
            color=colors.get(UPPER_LIMIT_COLOR, "#800080"),
            linestyle="--",
            linewidth=1.0,
            label="Upper limit",
        )
    out_mask = _array_value(payload.get("out_mask"), dtype=bool)
    if out_mask.size == x_values.size and np.any(out_mask):
        axis.scatter(
            x_values[out_mask],
            y_values[out_mask],
            s=4,
            color="#F44336",
            label="Out of limit",
        )
    if payload.get("log_x"):
        axis.set_xscale("log")
    _finish_axis(axis, payload)


def _render_bar(axis, payload):
    y_values = _array_value(payload.get("y"))
    if y_values.size == 0:
        raise ValueError("频段结果没有可绘制的数据")
    positions = np.arange(y_values.size)
    colors = dict(payload.get("colors") or {})
    bar_colors = np.full(y_values.size, colors.get(MAIN_CURVE_COLOR, "#33C44D"), dtype=object)
    out_mask = _array_value(payload.get("out_mask"), dtype=bool)
    if out_mask.size == y_values.size:
        bar_colors[out_mask] = "#F44336"
    finite = np.isfinite(y_values)
    bar_colors[~finite] = "#BDBDBD"
    axis.bar(positions, np.where(finite, y_values, 0.0), color=bar_colors)
    labels = list(payload.get("labels") or [])
    if len(labels) == y_values.size:
        step = max(1, int(np.ceil(len(labels) / 20.0)))
        shown = positions[::step]
        axis.set_xticks(shown)
        axis.set_xticklabels(
            [labels[index] for index in shown],
            rotation=45,
            ha="right",
        )
    for field, color_key, label in (
        ("lower", LOWER_LIMIT_COLOR, "Lower limit"),
        ("upper", UPPER_LIMIT_COLOR, "Upper limit"),
    ):
        limits = _optional_same_length(payload.get(field), y_values)
        if limits is not None:
            axis.plot(
                positions,
                limits,
                color=colors.get(color_key, "#800080"),
                linestyle="--",
                linewidth=1.0,
                label=label,
            )
    _finish_axis(axis, payload)


def _render_spectrogram(figure, axis, payload):
    x_values = _array_value(payload.get("x"))
    y_values = _array_value(payload.get("y"))
    z_values = np.asarray(payload.get("z"), dtype=np.float32)
    if (
        x_values.size == 0
        or y_values.size == 0
        or z_values.shape != (y_values.size, x_values.size)
    ):
        raise ValueError("频谱图数据维度不一致")
    levels = payload.get("levels")
    image_kwargs = {}
    if isinstance(levels, (tuple, list)) and len(levels) == 2:
        image_kwargs.update(vmin=float(levels[0]), vmax=float(levels[1]))
    image = axis.imshow(
        z_values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
        cmap=str(payload.get("color_map", "viridis") or "viridis"),
        **image_kwargs,
    )
    figure.colorbar(image, ax=axis, label="dB")
    _finish_axis(axis, payload)


def _render_values(axis, payload):
    axis.axis("off")
    title = str(payload.get("title") or "分析结果")
    axis.set_title(title)
    values = dict(payload.get("values") or {})
    lines = [f"{name}: {value if value is not None else '--'}" for name, value in values.items()]
    axis.text(
        0.5,
        0.5,
        "\n\n".join(lines),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=18,
    )


def _finish_axis(axis, payload):
    axis.set_xlabel(str(payload.get("x_label") or ""))
    axis.set_ylabel(str(payload.get("y_label") or ""))
    # The containing result window and artifact name already identify the item.
    # Omitting an inner plot title keeps every graphical analysis type consistent.
    axis.grid(True, alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best")


def _optional_same_length(raw_values, reference):
    if raw_values is None:
        return None
    values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    if values.size != reference.size:
        raise ValueError("绘图辅助曲线长度不一致")
    return values


def _array_value(value, *, dtype=np.float64):
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)
