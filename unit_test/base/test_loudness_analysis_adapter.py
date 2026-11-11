"""Manual analysis, rendering and CSV contract for the loudness entry."""

import base64
import csv
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis_artifact_paths import AnalysisStorageContext
from base.analysis_csv_exporter import export_item_csvs
from base.analysis_plot_renderer import render_analysis_png
from base.analysis_worker import _contributes_to_final, _extract_output


def calculate(config):
    rate = 48000
    tone = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(rate // 2) / rate)
    return calculate_analysis_instance(
        "LOUD", tone, rate, {"method": "per_segment", **config}, 2.0,
        source="手动查看", sequence_snapshot={},
    )


def test_manual_loudness_image_and_csv_use_phon_and_full_time_curve(tmp_path):
    config = {
        "advanced": {"curve_y_unit": "phon", "show_specific_loudness_heatmap": True},
        "display": {"summary_metrics": ["steady_state_average_loudness"],
                    "curves": ["loudness_time", "specific_loudness_profile"],
                    "heatmaps": ["specific_loudness"]},
        "limit_checked": True, "curve_limit_unit": "phon", "limit_metric": "curve_y",
        "curve_upper_enabled": True, "curve_upper_value": 40,
    }
    before = deepcopy(config)
    calculation = calculate(config)
    png = render_analysis_png(calculation["plot"])
    assert png.startswith(b"\x89PNG")
    assert calculation["judgement"] == "NG"
    assert _contributes_to_final("LOUD", config)
    output = _extract_output(SimpleNamespace(analysis_type="LOUD"), calculation, "NG", "手动查看", png)
    output.update(raw_channel=0, judgement="NG")
    assert base64.b64decode(output["display_payload"]["png_base64"]) == png
    assert output["csv_curve"]["y"] == calculation["loudness_curves"]["phon"].tolist()
    assert calculation["plot"]["curves"][0]["y_label"] == "Loudness (phon)"
    storage = AnalysisStorageContext(str(tmp_path), "项目", "型号", "样本", 1, "端口", "工况", datetime(2026, 11, 11))
    records = export_item_csvs(storage, "source", "响度", "LOUD", config, [output])
    assert records and all(record.ok for record in records)
    with open(records[0].file_path, encoding="utf-8-sig", newline="") as stream:
        header, *rows = csv.reader(stream)
    assert header[-2:] == ["下限_Y轴phon", "上限_Y轴phon"]
    assert len(rows) == len(calculation["curve"]["x"])
    assert config == before


def test_hidden_display_can_still_render_and_judge_scalar():
    result = calculate({
        "display": {"curves": [], "summary_metrics": []},
        "limit_checked": True, "limit_metric": "max_transient",
        "scalar_upper_enabled": True, "scalar_upper_value": 1000,
        "scalar_lower_enabled": False,
    })
    assert result["judgement"] == "OK"
    assert render_analysis_png(result["plot"]).startswith(b"\x89PNG")


def test_csv_threshold_only_applies_where_time_ranges_overlap():
    result = calculate({
        "limit_checked": True, "limit_mode": "csv", "limit_metric": "curve_y",
        "limit_data": [[0.2, 0.35], [0.01, 0.01], [None, None]],
    })
    assert result["judgement"] == "NG"
    time = result["curve"]["x"]
    assert np.isnan(result["curve"]["upper"][time < 0.2]).all()
    assert np.isnan(result["curve"]["upper"][time > 0.35]).all()


def test_nonoverlapping_threshold_is_an_error_not_ok():
    with pytest.raises(ValueError, match="没有重叠"):
        calculate({"limit_checked": True, "limit_mode": "csv", "limit_metric": "curve_y",
                   "limit_data": [[10, 11], [1, 1], [None, None]]})
