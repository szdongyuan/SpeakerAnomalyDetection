import math
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ui.ui_analysis_config.threshold_csv_manual import (
    ThresholdCsvManualError,
    csv_rows_from_manual_config,
    load_threshold_csv,
    manual_config_from_limit_data,
    manual_config_has_complete_segments,
    write_manual_config_csv,
)


@pytest.fixture
def local_tmp_path(request):
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in request.node.name)
    path = REPO_ROOT / ".tmp" / "threshold_csv_manual" / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        for parent in (path.parent, path.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                pass


def _nan_list(length):
    return [math.nan] * length


def _upper_segment(start_x, start_y, end_x, end_y):
    return {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}


def _config(upper_segments=None, lower_segments=None):
    upper_segments = upper_segments or []
    lower_segments = lower_segments or []
    return {
        "manual_upper_enabled": bool(upper_segments),
        "manual_lower_enabled": bool(lower_segments),
        "manual_upper_segments": upper_segments,
        "manual_lower_segments": lower_segments,
    }


def _roundtrip_manual_config(tmp_path, config):
    path = tmp_path / "roundtrip.csv"
    write_manual_config_csv(config, str(path))
    return manual_config_from_limit_data(load_threshold_csv(str(path)))


def test_limit_data_to_manual_segments_handles_duplicate_x_discontinuity():
    config = manual_config_from_limit_data(
        ([0.0, 1.0, 1.0, 2.0], [10.0, 20.0, 30.0, 40.0], _nan_list(4))
    )

    assert config == _config(
        [
            _upper_segment(0.0, 10.0, 1.0, 20.0),
            _upper_segment(1.0, 30.0, 2.0, 40.0),
        ]
    )


def test_limit_data_to_manual_segments_repeated_equal_duplicate_rows_collapse():
    config = manual_config_from_limit_data(
        ([0.0, 1.0, 1.0, 2.0], [10.0, 20.0, 20.0, 40.0], _nan_list(4))
    )

    assert config == _config(
        [
            _upper_segment(0.0, 10.0, 1.0, 20.0),
            _upper_segment(1.0, 20.0, 2.0, 40.0),
        ]
    )


@pytest.mark.parametrize(
    "limit_data",
    [
        ([1.0, 1.0, 2.0], [20.0, 30.0, 40.0], _nan_list(3)),
        ([0.0, 1.0, 1.0], [10.0, 20.0, 30.0], _nan_list(3)),
    ],
)
def test_limit_data_to_manual_segments_rejects_boundary_duplicate_without_neighbors(limit_data):
    with pytest.raises(ThresholdCsvManualError, match="重复X"):
        manual_config_from_limit_data(limit_data)


@pytest.mark.parametrize(
    "values",
    [
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [10.0, 20.0, 30.0, 20.0, 40.0],
    ],
)
def test_limit_data_to_manual_segments_rejects_multiple_jumps_at_one_x(values):
    with pytest.raises(ThresholdCsvManualError, match="重复X"):
        manual_config_from_limit_data(([0.0, 1.0, 1.0, 1.0, 2.0], values, _nan_list(5)))


def test_limit_data_to_manual_segments_ignores_blank_cells_per_side():
    config = manual_config_from_limit_data(
        (
            [0.0, 1.0, 1.0, 2.0],
            [10.0, 20.0, 30.0, 40.0],
            [math.nan, 5.0, math.nan, 6.0],
        )
    )

    assert config == {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            _upper_segment(0.0, 10.0, 1.0, 20.0),
            _upper_segment(1.0, 30.0, 2.0, 40.0),
        ],
        "manual_lower_segments": [_upper_segment(1.0, 5.0, 2.0, 6.0)],
    }


@pytest.mark.parametrize("limit_data", [(None, [], []), (1, 2, 3)])
def test_limit_data_to_manual_segments_rejects_non_iterable_tuple_members(limit_data):
    with pytest.raises(ThresholdCsvManualError, match="CSV阈值数据.*格式不正确"):
        manual_config_from_limit_data(limit_data)


def test_load_threshold_csv_accepts_existing_two_column_formats(local_tmp_path):
    upper_path = local_tmp_path / "upper.csv"
    lower_path = local_tmp_path / "lower.csv"
    upper_path.write_text("x,upperbound\n0,10\n1,20\n", encoding="utf-8")
    lower_path.write_text("x,lowerbound\n0,1\n1,2\n", encoding="utf-8")

    x_values, upper, lower = load_threshold_csv(str(upper_path))
    assert x_values == [0.0, 1.0]
    assert upper == [10.0, 20.0]
    assert all(math.isnan(value) for value in lower)

    x_values, upper, lower = load_threshold_csv(str(lower_path))
    assert x_values == [0.0, 1.0]
    assert all(math.isnan(value) for value in upper)
    assert lower == [1.0, 2.0]


def test_load_threshold_csv_accepts_hd_style_x_header(local_tmp_path):
    path = local_tmp_path / "hd_limit_3.csv"
    path.write_text(
        "THD_x,upperbound\n"
        "100,0.2\n"
        "199,0.2\n"
        "200,0.1\n"
        "309,0.1\n"
        "310,0.4\n"
        "2000,0.4\n",
        encoding="utf-8",
    )

    x_values, upper, lower = load_threshold_csv(str(path))

    assert x_values == [100.0, 199.0, 200.0, 309.0, 310.0, 2000.0]
    assert upper == [0.2, 0.2, 0.1, 0.1, 0.4, 0.4]
    assert all(math.isnan(value) for value in lower)


def test_load_threshold_csv_rejects_blank_x_header(local_tmp_path):
    path = local_tmp_path / "blank_x_header.csv"
    path.write_text(",upperbound\n0,10\n", encoding="utf-8")

    with pytest.raises(ThresholdCsvManualError, match="格式"):
        load_threshold_csv(str(path))


def test_load_threshold_csv_accepts_existing_three_column_lower_upper_order(local_tmp_path):
    path = local_tmp_path / "limit.csv"
    path.write_text("x,lowerbound,upperbound\n0,1,10\n1,2,20\n", encoding="utf-8")

    assert load_threshold_csv(str(path)) == ([0.0, 1.0], [10.0, 20.0], [1.0, 2.0])


def test_load_threshold_csv_accepts_exported_blank_cell_three_column_format(local_tmp_path):
    path = local_tmp_path / "limit.csv"
    path.write_text("x,upperbound,lowerbound\n0,10,\n1,20,\n1,30,5\n2,40,6\n", encoding="utf-8")

    x_values, upper, lower = load_threshold_csv(str(path))

    assert x_values == [0.0, 1.0, 1.0, 2.0]
    assert upper == [10.0, 20.0, 30.0, 40.0]
    assert math.isnan(lower[0])
    assert math.isnan(lower[1])
    assert lower[2:] == [5.0, 6.0]


def test_load_threshold_csv_rejects_rows_with_no_threshold_value(local_tmp_path):
    path = local_tmp_path / "limit.csv"
    path.write_text("x,upperbound,lowerbound\n0,,\n", encoding="utf-8")

    with pytest.raises(ThresholdCsvManualError, match="至少"):
        load_threshold_csv(str(path))


def test_load_threshold_csv_rejects_unique_row_lower_above_upper(local_tmp_path):
    path = local_tmp_path / "limit.csv"
    path.write_text("x,upperbound,lowerbound\n0,10,11\n", encoding="utf-8")

    with pytest.raises(ThresholdCsvManualError, match="下限不能大于上限"):
        load_threshold_csv(str(path))


def test_load_threshold_csv_defers_duplicate_x_lower_upper_validation_until_conversion(local_tmp_path):
    path = local_tmp_path / "limit.csv"
    path.write_text("x,upperbound,lowerbound\n0,10,\n1,20,25\n1,30,\n2,40,26\n", encoding="utf-8")

    config = manual_config_from_limit_data(load_threshold_csv(str(path)))

    assert config == {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            _upper_segment(0.0, 10.0, 1.0, 20.0),
            _upper_segment(1.0, 30.0, 2.0, 40.0),
        ],
        "manual_lower_segments": [_upper_segment(1.0, 25.0, 2.0, 26.0)],
    }


def test_csv_export_upper_and_lower_one_sided_headers():
    assert csv_rows_from_manual_config(
        _config([_upper_segment(0.0, 10.0, 1.0, 20.0)])
    ) == [
        ["x", "upperbound"],
        ["0.0", "10.0"],
        ["1.0", "20.0"],
    ]

    assert csv_rows_from_manual_config(
        _config(lower_segments=[_upper_segment(0.0, 1.0, 1.0, 2.0)])
    ) == [
        ["x", "lowerbound"],
        ["0.0", "1.0"],
        ["1.0", "2.0"],
    ]


def test_csv_export_two_sided_uses_sorted_x_union_and_blank_missing_cells():
    config = _config(
        [_upper_segment(0.0, 10.0, 1.0, 20.0)],
        [_upper_segment(0.5, 5.0, 1.5, 6.0)],
    )

    assert csv_rows_from_manual_config(config) == [
        ["x", "upperbound", "lowerbound"],
        ["0.0", "10.0", ""],
        ["0.5", "", "5.0"],
        ["1.0", "20.0", ""],
        ["1.5", "", "6.0"],
    ]


def test_csv_export_two_sided_orientation_pairs_start_only_endpoint():
    config = _config(
        [
            _upper_segment(0.0, 10.0, 1.0, 20.0),
            _upper_segment(1.0, 30.0, 2.0, 40.0),
        ],
        [_upper_segment(1.0, 5.0, 2.0, 6.0)],
    )

    assert csv_rows_from_manual_config(config) == [
        ["x", "upperbound", "lowerbound"],
        ["0.0", "10.0", ""],
        ["1.0", "20.0", ""],
        ["1.0", "30.0", "5.0"],
        ["2.0", "40.0", "6.0"],
    ]


def test_csv_export_rejects_invalid_manual_config():
    with pytest.raises(ThresholdCsvManualError, match="下限不能大于上限"):
        csv_rows_from_manual_config(
            _config(
                [_upper_segment(0.0, 10.0, 1.0, 10.0)],
                [_upper_segment(0.0, 11.0, 1.0, 11.0)],
            )
        )


@pytest.mark.parametrize(
    "config",
    [
        _config(
            [
                _upper_segment(0.0, 10.0, 1.0, 20.0),
                _upper_segment(1.0, 30.0, 2.0, 40.0),
            ],
            [
                _upper_segment(0.0, 1.0, 1.0, 2.0),
                _upper_segment(1.0, 3.0, 2.0, 4.0),
            ],
        ),
        _config(
            [
                _upper_segment(0.0, 10.0, 1.0, 20.0),
                _upper_segment(1.0, 30.0, 2.0, 40.0),
            ],
            [
                _upper_segment(0.0, 1.0, 1.0, 2.0),
                _upper_segment(1.0, 2.0, 2.0, 4.0),
            ],
        ),
        _config(
            [
                _upper_segment(0.0, 10.0, 1.0, 20.0),
                _upper_segment(1.0, 30.0, 2.0, 40.0),
            ],
            [_upper_segment(1.5, 5.0, 2.0, 6.0)],
        ),
        _config(
            [
                _upper_segment(0.0, 10.0, 1.0, 20.0),
                _upper_segment(1.0, 30.0, 2.0, 40.0),
            ],
            [_upper_segment(1.0, 5.0, 2.0, 6.0)],
        ),
        _config(
            [
                _upper_segment(0.0, 10.0, 1.0, 20.0),
                _upper_segment(1.0, 30.0, 2.0, 40.0),
            ],
            [_upper_segment(0.0, 5.0, 1.0, 6.0)],
        ),
    ],
)
def test_two_sided_export_import_roundtrips_discontinuous_orientation_cases(local_tmp_path, config):
    assert _roundtrip_manual_config(local_tmp_path, config) == config


def test_manual_config_has_complete_segments_checks_enabled_sides():
    assert manual_config_has_complete_segments(_config([_upper_segment(0.0, 1.0, 1.0, 2.0)]))
    assert manual_config_has_complete_segments(_config(lower_segments=[_upper_segment(0.0, 1.0, 1.0, 2.0)]))
    assert not manual_config_has_complete_segments(_config())
    assert not manual_config_has_complete_segments(
        {
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": [{"start_x": 0.0, "start_y": 1.0}],
            "manual_lower_segments": [],
        }
    )
