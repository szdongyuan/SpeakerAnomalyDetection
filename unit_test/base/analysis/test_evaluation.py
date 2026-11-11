import numpy as np
import pytest

from base.analysis.contracts import AnalysisInstance
from base.analysis.evaluation import evaluate_instance


def evaluate(kind, x, y, upper, lower):
    instance = AnalysisInstance("item", kind, 0, {
        "limit_mode": "csv", "limit_data": [x, upper, lower],
    })
    calculation = {"curve": {"x": np.array(x), "y": np.array(y)}}
    return evaluate_instance(instance, calculation)


def test_spl_gaps_split_runs_and_keep_true_maximum():
    result = evaluate("SPL", list(range(7)), [12, 14, 13, 20, 15, 9, 2],
                      [10, 10, 10, None, 10, 10, 10], [0, 0, 0, 0, 0, 0, 5])
    assert result["judgement"] == "NG"
    records = result["violations"]
    assert len(records) == 3
    assert records[0] == {
        "side": "above_upper", "x_start": 0.0, "x_end": 2.0, "x_unit": "s",
        "max_deviation_at": 1.0, "measured_value": 14.0, "limit_value": 10.0,
        "max_deviation": 4.0, "deviation_unit": "dB",
    }
    assert records[1]["x_start"] == 4
    assert records[2]["side"] == "below_lower"
    assert records[2]["max_deviation"] == 3


def test_fba_bands_are_not_merged_into_continuous_frequency_interval():
    result = evaluate("FBA", [100, 200, 400], [20, 30, 10], [15] * 3, [None] * 3)
    assert [(r["x_start"], r["x_end"]) for r in result["violations"]] == [(100, 100), (200, 200)]


def test_no_threshold_overlap_is_not_ok():
    instance = AnalysisInstance("item", "FFT", 0, {
        "limit_mode": "csv", "limit_data": [[10, 20], [100, 100], [None, None]],
    })
    calculation = {"curve": {"x": np.array([100, 200]), "y": np.array([50, 60])}}
    with pytest.raises(ValueError, match="没有重叠"):
        evaluate_instance(instance, calculation)


def test_missing_measurement_splits_violation_run():
    result = evaluate("SPL", [0, 1, 2], [20, np.nan, 30], [10] * 3, [None] * 3)
    assert len(result["violations"]) == 2


def test_scalar_limits_ignore_inactive_csv_configuration():
    instance = AnalysisInstance("SPL", "SPL", 0, {
        "limit_metric": "overall_spl", "limit_data": [],
        "scalar_upper_enabled": True, "scalar_upper_value": 100.0,
    })
    result = evaluate_instance(instance, {"metrics": {"overall_spl": 80.0}})
    assert result["judgement"] == "OK"
