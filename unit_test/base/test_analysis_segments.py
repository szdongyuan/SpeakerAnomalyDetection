import pytest

from base.analysis_segments import build_segment_plan, normalize_segmented_analysis


@pytest.mark.parametrize("settings", [
    {"mode": "output_load", "load_values": list(range(10)), "analysis_seconds": 10},
    {"mode": "time", "interval_seconds": 60, "analysis_seconds": 10},
])
def test_windows_are_centered_in_recording_relative_segments(settings):
    plan = build_segment_plan(settings, 600, 44100)
    assert len(plan) == 10
    assert [(s.window_start_sample / s.sample_rate, s.window_end_sample / s.sample_rate)
            for s in plan[:2]] == [(25, 35), (85, 95)]
    assert plan[-1].end_sample == 600 * 44100
    assert plan[0].available(35 * 44100)
    assert not plan[1].available(35 * 44100)


def test_close_load_values_keep_distinct_labels_and_repeated_values_keep_indices():
    loads = [0.12345678, 0.12345679, 0.12345678]
    plan = build_segment_plan({"mode": "output_load", "load_values": loads,
                               "load_unit": "A", "analysis_seconds": 1}, 60, 48000)
    assert [s.label for s in plan] == [
        "输出负载0.12345678A_第1段", "输出负载0.12345679A", "输出负载0.12345678A_第3段",
    ]
    assert [s.value for s in plan] == loads


def test_fractional_time_segment_labels_preserve_interval_precision():
    plan = build_segment_plan({"mode": "time", "interval_seconds": 0.12345678,
                               "display_time_unit": "s", "analysis_seconds": 0.01},
                              0.12345678 * 2, 48000)
    assert [s.label for s in plan] == ["时间0.12345678s", "时间0.24691356s"]


def test_decimal_time_intervals_do_not_add_float_artifacts_to_labels():
    plan = build_segment_plan({"mode": "time", "interval_seconds": 0.1,
                               "display_time_unit": "s", "analysis_seconds": 0.01},
                              0.3, 48000)
    assert [s.label for s in plan] == ["时间0.1s", "时间0.2s", "时间0.3s"]


@pytest.mark.parametrize("mode", ["time", "output_load"])
@pytest.mark.parametrize("total,interval,count", [(10.1, 0.1, 101), (0.6, 0.2, 3)])
@pytest.mark.parametrize("sample_rate", [1000, 48000])
def test_decimal_full_segment_window_is_allowed(mode, total, interval, count, sample_rate):
    settings = {"mode": mode, "analysis_seconds": interval}
    if mode == "time":
        settings["interval_seconds"] = interval
    else:
        settings["load_values"] = list(range(count))

    plan = build_segment_plan(settings, total, sample_rate)

    assert len(plan) == count
    assert all(segment.window_start_sample == segment.start_sample for segment in plan)
    assert all(segment.window_end_sample == segment.end_sample for segment in plan)
    assert plan[-1].end_sample == round(total * sample_rate)
    assert settings["analysis_seconds"] == interval


@pytest.mark.parametrize("mode", ["time", "output_load"])
@pytest.mark.parametrize("duration", [0.11, 0.100000000001])
def test_real_decimal_duration_overrun_is_still_rejected(mode, duration):
    settings = {"mode": mode, "analysis_seconds": duration}
    if mode == "time":
        settings["interval_seconds"] = 0.1
    else:
        settings["load_values"] = list(range(101))

    with pytest.raises(ValueError, match="每段分析时长不能超过分段时长"):
        build_segment_plan(settings, 10.1, 48000)


def test_repeated_loads_and_fractional_sample_boundaries():
    settings = normalize_segmented_analysis({"output_load": {
        "enabled": True, "values": [0, 0.3, 0.3], "analysis_seconds": 0.1}})
    plan = build_segment_plan(settings, 1.001, 1000)
    assert [s.end_sample for s in plan] == [334, 667, 1001]
    assert len({s.label for s in plan}) == 3
    assert plan[1].end_sample == plan[2].start_sample
    assert all(s.window_end_sample - s.window_start_sample == 100 for s in plan)


@pytest.mark.parametrize("unit,labels", [
    ("s", ["时间3600s", "时间7200s"]),
    ("min", ["时间60min", "时间120min"]),
    ("h", ["时间1h", "时间2h"]),
])
def test_time_unit_display_does_not_change_hour_segment_windows(unit, labels):
    plan = build_segment_plan({"mode": "time", "interval_seconds": 3600,
                               "display_time_unit": unit, "analysis_seconds": 10}, 7200, 1000)
    assert [segment.label for segment in plan] == labels
    assert [(s.window_start_sample / 1000, s.window_end_sample / 1000) for s in plan] == [
        (1795, 1805), (5395, 5405),
    ]


@pytest.mark.parametrize("seconds,interval,window", [(100, 60, 10), (60, 60, 61), (60, 0, 10)])
def test_invalid_time_plan_is_rejected(seconds, interval, window):
    with pytest.raises(ValueError):
        build_segment_plan({"mode": "time", "interval_seconds": interval,
                            "analysis_seconds": window}, seconds, 44100)


@pytest.mark.parametrize("values", [[], [-1], [float('nan')], [True]])
def test_invalid_loads_are_rejected(values):
    with pytest.raises(ValueError):
        normalize_segmented_analysis({"segmented_analysis": {
            "mode": "output_load", "load_values": values, "analysis_seconds": 10}})


@pytest.mark.parametrize("unit", ["A", "mA", "W", "%", "Ω", "mW", "自定义单位"])
def test_custom_unit_is_trimmed_and_used_as_label_without_converting_values(unit):
    settings = normalize_segmented_analysis({"segmented_analysis": {
        "mode": "output_load", "load_values": [0, 0.3],
        "load_unit": f"  {unit}  ", "analysis_seconds": 10,
    }})
    assert settings["load_unit"] == unit
    plan = build_segment_plan(settings, 120, 1000)
    assert plan[1].label == f"输出负载0.3{unit}"
    assert plan[1].value == 0.3 and plan[1].unit == unit


@pytest.mark.parametrize("unit", ["", "   ", None, 123])
def test_empty_or_non_text_unit_is_rejected(unit):
    with pytest.raises(ValueError, match="请选择或输入负载单位"):
        normalize_segmented_analysis({"segmented_analysis": {
            "mode": "output_load", "load_values": [0],
            "load_unit": unit, "analysis_seconds": 10,
        }})
