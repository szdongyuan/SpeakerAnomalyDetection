"""Independent four-station/two-condition samples for workbench tests."""

import json


def workbench_source(*, all_stations=False):
    stations = [{
        "station_id": f"s{index + 1}",
        "station_name": f"工位{index + 1}",
        "channels": [{
            "channel_id": f"ch{channel + 1}",
            "channel_name": f"声音{channel + 1}",
            "signal_type": "audio",
            "hardware_channel": {
                "device_ref": "device-1",
                "physical_channel_index": index * 2 + channel,
            },
        } for channel in range(2)],
    } for index in range(4)]
    conditions = [{
        "condition_id": f"c{index + 1}",
        "condition_name": f"工况{index + 1}",
        "test_queue": f"测试队列{queue}",
        "applicable_station_ids": (
            [station["station_id"] for station in stations]
            if all_stations else (["s1", "s2"] if index == 0 else ["s2", "s4"])
        ),
        "trigger_state": "",
        "segmented_analysis": {"mode": "time", "settings": {"intervals": [[0, 1], [1, 2]]}},
    } for index, queue in enumerate(("A", "B"))]
    queues = {}
    for index, queue in enumerate(("A", "B")):
        analysis = {
            "display_sequence": ["声压级", "频谱"],
            "auto_analysis": True,
            "声压级": {
                "type": "SPL", "enabled": True,
                "weighting": "A" if index == 0 else "C",
                "analysis_channel": index, "analysis_channels": [index],
                "analysis_time_range_enabled": True,
                "analysis_start_time_sec": 0.1, "analysis_end_time_sec": 2.0,
                "smooth_checked": True, "smooth_enabled": True,
                "smooth_unit": "time", "smooth_time_sec": 0.125,
                "smooth_points": 32, "smooth_algo": 3,
                "limit_checked": True, "limit_metric": "overall_spl",
                "scalar_upper_enabled": True, "scalar_upper_value": 65 + index * 10,
                "scalar_lower_enabled": False, "scalar_lower_value": 0.0,
                "threshold": 65 + index * 10, "offset": 1.5,
                "limits": {"values": [[0, 30 + index], [1, 60 + index]]},
                "baseline": {"enabled": True, "values": [1, 2]},
                "display": {"axis": {"minimum": 0, "maximum": 100}},
                "save_result": True, "save_image": True,
                "custom_valid_field": {"nested": [None, {"gain": 1.25}]},
            },
            "频谱": {
                "type": "FFT", "enabled": False,
                "weighting": "None" if index == 0 else "Z",
                "analysis_channel": 1 - index, "analysis_channels": [1 - index],
                "window": "hann" if index == 0 else "hamming",
                "n_fft": 1024 * (index + 1), "overlap_ratio": 0.5,
                "octave_smoothing": 3 if index == 0 else 6,
                "x_axis_scale": "log", "focus_range_enabled": True,
                "focus_min_hz": 100, "focus_max_hz": 20000,
                "baseline_file_path": "reference.wav", "baseline_display_mode": "overlay",
                "baseline_smooth_third_octave": True, "limit_checked": True,
                "limits": {"segments": [[20, 200, 50], [200, 2000, 70]]},
            },
        }
        for name in ("声压级", "频谱"):
            analysis[name].update({
                "limit_data": [[20, 50 + index, 10], [2000, 70 + index, 20]],
                "limit_mode": "manual", "manual_input_mode": "segments",
                "constant_upper_enabled": True, "constant_lower_enabled": False,
                "constant_upper_value": 80 + index, "constant_lower_value": 0.0,
                "manual_upper_enabled": True, "manual_lower_enabled": False,
                "manual_upper_segments": [[20, 200, 50 + index], [200, 2000, 70 + index]],
                "manual_lower_segments": [],
                "display": {
                    "main_curve_color": "#00ff00",
                    "plot_view": {"x_enabled": True, "x_min": 0, "x_max": 20000,
                                  "y_enabled": True, "y_min": -20, "y_max": 100},
                },
            })
        queues[f"测试队列{queue}"] = [{"seq1": {
            "acq": {"mode": "RECORD_ONLY", "detail": {
                "total_time": 2.5 + index, "startup_trim_ms": 0,
                "audio_validation": {"enabled": True, "min_rms_dbfs": -65.0,
                                     "min_peak": 0.0001, "min_variance": 1e-10},
                "sample_rate": 44100, "use_streaming_recording": False,
                "recording_root": "ignored-queue-root",
                "recording_preview_time_mode": "cumulative",
            }},
            "analysis_channel_basis": "wav",
            "analysis_list": analysis,
        }}, {"seq2": {"custom": {"values": [4, 6]}}}]
    return {
        "product": {
            "project_id": "motor", "project_name": "电机",
            "result_root_directory": "unused-results",
            "stations": stations, "conditions": conditions,
        },
        "queues": queues,
        "layout": {"rows": 2, "columns": 4, "cells": [[{
            "title": f"{name}{column + 1}", "analysis_item": name, "analysis_type": kind,
        } for column in range(4)] for name, kind in (("声压级", "SPL"), ("频谱", "FFT"))]},
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_workbench_files(tmp_path, *, all_stations=False):
    source = workbench_source(all_stations=all_stations)
    source["product"]["result_root_directory"] = str(tmp_path / "unused-results")
    files = {
        "programs": tmp_path / "products" / "program_registry.json",
        "product": tmp_path / "products" / "motor.json",
        "queues": tmp_path / "queues" / "sequence_config_registry.json",
        "layout": tmp_path / "layout.json",
    }
    write_json(files["programs"], {
        "active_file": "motor.json",
        "configs": [{"file": "motor.json", "project_name": "电机"}],
    })
    registry = {}
    for index, (name, sequence) in enumerate(source["queues"].items()):
        path = files["queues"].parent / f"queue-{index}.json"
        files[name] = path
        registry[name] = path.name
        write_json(path, sequence)
    write_json(files["queues"], registry)
    write_json(files["product"], source["product"])
    write_json(files["layout"], source["layout"])
    return files
