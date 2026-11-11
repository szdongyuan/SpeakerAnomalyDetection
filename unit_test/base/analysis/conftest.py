"""Controller payloads for input tests; no device or user configuration access."""

import pytest


@pytest.fixture
def batch_payload(tmp_path):
    stations = []
    for number in range(4):
        stations.append({
            "station_id": f"station-{number + 1}",
            "station_name": f"工位{number + 1}",
            "device_ref": "demo-device",
            "channels": [
                {
                    "channel_id": f"station-{number + 1}-channel-{index}",
                    "channel_name": f"CH{index + 1}",
                    "signal_type": "sound_pressure" if index == 0 else "acceleration",
                    "physical_channel_index": number * 4 + index,
                }
                for index in range(4)
            ],
        })
    conditions = [
        {
            "condition_id": f"condition-{number}",
            "condition_name": f"工况{number}",
            "segmented_analysis": {"mode": "none"},
        }
        for number in (1, 2)
    ]
    # Deliberately shared by the caller. Built tasks must not share mutable data.
    config = {
        "display_sequence": ["SPL 1", "FFT 1"],
        "SPL 1": {
            "type": "SPL", "analysis_channels": [0], "weighting": "A",
            "limit_checked": False,
            "analysis_time_range_enabled": False,
            "analysis_start_time_sec": 0.0, "analysis_end_time_sec": 0.0,
        },
        "FFT 1": {
            "type": "FFT", "analysis_channels": [0, 1, 2, 3],
            "n_fft": 4096, "overlap_ratio": 0.5,
            "display": {"main_curve_color": "#33C44D"},
        },
    }
    tasks = [
        {
            "task_id": f"analysis-{c + 1}-{s + 1}",
            "station_id": station["station_id"],
            "condition_id": condition["condition_id"],
            "wav_path": str(tmp_path / f"recording-{c}-{s}.wav"),
            "analysis_result_directory": None,
            "analysis_config": config,
        }
        for c, condition in enumerate(conditions)
        for s, station in enumerate(stations)
    ]
    return {"stations": stations, "conditions": conditions, "tasks": tasks}
