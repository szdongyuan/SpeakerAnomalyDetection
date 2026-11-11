from dataclasses import FrozenInstanceError, asdict
import json
import pickle
import subprocess
import sys

import pytest

from base.analysis import build_analysis_batch


def test_snapshots_isolate_caller_tasks_and_channel_instances(batch_payload):
    batch = build_analysis_batch(batch_payload)
    first, second = batch.tasks[:2]
    batch_payload["tasks"][0]["analysis_config"]["FFT 1"]["display"]["main_curve_color"] = "changed"
    batch_payload["stations"][0]["channels"][0]["channel_name"] = "renamed"
    batch_payload["conditions"][0]["segmented_analysis"]["mode"] = "changed"
    assert first.channels[0].channel_name == "CH1"
    assert first.segmented_analysis == {"mode": "none"}
    assert first.instances[1].parameters["display"]["main_curve_color"] == "#33C44D"
    # Nested parameter dictionaries are owned by each instance, not shared.
    first.instances[1].parameters["display"]["main_curve_color"] = "local"
    assert first.instances[2].parameters["display"]["main_curve_color"] == "#33C44D"
    assert second.instances[1].parameters["display"]["main_curve_color"] == "#33C44D"
    with pytest.raises(FrozenInstanceError):
        first.station_id = "wrong-station"


def test_batch_round_trips_through_pickle_and_plain_json(batch_payload):
    batch = build_analysis_batch(batch_payload)
    assert pickle.loads(pickle.dumps(batch)) == batch
    plain = json.loads(json.dumps(asdict(batch), ensure_ascii=False, allow_nan=False))
    assert plain["tasks"][1]["instances"][0]["channel_index"] == 0
    assert plain["preparation_issues"][0]["task_id"]


def test_import_does_not_load_ui_recording_or_calculation_libraries():
    script = """
import sys
from base.analysis import build_analysis_batch
assert build_analysis_batch({'stations': [], 'conditions': [], 'tasks': []}).tasks == ()
for prefix in ('ui', 'PyQt5', 'PySide6', 'numpy', 'scipy', 'matplotlib', 'soundfile',
               'base.recording_process_protocol', 'base.analysis_algorithm_adapters'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
