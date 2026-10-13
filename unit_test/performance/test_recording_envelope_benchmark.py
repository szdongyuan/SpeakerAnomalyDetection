"""Small real-process smoke for the developer-only local benchmark."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_benchmark_runs_fresh_baseline_and_new_processes(tmp_path):
    output = tmp_path / 'profile.json'
    result = subprocess.run([
        sys.executable, 'unit_test/performance/benchmark_recording_final_envelope.py',
        '--seconds', '.02', '.03', '--channels', '5', '--sample-rate', '1000',
        '--block-frames', '8', '--output', str(output),
    ], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    profile = json.loads(output.read_text(encoding='utf-8'))
    assert len(profile['runs']) == 4
    assert len({run['parent_pid'] for run in profile['runs']}) == 4
    for baseline, new in zip(profile['runs'][::2], profile['runs'][1::2]):
        assert baseline['display_digest'] == new['display_digest']
    for run in profile['runs']:
        assert run['frames'] in (20, 30)
        assert run['channels'] == 5
        assert run['memory']['sampled_simultaneous_peak_bytes'] > 0
        assert run['memory']['parent_peak_bytes'] > 0
        assert run['memory']['child_peak_bytes'] > 0
        assert run['stages']['capture']['wall_seconds'] > 0
        assert run['gui_publication'] is None
        assert run['child_target_to_descriptor_seconds'] > 0
        assert run['parent_descriptor_to_offer_seconds'] > 0
        if run['route'] == 'new':
            assert run['sample_read_calls'] == 0
            assert run['raw_array_bytes'] == 0
            assert run['stages']['parent_prepare']['wall_seconds'] > 0
            assert all(points == run['frames'] for points in run['display_points'])
        else:
            assert run['sample_read_calls'] > 0
            assert run['raw_array_bytes'] == run['frames'] * 6 * 4


@pytest.mark.parametrize('target', [r'\\server\share\profile.json', '//server/share/profile.json'])
def test_benchmark_rejects_network_output_before_io(target):
    from unit_test.performance.benchmark_recording_final_envelope import local_output_path
    with pytest.raises(ValueError, match='local'):
        local_output_path(target)
