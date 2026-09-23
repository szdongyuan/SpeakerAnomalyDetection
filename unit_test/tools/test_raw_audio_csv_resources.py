"""Exited handles provide lifetime CPU totals, never live resource samples."""
import subprocess
import sys

import pytest

from tools.benchmark_raw_audio_csv_process import validate_resources
from tools import raw_audio_csv_benchmark_load as load


def test_native_exited_handle_has_only_final_lifetime_cpu():
    native = load.WindowsProcessMetrics()
    child = subprocess.Popen([sys.executable, '-c',
        'import sys; sys.stdin.readline(); sum(i*i for i in range(2000000))'], stdin=subprocess.PIPE)
    handle = native.open(child.pid)
    try:
        live = native.read(handle)
        assert live['exited'] is False
        assert live['working_set_bytes'] > 0
        child.communicate(b'go\n', timeout=15)
        final = native.read(handle)
        assert final['exited'] is True
        assert final['cpu_seconds'] is None
        assert final['working_set_bytes'] is None
        assert final['peak_working_set_bytes'] is None
        assert final['final_lifetime_cpu_seconds'] > live['cpu_seconds']
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        native.close(handle)


def evidence(cpu=1.0, memory=100, *, exited=False, final=None):
    return dict(cpu_seconds=cpu, working_set_bytes=memory, peak_working_set_bytes=10000,
                exited=exited, final_lifetime_cpu_seconds=final)


def sampler_report(monkeypatch, values):
    readings = iter(values)
    class Native:
        def open(self, pid): return pid
        def read(self, handle): return next(readings)
        def close(self, handle): return None
    monkeypatch.setattr(load, 'WindowsProcessMetrics', Native)
    sampler = load.ResourceSampler()
    sampler.expected_roles = ['parent']
    sampler.register('parent', 42)
    sampler.sample('periodic')
    return sampler.finish()


def test_live_samples_plus_exit_total_include_work_after_last_periodic(monkeypatch):
    report = sampler_report(monkeypatch, [evidence(), evidence(2, 150),
        evidence(None, None, exited=True, final=3.5)])
    assert report['validation_status'] == 'pass'
    record = report['processes']['parent:42']
    assert record['cpu_seconds'] == 2.5
    assert record['final_lifetime_cpu_seconds'] == 3.5
    assert record['peak_working_set_bytes'] == 150
    assert record['final_cpu_source'] == 'exited_process_lifetime'
    assert report['samples'][-1]['cpu_seconds'] is None
    assert report['samples'][-1]['working_set_bytes'] is None


@pytest.mark.parametrize('case', ['numeric_exited_only', 'missing_live_memory', 'missing_final_cpu'])
def test_missing_live_or_final_evidence_is_unverified(monkeypatch, case):
    values = [evidence(), evidence(2), evidence(None, None, exited=True, final=3)]
    if case == 'numeric_exited_only':
        values = [evidence(1, 32768, exited=True, final=3)] * 3
    elif case == 'missing_live_memory':
        values[:2] = [evidence(1, None), evidence(2, None)]
    else:
        values[-1] = evidence(None, None, exited=True)
    report = sampler_report(monkeypatch, values)
    assert report['validation_status'] == 'unverified'
    if case == 'numeric_exited_only':
        assert all(s['cpu_seconds'] is None and s['working_set_bytes'] is None for s in report['samples'])


def test_validator_does_not_trust_numeric_exited_samples():
    report = dict(expected_roles=['parent'], samples=[dict(role='parent', pid=42, **evidence(exited=True))],
        processes={'parent:42': dict(role='parent', pid=42, cpu_seconds=1, peak_working_set_bytes=100,
            final_cumulative_cpu_seconds=2, final_cpu_source='exited_process_lifetime', incomplete=False)})
    assert validate_resources(report) == 'unverified'
