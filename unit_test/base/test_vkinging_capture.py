"""Capture tests exercise the shared SDK double and real floating WAV writer."""
import ctypes
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.vkinging_input import VkingingCaptureRequest
from unit_test.base.vkinging_fakes import FakeSDK


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Control:
    def __init__(self):
        self.cancelled = False
        self.phases = []
        self.progress = []
        self.released = False
        self.cleanup = []

    def set_phase(self, phase):
        self.phases.append(phase)

    def report_progress(self, frames):
        self.progress.append(frames)

    def confirm_released(self):
        self.released = True

    def report_cleanup_failure(self, reason):
        self.cleanup.append(reason)


def request(tmp_path, channels=(15, 0, 8), target=5, discard=1, bits=64):
    metadata = {
        'backend': 'vkinging', 'schema_version': 1,
        'acquisition': {'hardware_id': 'id', 'model': None, 'input_mode': 'IEPE',
                        'unit': 'V', 'range_min': -10.0, 'range_max': 10.0,
                        'sample_rate': 51200, 'bit_depth': bits},
        'recorded_channels': [
            {'wav_channel_index': i, 'physical_channel_index': ch, 'calibrated': False,
             'v2pa_factor': None, 'standard_spl': None, 'calibration_conditions': None}
            for i, ch in enumerate(channels)]}
    return VkingingCaptureRequest(
        request_id='capture', purpose='main', device={
            'hostapi_name': 'vkinging', 'hardware_id': 'id', 'name': 'old',
            'physical_channels': channels, 'samplerate': 51200, 'bit_depth': bits},
        channels=channels, sample_rate=51200, bit_depth=bits, range_index=0,
        target_samples=target, discard_initial_samples=discard,
        output_path=str(tmp_path / 'final.wav'), calibration_snapshot=metadata)


def sdk_for(req, reads=()):
    return FakeSDK(devices=(('addr', 'fresh'),),
                   attributes={('fresh', 'MachineId'): 'id', ('fresh', 'Model'): ''},
                   channels={'fresh': tuple(f'fresh/AIN{ch+1}' for ch in req.channels)},
                   reads=reads)


def publish(result, control):
    assert result.resources_released
    prepared = result.finalize(control)
    assert prepared.resources_released
    assert prepared.finalize is None
    return prepared.commit(control)


def test_scan_prefix_short_zero_reads_trim_and_exact_tail(tmp_path):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path)
    values = np.arange(24, dtype='float64').reshape(8, 3) + 2.5
    sdk = sdk_for(req, [(values[:2].ravel(), 2), ([], 0),
                        (values[2:4].ravel(), 2), (values[4:].ravel(), 4)])
    control = Control()
    result = capture_vkinging(sdk, req, control, block_frames=5)
    assert not sdk.closed  # Only the service owns SDK close.
    assert not Path(req.output_path).exists()
    descriptor = publish(result, control)
    actual, rate = sf.read(descriptor['path'], always_2d=True, dtype='float64')
    np.testing.assert_array_equal(actual, values[1:6])
    assert rate == 51200
    assert control.progress == [2, 4, 8]
    calls = [call[0] for call in sdk.calls]
    assert calls.index('start_task') < calls.index('verify_actual_sample_rate') < calls.index('read_task_data')
    assert calls[-2:] == ['stop_task', 'clear_task']
    route = next(call[1][1] for call in sdk.calls if call[0] == 'create_iepe_voltage_channel')
    assert route == 'fresh/AIN16,fresh/AIN1,fresh/AIN9'
    reads = [call[1] for call in sdk.calls if call[0] == 'read_task_data']
    assert all(args[1:3] == (3, 5) and 0 < args[3] <= .2 for args in reads)
    assert descriptor['physical_channels'] == [15, 0, 8]


@pytest.mark.parametrize('channels', [tuple(range(32)), (63, 31, 8, 0)])
@pytest.mark.parametrize('bits', [32, 64])
def test_arbitrary_channel_count_and_order_preserve_raw_volts(tmp_path, channels, bits):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path, channels=channels, target=2, discard=0, bits=bits)
    values = np.arange(2 * len(channels), dtype='float64').reshape(2, -1) - 3
    sdk = sdk_for(req, [(values.ravel(), 2)])
    descriptor = publish(capture_vkinging(sdk, req, Control()), Control())
    actual, _ = sf.read(descriptor['path'], always_2d=True, dtype='float64')
    np.testing.assert_array_equal(actual, values)
    assert descriptor['physical_channels'] == list(channels)


@pytest.mark.parametrize('operation', [
    'create_task', 'create_iepe_voltage_channel', 'configure_sample_clock',
    'start_task', 'verify_actual_sample_rate', 'read_task_data', 'stop_task', 'clear_task'])
def test_native_failures_keep_diagnostic_and_attempt_all_cleanup(tmp_path, operation):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path, target=1, discard=0)
    sdk = sdk_for(req, [([1., 2., 3.], 1)])
    sdk.failures[operation] = RuntimeError('native ' + operation)
    control = Control()
    with pytest.raises(RuntimeError, match='native ' + operation):
        capture_vkinging(sdk, req, control)
    operations = [call[0] for call in sdk.calls]
    assert ('stop_task' in operations) == (operation != 'create_task')
    assert ('clear_task' in operations) == (operation != 'create_task')
    assert not Path(req.output_path).exists()
    assert not sdk.closed
    if operation in ('stop_task', 'clear_task'):
        assert control.cleanup and not control.released
    else:
        assert control.released


def test_primary_and_both_cleanup_failures_are_retained(tmp_path):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path)
    sdk = sdk_for(req)
    for operation in ('read_task_data', 'stop_task', 'clear_task'):
        sdk.failures[operation] = RuntimeError(operation)
    control = Control()
    with pytest.raises(RuntimeError, match='read_task_data'):
        capture_vkinging(sdk, req, control)
    assert len(control.cleanup) == 2
    assert 'stop_task' in control.cleanup[0] and 'clear_task' in control.cleanup[1]
    assert not control.released


@pytest.mark.parametrize('count', [-1, 6, True, 1.5])
def test_invalid_frame_counts_are_rejected(tmp_path, count):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path)
    sdk = sdk_for(req, [([1., 2., 3.], count)])
    with pytest.raises(ValueError, match='count'):
        capture_vkinging(sdk, req, Control(), block_frames=5)


@pytest.mark.parametrize('buffer', [
    (ctypes.c_float * 15)(*range(15)), (ctypes.c_double * 14)(*range(14)),
    (ctypes.c_double * 15)(float('nan'), *range(14)),
    (ctypes.c_double * 15)(float('inf'), *range(14)), np.arange(15, dtype='float64')])
def test_invalid_buffer_type_capacity_or_valid_prefix_is_rejected(tmp_path, buffer):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path)
    sdk = sdk_for(req)
    sdk.read_task_data = lambda *args, **kwargs: (buffer, 1)
    with pytest.raises(ValueError, match='buffer|finite'):
        capture_vkinging(sdk, req, Control(), block_frames=5)


@pytest.mark.parametrize('kind', ['stalled', 'overall'])
def test_fake_clock_deadlines_do_not_require_service_tick_or_real_wait(tmp_path, kind):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path, target=100, discard=0)
    clock, control = Clock(), Control()
    sdk = sdk_for(req)
    def read(*args, **kwargs):
        clock.now += 3 if kind == 'stalled' else 1
        count = 0 if kind == 'stalled' else 1
        buffer = (ctypes.c_double * (kwargs['samples_per_channel'] * 3))()
        return buffer, count
    sdk.read_task_data = read
    with pytest.raises(TimeoutError, match='read_stall|capture_timeout'):
        capture_vkinging(sdk, req, control, clock=clock)
    assert control.released
    assert [c[0] for c in sdk.calls][-2:] == ['stop_task', 'clear_task']


@pytest.mark.parametrize('change', ['identity', 'routes', 'snapshot', 'missing_snapshot', 'riff'])
def test_preflight_rejects_before_task_creation(tmp_path, change):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path)
    sdk = sdk_for(req)
    if change == 'identity':
        sdk.attributes['fresh', 'MachineId'] = 'different'
    elif change == 'routes':
        sdk.channels['fresh'] = ('fresh/AIN1',)
    elif change == 'snapshot':
        req = replace(req, range_index=1)
    elif change == 'missing_snapshot':
        req = replace(req, calibration_snapshot=None)
    else:
        req = replace(req, target_samples=2**32)
    control = Control()
    with pytest.raises(ValueError):
        capture_vkinging(sdk, req, control)
    assert not any(c[0] == 'create_task' for c in sdk.calls)
    assert control.released


def test_cancel_during_read_never_publishes(tmp_path):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path, target=1, discard=0)
    control = Control()
    sdk = sdk_for(req, [([1., 2., 3.], 1)])
    original = sdk.read_task_data
    def read(*args, **kwargs):
        control.cancelled = True
        return original(*args, **kwargs)
    sdk.read_task_data = read
    with pytest.raises(RuntimeError, match='cancel'):
        capture_vkinging(sdk, req, control)
    assert control.released and not Path(req.output_path).exists()


def test_calibration_generates_only_ten_second_mono_wav_descriptor(tmp_path):
    from base.vkinging_capture import capture_vkinging
    req = request(tmp_path, channels=(15,))
    device = dict(req.device, samplerate=4000)
    req = replace(req, purpose='calibration', device=device, sample_rate=4000,
                  target_samples=40000, discard_initial_samples=0,
                  calibration_snapshot=None, output_path=None)
    sdk = sdk_for(req, [(np.full(4000, 2.5), 4000)] * 10)
    result = capture_vkinging(sdk, req, Control(), block_frames=4000, temp_directory=tmp_path)
    descriptor = publish(result, Control())
    actual, rate = sf.read(descriptor['path'], dtype='float64')
    np.testing.assert_array_equal(actual, np.full(40000, 2.5))
    assert rate == 4000 and descriptor['frames'] == 40000
    assert descriptor['metadata']['recorded_channels'][0]['calibrated'] is False


def test_cancelled_finalizer_checks_before_and_during_copy(tmp_path, monkeypatch):
    from base.vkinging_capture import capture_vkinging
    from base.vkinging_wav import VkingingVoltageWriter
    req = request(tmp_path, target=1, discard=0)
    result = capture_vkinging(sdk_for(req, [([1., 2., 3.], 1)]), req, Control())
    control = Control()
    original = VkingingVoltageWriter.finalize
    def finalize(writer, *, checkpoint):
        def cancel():
            control.cancelled = True
            checkpoint()
        return original(writer, checkpoint=cancel)
    monkeypatch.setattr(VkingingVoltageWriter, 'finalize', finalize)
    with pytest.raises(RuntimeError, match='cancel'):
        result.finalize(control)
    assert control.released and not Path(req.output_path).exists()
    with pytest.raises(RuntimeError, match='cancel'):
        result.finalize(control)


def test_broken_cleanup_exception_text_cannot_skip_other_releases(tmp_path):
    from base.vkinging_capture import capture_vkinging
    class BrokenError(RuntimeError):
        def __str__(self):
            raise RuntimeError('broken diagnostic formatter')
    req = request(tmp_path)
    sdk = sdk_for(req)
    sdk.failures['read_task_data'] = RuntimeError('primary read')
    sdk.failures['stop_task'] = BrokenError()
    sdk.failures['clear_task'] = RuntimeError('secondary clear')
    control = Control()
    with pytest.raises(RuntimeError, match='primary read'):
        capture_vkinging(sdk, req, control)
    assert [call[0] for call in sdk.calls][-2:] == ['stop_task', 'clear_task']
    assert 'BrokenError' in control.cleanup[0] and 'secondary clear' in control.cleanup[1]
    assert not control.released


@pytest.mark.parametrize('phase', ['write', 'close', 'finalize', 'commit'])
def test_file_failures_keep_release_proof_and_never_publish(tmp_path, monkeypatch, phase):
    from base.vkinging_capture import capture_vkinging
    from base.vkinging_wav import VkingingVoltageWriter
    req = request(tmp_path, target=1, discard=0)
    sdk = sdk_for(req, [([1., 2., 3.], 1)])
    control, writers = Control(), []
    original_init = VkingingVoltageWriter.__init__
    def init(writer, *args, **kwargs):
        original_init(writer, *args, **kwargs)
        writers.append(writer)
    monkeypatch.setattr(VkingingVoltageWriter, '__init__', init)
    def fail(*args, **kwargs):
        raise OSError(phase + ' disk failure')
    method = {'write': 'write_chunk', 'close': 'close', 'finalize': 'finalize', 'commit': 'commit'}[phase]
    monkeypatch.setattr(VkingingVoltageWriter, method, fail)
    try:
        with pytest.raises((OSError, RuntimeError), match=phase + ' disk failure'):
            result = capture_vkinging(sdk, req, control)
            prepared = result.finalize(control)
            prepared.commit(control)
        assert not Path(req.output_path).exists()
        assert control.released == (phase != 'close')
        assert [call[0] for call in sdk.calls][-2:] == ['stop_task', 'clear_task']
        assert bool(control.cleanup) == (phase == 'close')
    finally:
        # A fake close failure really left the test-owned handle open.
        for writer in writers:
            if not writer._handle.closed:
                writer._handle.close()


@pytest.mark.parametrize('phase', ['finalize', 'commit'])
def test_file_hook_exception_uses_new_release_proof(tmp_path, monkeypatch, phase):
    from base.vkinging_capture import capture_vkinging
    from base.vkinging_wav import VkingingVoltageWriter
    req = request(tmp_path, target=1, discard=0)
    result = capture_vkinging(sdk_for(req, [([1., 2., 3.], 1)]), req, Control())
    control = Control()
    prepared = result.finalize(control) if phase == 'commit' else None
    def fail(writer, **kwargs):
        writer._uncertain = True
        raise OSError('new handle release uncertain')
    monkeypatch.setattr(VkingingVoltageWriter, phase, fail)
    with pytest.raises(OSError, match='new handle release uncertain'):
        (prepared.commit if prepared else result.finalize)(control)
    assert not control.released and control.cleanup
    assert not Path(req.output_path).exists()
