"""Raw-voltage calibration integrity, with no GUI, SDK or real device."""
from dataclasses import replace
from copy import copy

import numpy as np
import pytest

from base.recording_process_protocol import FrozenConfig, RecordingResult
from unit_test.base.ve3668n_fakes import capture_request


def calibration_request(tmp_path, rate=51200, **overrides):
    return capture_request(
        tmp_path / "calibration.wav", sample_rate=rate, purpose="calibration",
        target_samples=rate * 10, channels=(7,), trim_samples=0,
        calibration_metadata=None, **overrides,
    )


def descriptor_for(request):
    return RecordingResult(
        request.request_id, request.purpose, request.path, request.sample_rate,
        request.channels, request.target_samples, request.target_samples, False,
    )


def sine_volts(rate=51200, standard=94, factor=10):
    p_rms = 20e-6 * 10 ** (standard / 20)
    time = np.arange(rate * 10, dtype=np.float64) / rate
    return (np.sqrt(2) * p_rms / factor * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_verify_accepts_ten_second_raw_voltage_without_existing_factor(tmp_path, rate):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path, rate)
    volts = sine_volts(rate)
    original = volts.copy()
    result = verify_ve_calibration_result(request, descriptor_for(request), volts)
    assert result is volts
    assert len(result) == rate * 10
    assert request.device["input_config"]["unit"] == "V"
    assert request.calibration_metadata is None
    np.testing.assert_array_equal(volts, original)


@pytest.mark.parametrize("field,value", [
    ("request_id", "stale"), ("purpose", "main"), ("path", "wrong.wav"),
    ("sample_rate", 44100), ("sample_rate", 51200.0),
    ("channels", (1,)), ("channels", (7, 1)),
    ("raw_frames", 511999), ("final_frames", 511999),
    ("raw_frames", 512000.0), ("handles_released", False),
])
def test_verify_rejects_descriptor_mismatch(tmp_path, field, value):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path)
    descriptor = replace(descriptor_for(request), **{field: value})
    with pytest.raises(ValueError):
        verify_ve_calibration_result(request, descriptor, sine_volts())


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("purpose", "main"), ("sample_rate", 44100),
    ("target_samples", 1200), ("trim_samples", 1), ("channels", (8,)),
    ("channels", (7, 1)), ("monitor", FrozenConfig.snapshot({"enabled": True})),
    ("streaming", True),
])
def test_verify_revalidates_frozen_request_contract(tmp_path, field, value):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = copy(calibration_request(tmp_path))
    # Emulate a malformed value crossing a deserialization boundary.
    object.__setattr__(request, field, value)
    with pytest.raises(ValueError):
        verify_ve_calibration_result(request, descriptor_for(request), sine_volts())


@pytest.mark.parametrize("field,value", [
    ("backend", "sounddevice"), ("machine_id", ""), ("model", "other"),
    ("available", False), ("physical_channels", [1]),
    ("input_config", {"sample_rate": 51200}),
])
def test_verify_requires_explicit_valid_ve_identity(tmp_path, field, value):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path)
    device = {**request.device.to_dict(), field: value}
    object.__setattr__(request, "device", FrozenConfig.snapshot(device))
    with pytest.raises(ValueError):
        verify_ve_calibration_result(request, descriptor_for(request), sine_volts())


@pytest.mark.parametrize("invalid", ["scalar", "row", "columns", "cube", "short", "double", "nan", "inf"])
def test_verify_rejects_bad_raw_arrays_before_any_flatten(tmp_path, invalid):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path)
    volts = sine_volts()
    if invalid == "scalar":
        volts = np.array(1, dtype=np.float32)
    elif invalid == "row":
        volts = volts[None, :]
    elif invalid == "columns":
        volts = volts.reshape(-1, 2)
    elif invalid == "cube":
        volts = volts[:, None, None]
    elif invalid == "short":
        volts = volts[:1200]
    elif invalid == "double":
        volts = volts.astype(np.float64)
    else:
        volts[0] = np.nan if invalid == "nan" else np.inf
    with pytest.raises(ValueError):
        verify_ve_calibration_result(request, descriptor_for(request), volts)


def test_verify_accepts_reader_column_without_copy_or_file_reopen(tmp_path):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path)
    volts = sine_volts()[:, None]
    result = verify_ve_calibration_result(request, descriptor_for(request), volts)
    assert result.shape == (request.target_samples,)
    assert np.shares_memory(result, volts)
    assert not (tmp_path / "calibration.wav").exists()


@pytest.mark.parametrize("amplitude", [0.0, 0.5e-10, 1e-10])
def test_ac_floor_is_strict_and_cannot_create_fake_calibration(amplitude):
    from base.ve3668n_calibration import require_ac_signal

    volts = np.tile([amplitude, -amplitude], 3072)
    with pytest.raises(ValueError, match="无有效交流信号，无法校准"):
        require_ac_signal(volts)


@pytest.mark.parametrize("volts", [
    np.full(6144, 2.5), np.zeros(0), np.zeros(1200),
    np.r_[np.nan, np.ones(6143)], np.r_[np.inf, np.ones(6143)],
    np.zeros((6144, 2)), np.zeros((6144, 1, 1)),
])
def test_ac_rejects_dc_nonfinite_short_and_wrong_shape(volts):
    from base.ve3668n_calibration import require_ac_signal

    with pytest.raises(ValueError):
        require_ac_signal(volts)


@pytest.mark.parametrize("amplitude,offset", [(1.01e-10, 0), (0.25, 2.5)])
def test_ac_above_floor_uses_float64_and_does_not_demean_algorithm_input(amplitude, offset):
    from base.ve3668n_calibration import require_ac_signal

    volts = np.tile([amplitude, -amplitude], 3072) + offset
    before = volts.copy()
    require_ac_signal(volts)
    np.testing.assert_array_equal(volts, before)


def test_ac_uses_only_raw_central_interval_for_signal_energy():
    from base.ve3668n_calibration import require_ac_signal

    volts = np.zeros(6144)
    volts[:1000] = np.tile([1., -1.], 500)
    with pytest.raises(ValueError, match="无有效交流信号"):
        require_ac_signal(volts)


@pytest.mark.parametrize("volts", [np.zeros(512000, np.float32), np.full(512000, 2.5, np.float32),
                                  sine_volts(factor=1e12)])
def test_result_ac_gate_runs_even_when_optional_quality_is_disabled(tmp_path, volts):
    from base.ve3668n_calibration import verify_ve_calibration_result

    request = calibration_request(tmp_path)
    assert request.validation_thresholds["enabled"] is False
    with pytest.raises(ValueError, match="无有效交流信号"):
        verify_ve_calibration_result(request, descriptor_for(request), volts)


def sine_capture_dependencies(*, standard=94, parent_pid=None):
    """Importable spawn fake: real VE owner/capture/writer, no native library."""
    import ctypes
    import os
    import time

    from unit_test.base.ve3668n_fakes import CaptureSDK

    assert parent_pid is None or os.getpid() != parent_pid

    class SineSDK(CaptureSDK):
        def __init__(self):
            super().__init__()
            self.offset = 0

        def read_task_data(self, task, *, channel_count, samples_per_channel, timeout_seconds):
            self._call("read_task_data", task)
            count = samples_per_channel
            t = (np.arange(count, dtype=np.float64) + self.offset) / self.rate
            values = np.sqrt(2) * (20e-6 * 10 ** (standard / 20)) / 10 * np.sin(2 * np.pi * 1000 * t)
            buffer = (ctypes.c_double * (channel_count * count))()
            np.ctypeslib.as_array(buffer).reshape(channel_count, count)[:] = values
            self.offset += count
            time.sleep(.002)  # Give the real bounded writer queue time to drain.
            return buffer, count

    return {"ve_sdk_factory": SineSDK}
