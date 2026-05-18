import importlib.util
import sys
from pathlib import Path

import pytest


def _load_stream_test():
    module_path = Path(__file__).resolve().parents[1] / "stream_test.py"
    spec = importlib.util.spec_from_file_location("stream_test_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _device(name, max_input_channels=0, max_output_channels=0, default_samplerate=48000.0):
    return {
        "name": name,
        "max_input_channels": max_input_channels,
        "max_output_channels": max_output_channels,
        "default_samplerate": default_samplerate,
    }


def test_configured_device_ids_are_used_directly(monkeypatch):
    stream_test = _load_stream_test()
    devices = [
        _device("Unused", max_input_channels=8, max_output_channels=8),
        _device("Configured Mic", max_input_channels=4),
        _device("Configured Speaker", max_output_channels=2),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 2)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    plan = stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)

    assert plan.input_id == 1
    assert plan.output_id == 2
    assert plan.input_name == "Configured Mic"
    assert plan.output_name == "Configured Speaker"
    assert plan.channels_in == 4
    assert plan.channels_out == 2


def test_portaudio_device_enumeration_error_is_normalized(monkeypatch):
    stream_test = _load_stream_test()

    def query_devices_fails():
        raise stream_test.sd.PortAudioError("PortAudio cannot enumerate devices")

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 0)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", query_devices_fails)

    with pytest.raises(
        stream_test.StreamTestError,
        match="Configured input/output device enumeration failed: input=0, output=1",
    ):
        stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)


def test_configured_devices_allow_digital_names_and_non_wasapi_hostapi(monkeypatch):
    stream_test = _load_stream_test()
    devices = [
        _device("Digital Output (S/PDIF)", max_input_channels=2),
        _device("Digital Output (S/PDIF)", max_output_channels=2),
    ]
    devices[0]["hostapi"] = 3
    devices[1]["hostapi"] = 7

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 0)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    plan = stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)

    assert plan.input_name == "Digital Output (S/PDIF)"
    assert plan.output_name == "Digital Output (S/PDIF)"
    assert plan.channels_in == 2
    assert plan.channels_out == 2


@pytest.mark.parametrize("configured_input_id", [-1, 1.5, "1"])
def test_invalid_configured_device_ids_fail_before_indexing(monkeypatch, configured_input_id):
    stream_test = _load_stream_test()
    devices = [
        _device("Configured Mic", max_input_channels=1),
        _device("Last Device Must Not Be Used", max_output_channels=2),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", configured_input_id)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    with pytest.raises(stream_test.StreamTestError, match="non-negative"):
        stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)


def test_invalid_configured_device_ids_fail_before_device_enumeration(monkeypatch):
    stream_test = _load_stream_test()

    def query_devices_must_not_run():
        raise AssertionError("query_devices should not run before configured ID validation")

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", -1)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", query_devices_must_not_run)

    with pytest.raises(stream_test.StreamTestError, match="non-negative"):
        stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)


@pytest.mark.parametrize(
    ("configured_input_id", "configured_output_id"),
    [
        (2, 1),
        (0, 2),
    ],
)
def test_out_of_range_configured_device_ids_raise(monkeypatch, configured_input_id, configured_output_id):
    stream_test = _load_stream_test()
    devices = [
        _device("Configured Mic", max_input_channels=1),
        _device("Configured Speaker", max_output_channels=2),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", configured_input_id)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", configured_output_id)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    with pytest.raises(stream_test.StreamTestError, match="Configured input/output device IDs are invalid"):
        stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)


@pytest.mark.parametrize(
    ("input_sample_rate", "output_sample_rate", "mismatched_device"),
    [
        (44100.0, 48000.0, "Configured Mic"),
        (48000.0, 44100.0, "Configured Speaker"),
    ],
)
def test_configured_device_default_sample_rate_mismatch_raises(
    monkeypatch,
    input_sample_rate,
    output_sample_rate,
    mismatched_device,
):
    stream_test = _load_stream_test()
    devices = [
        _device("Configured Mic", max_input_channels=1, default_samplerate=input_sample_rate),
        _device("Configured Speaker", max_output_channels=2, default_samplerate=output_sample_rate),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 0)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    with pytest.raises(stream_test.StreamTestError, match=f"Device {mismatched_device!r} default_samplerate"):
        stream_test._require_configured_duplex(file_ch=1, file_sr=48000.0)


def test_stereo_file_with_mono_input_is_accepted_when_output_supports_stereo(monkeypatch):
    stream_test = _load_stream_test()
    devices = [
        _device("Mono Mic", max_input_channels=1),
        _device("Stereo Speaker", max_output_channels=2),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 0)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    plan = stream_test._require_configured_duplex(file_ch=2, file_sr=48000.0)

    assert plan.channels_in == 1
    assert plan.channels_out == 2
    assert plan.file_channels == 2


def test_output_device_channel_validation_remains_enforced(monkeypatch):
    stream_test = _load_stream_test()
    devices = [
        _device("Input With Many Channels", max_input_channels=8),
        _device("Mono Speaker", max_output_channels=1),
    ]

    monkeypatch.setattr(stream_test, "INPUT_DEVICE_ID", 0)
    monkeypatch.setattr(stream_test, "OUTPUT_DEVICE_ID", 1)
    monkeypatch.setattr(stream_test.sd, "query_devices", lambda: devices)

    with pytest.raises(stream_test.StreamTestError, match="Configured output"):
        stream_test._require_configured_duplex(file_ch=2, file_sr=48000.0)
