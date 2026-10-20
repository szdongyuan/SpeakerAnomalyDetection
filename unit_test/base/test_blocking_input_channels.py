import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base.soundcard_audio_processor as soundcard
from consts import error_code


@pytest.mark.parametrize("mode", ["record", "playrec"])
def test_selected_first_channel_captures_six_inputs(monkeypatch, mode):
    raw = np.arange(18, dtype=np.float32).reshape(3, 6)
    calls = []
    monkeypatch.setattr(soundcard.sd, "query_devices",
                        lambda device=None, kind=None: {"max_input_channels": 6})

    def capture(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["channels"] == 6, "device only accepts its full input count"
        return raw.copy()

    monkeypatch.setattr(soundcard.sd, "rec", capture)
    monkeypatch.setattr(soundcard.sd, "playrec", capture)
    monkeypatch.setattr(soundcard.SoundcardAudioProcessor, "calculate_alignment",
                        staticmethod(lambda reference, recorded: 0))
    saved = []
    monkeypatch.setattr(soundcard, "save_audio_simple",
                        lambda path, data, sr, **kw: saved.append(data.copy()))
    record = {"device": {"index": 18}, "input_channels": [0],
              "sr": 48000, "sample_rate": 48000, "num_frames": 3,
              "prepare_frames": 0, "prolong_frames": 0}
    if mode == "record":
        code, data = soundcard.SoundcardAudioProcessor.sd_rec(record)
    else:
        record["output_device"] = {"index": 16}
        stimulus = {"sr": 48000, "data": np.ones(3, dtype=np.float32),
                    "amplitude": 1.0}
        code, data = soundcard.SoundcardAudioProcessor().sd_play_rec(
            record, stimulus, "unused.wav")
        np.testing.assert_array_equal(saved[0], raw[:, 0])
    assert code == error_code.OK
    np.testing.assert_array_equal(data, raw[:, 0])
    assert len(calls) == 1


@pytest.fixture
def blocking_device(monkeypatch):
    class Device:
        maximum = 6
        query_error = None
        capture_error = None

        def __init__(self):
            self.queries = []
            self.captures = []
            self.saved = []
            self.alignments = []

        def query(self, device=None, kind=None):
            self.queries.append((device, kind))
            if self.query_error is not None:
                raise self.query_error
            return {"max_input_channels": self.maximum}

        def capture(self, *args, **kwargs):
            self.captures.append(kwargs)
            if self.capture_error is not None:
                raise self.capture_error
            assert kwargs["channels"] == self.maximum
            return self.raw.copy()

        @property
        def raw(self):
            return np.arange(3 * self.maximum, dtype=np.float32).reshape(3, self.maximum)

        def save(self, path, data, sr, *args, **kwargs):
            self.saved.append(np.asarray(data).copy())

        def align(self, reference, recorded):
            self.alignments.append(np.asarray(recorded).copy())
            return 0

        def run(self, mode, **overrides):
            record = {"device": {"index": 18},
                      "sr": 48000, "sample_rate": 48000, "num_frames": 3,
                      "prepare_frames": 0, "prolong_frames": 0}
            record.update(overrides)
            if mode == "record":
                return soundcard.SoundcardAudioProcessor.sd_rec(record)
            stimulus = {"sr": 48000, "data": np.ones(3, dtype=np.float32), "amplitude": 1.0}
            return soundcard.SoundcardAudioProcessor().sd_play_rec(record, stimulus, "unused.wav")

    device = Device()
    monkeypatch.setattr(soundcard.sd, "query_devices", device.query)
    monkeypatch.setattr(soundcard.sd, "rec", device.capture)
    monkeypatch.setattr(soundcard.sd, "playrec", device.capture)
    monkeypatch.setattr(soundcard, "save_audio_simple", device.save)
    monkeypatch.setattr(soundcard, "save_audio_with_calibration_metadata", device.save)
    monkeypatch.setattr(soundcard.SoundcardAudioProcessor, "calculate_alignment", staticmethod(device.align))
    return device


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("selection", [[2], [2, 4], [4, 2], [5], 2, np.int64(4)])
def test_full_capture_retains_selected_columns(blocking_device, mode, selection):
    device = blocking_device
    selected = [int(selection)] if isinstance(selection, (int, np.integer)) else selection
    code, data = device.run(mode, input_channels=selection, channels=99)
    expected = device.raw[:, selected] if mode == "record" else device.raw[:, selected[0]]
    assert code == error_code.OK
    np.testing.assert_array_equal(data, expected)
    assert data.shape == expected.shape
    assert device.queries == [(18, "input")]
    assert len(device.captures) == 1
    if mode == "playrec":
        np.testing.assert_array_equal(device.saved[0], expected)
        np.testing.assert_array_equal(device.alignments[0], expected)
    else:
        assert device.saved == []


@pytest.mark.parametrize(
    "params,record_input,playrec_input,playrec_device",
    [
        ({"device": {"index": 0}}, 0, 0, (0, None)),
        ({"device": 18}, 18, 18, (18, None)),
        ({"device": {"index": 18}, "output_device": {"index": 18}}, 18, 18, 18),
        ({"device": {"index": 18}, "output_device": 16}, 18, 18, (18, 16)),
        ({"device": None, "input_device": 12}, 12, 12, (12, None)),
        ({"device": None, "output_device": 16}, None, None, (None, 16)),
        ({"device": None}, None, None, None),
        ({"device": 18, "input_device": 12, "output_device": 16}, 18, 12, (12, 16)),
        ({"device": 0, "input_device": 12}, 12, 12, (12, None)),
        ({"device": 18, "input_device": 0}, 18, 18, (18, None)),
        ({"device": {"index": 0}, "input_device": {"index": 12}}, 0, 12, (12, None)),
    ],
)
@pytest.mark.parametrize("mode", ["record", "playrec"])
def test_query_matches_actual_input_device(
    blocking_device, mode, params, record_input, playrec_input, playrec_device
):
    device = blocking_device
    code, _ = device.run(mode, **params)
    assert code == error_code.OK
    expected_input = record_input if mode == "record" else playrec_input
    assert device.queries == [(expected_input, "input")]
    expected_device = record_input if mode == "record" else playrec_device
    assert device.captures[0].get("device") == expected_device
    if mode == "playrec" and expected_device is None:
        assert "device" not in device.captures[0]


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("maximum", [1, np.int64(6)])
def test_current_capability_overrides_cached_device_info(blocking_device, mode, maximum):
    device = blocking_device
    device.maximum = maximum
    code, data = device.run(mode, device={"index": 18, "max_input_channels": 2})
    assert code == error_code.OK
    np.testing.assert_array_equal(data, device.raw[:, 0])
    assert device.queries == [(18, "input")]
    assert device.captures[0]["channels"] == maximum


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("selection", [[-1], [6], [2.0], [True], [np.bool_(False)], 2.0, True])
def test_invalid_selection_fails_before_capture(blocking_device, mode, selection):
    device = blocking_device
    with pytest.raises(ValueError) as error:
        device.run(mode, input_channels=selection)
    assert "18" in str(error.value)
    assert "input_channels" in str(error.value) or "0..5" in str(error.value)
    assert device.queries == [(18, "input")]
    assert device.captures == device.saved == device.alignments == []


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("maximum", [0, -1, None, 6.0, True, np.bool_(True)])
def test_invalid_capability_fails_before_capture(blocking_device, mode, maximum):
    device = blocking_device
    device.maximum = maximum
    with pytest.raises(ValueError, match="18.*max_input_channels"):
        device.run(mode)
    assert device.queries == [(18, "input")]
    assert device.captures == device.saved == device.alignments == []


@pytest.mark.parametrize("mode", ["record", "playrec"])
def test_missing_capability_fails_before_capture(monkeypatch, blocking_device, mode):
    device = blocking_device

    def query(device=None, kind=None):
        blocking_device.queries.append((device, kind))
        return {}

    monkeypatch.setattr(soundcard.sd, "query_devices", query)
    with pytest.raises(ValueError, match="18.*max_input_channels"):
        device.run(mode)
    assert device.queries == [(18, "input")]
    assert device.captures == device.saved == device.alignments == []


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("boundary", ["query", "capture"])
def test_hardware_error_propagates_unchanged(blocking_device, mode, boundary):
    device = blocking_device
    original = RuntimeError("audio hardware failure")
    setattr(device, boundary + "_error", original)
    with pytest.raises(RuntimeError) as error:
        device.run(mode)
    assert error.value is original
    assert device.queries == [(18, "input")]
    assert len(device.captures) == (1 if boundary == "capture" else 0)
    assert device.saved == device.alignments == []


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("selection", [None, []])
@pytest.mark.parametrize("count", [1, 2, np.int64(2)])
def test_default_selection_keeps_compatible_shape(blocking_device, mode, selection, count):
    device = blocking_device
    code, data = device.run(mode, input_channels=selection, channels=count)
    expected = device.raw[:, :count] if mode == "record" and count > 1 else device.raw[:, 0]
    assert code == error_code.OK
    np.testing.assert_array_equal(data, expected)
    assert data.shape == expected.shape
    assert device.captures[0]["channels"] == 6
    if mode == "playrec":
        np.testing.assert_array_equal(device.saved[0], expected)


@pytest.mark.parametrize("mode", ["record", "playrec"])
@pytest.mark.parametrize("count", [1, 2])
def test_omitted_selection_keeps_compatible_shape(blocking_device, mode, count):
    device = blocking_device
    code, data = device.run(mode, channels=count)
    expected = device.raw[:, :count] if mode == "record" and count > 1 else device.raw[:, 0]
    assert code == error_code.OK
    np.testing.assert_array_equal(data, expected)
    assert data.shape == expected.shape
    assert device.captures[0]["channels"] == 6
    if mode == "playrec":
        np.testing.assert_array_equal(device.saved[0], expected)


@pytest.mark.parametrize("selection", ["omitted", None, []])
def test_nonblocking_default_prefix_retains_live_capture_buffer(monkeypatch, selection):
    raw = np.zeros((3, 6), dtype=np.float32)
    calls = []
    monkeypatch.setattr(
        soundcard.sd, "query_devices",
        lambda device=None, kind=None: {"max_input_channels": 6},
    )

    def capture(**kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(soundcard.sd, "rec", capture)
    record = {"sample_rate": 48000, "num_frames": 3, "channels": 2, "blocking": False}
    if selection != "omitted":
        record["input_channels"] = selection

    code, data = soundcard.SoundcardAudioProcessor.sd_rec(record)

    assert code == error_code.OK
    assert calls[0]["channels"] == 6
    assert calls[0]["blocking"] is False
    assert data.shape == (3, 2)
    assert np.shares_memory(data, raw)
    raw[:] = 7
    np.testing.assert_array_equal(data, raw[:, :2])


@pytest.mark.parametrize("count", [0, -1, 7, 1.0, True, np.bool_(False), None])
@pytest.mark.parametrize("selection", [None, []])
def test_invalid_default_retained_count_fails_before_capture(blocking_device, count, selection):
    device = blocking_device
    with pytest.raises(ValueError, match="18.*1..6"):
        device.run("record", input_channels=selection, channels=count)
    assert device.queries == [(18, "input")]
    assert device.captures == device.saved == device.alignments == []


@pytest.mark.parametrize("mode", ["record", "playrec"])
def test_invalid_sample_rate_precedes_device_query(blocking_device, mode):
    device = blocking_device
    device.query_error = AssertionError("query must not run")
    code, _ = device.run(mode, sr=0, sample_rate=0)
    assert code == error_code.INVALID_RECORD
    assert device.queries == device.captures == device.saved == device.alignments == []
