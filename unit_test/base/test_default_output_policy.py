"""Default-output policy exercised without PortAudio or user configuration."""
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from base import sound_device_manager as module
from consts import error_code
import numpy as np

from base.playback_controller import PlaybackController
from base.soundcard_audio_processor import SoundcardAudioProcessor


@pytest.fixture
def backend(monkeypatch):
    class Defaults:
        hostapi = 2
        device = [7, 1]  # Application overrides must not select the output.

        @property
        def _default_device(self):
            raise AssertionError("Output lookup must not resolve the input default")

    state = SimpleNamespace(output=2)
    devices = {2: {"index": 2, "name": "B", "max_output_channels": 2},
               3: {"index": 3, "name": "C", "max_output_channels": 1}}
    fake = SimpleNamespace(default=Defaults(),
        query_hostapis=Mock(side_effect=lambda index: {"default_output_device": state.output}),
        query_devices=Mock(side_effect=lambda index: devices[index]),
        _terminate=Mock(), _initialize=Mock())
    monkeypatch.setattr(module, "sd", fake)
    return state, fake, devices


def test_system_output_ignores_override_and_observes_next_default(backend):
    state, fake, devices = backend
    manager = module.SoundDeviceManager()
    assert manager.get_default_device("speaker", refresh=False) == (error_code.OK, devices[2])
    state.output = 3
    assert manager.get_default_device("speaker", refresh=False) == (error_code.OK, devices[3])
    assert fake.default.device == [7, 1]
    fake._terminate.assert_not_called()
    fake._initialize.assert_not_called()


@pytest.mark.parametrize("missing", [-1, None])
def test_no_default_output_never_queries_application_default(backend, missing):
    state, fake, _ = backend
    state.output = missing
    assert module.SoundDeviceManager().get_default_device("speaker", refresh=False) == (
        error_code.MISSING_HARDWARE_DEVICE, None)
    fake.query_devices.assert_not_called()


def test_input_setters_never_write_legacy_output(backend):
    _, fake, _ = backend
    module.SoundDeviceManager.change_default_device(9, 99)
    assert fake.default.device == [9, 1]
    module.SoundDeviceManager.change_default_output_device(99)
    assert fake.default.device == [9, 1]
    module.SoundDeviceManager.change_default_input_device(8)
    assert fake.default.device == [8, 1]


@pytest.fixture
def real_convenience_backend(monkeypatch, tmp_path):
    """Keep real play/playrec/stop; fake only hardware and completion."""
    sd = module.sd
    monkeypatch.setattr(sd, "_last_callback", None)
    monkeypatch.setattr(sd, "_terminate", Mock())
    monkeypatch.setattr(sd, "_initialize", Mock())
    monkeypatch.setattr(module.SoundDeviceManager, "get_default_device", Mock(
        return_value=(error_code.OK, {"index": 2, "max_output_channels": 2})))
    path = tmp_path / "source.wav"
    path.touch()
    controller = PlaybackController()
    monkeypatch.setattr(controller, "_load_playback_audio", lambda _: (np.ones((4, 2)), 8000))
    monkeypatch.setattr(controller, "_monitor_playback_done", lambda *args: None)
    processor = SoundcardAudioProcessor()
    monkeypatch.setattr(processor, "calculate_alignment", lambda *args: 0)
    from base import soundcard_audio_processor
    monkeypatch.setattr(soundcard_audio_processor, "save_audio_simple", Mock())
    stimulus = {"data": np.ones(4), "amplitude": 1, "sr": 8000, "device": 1}

    def invoke(entry):
        if entry == "file":
            return controller.start_audio_playback(str(path), device=1)
        if entry == "play":
            return processor.sd_play(stimulus)
        if entry == "rec":
            return processor.sd_rec({"device": 7, "num_frames": 4, "blocking": False})
        return processor.sd_play_rec({"device": 7, "prepare_frames": 0, "prolong_frames": 0},
                                     stimulus, "unused.wav")

    return sd, controller, invoke


def convenience_stream():
    stream = SimpleNamespace(closed=False, active=True)
    stream.stop = Mock(side_effect=lambda *a, **kw: setattr(stream, "active", False))
    stream.close = Mock(side_effect=lambda *a, **kw: setattr(stream, "closed", True))
    return stream


@pytest.mark.parametrize("entry", ["file", "play", "playrec", "rec"])
def test_real_convenience_calls_preserve_unrelated_recorder(real_convenience_backend, monkeypatch, entry):
    sd, controller, invoke = real_convenience_backend
    recorder = convenience_stream()
    sd._last_callback = SimpleNamespace(stream=recorder)
    opening = Mock(side_effect=RuntimeError("device removed while opening"))
    monkeypatch.setattr(sd, "OutputStream", opening)
    monkeypatch.setattr(sd, "Stream", opening)
    monkeypatch.setattr(sd, "InputStream", opening)
    code, message = invoke(entry)
    recorder.stop.assert_not_called()
    recorder.close.assert_not_called()
    opening.assert_not_called()
    assert code == (error_code.INVALID_RECORD if entry == "rec" else error_code.INVALID_PLAY)
    assert "another audio operation" in message
    assert not controller.is_audio_playing()
    sd._terminate.assert_not_called()
    sd._initialize.assert_not_called()


def test_real_file_retry_preserves_recorder_started_after_open_failure(real_convenience_backend, monkeypatch):
    sd, controller, invoke = real_convenience_backend
    recorder = convenience_stream()

    def fail_open(**kwargs):
        sd._last_callback = SimpleNamespace(stream=recorder)
        raise RuntimeError("stereo stream rejected")

    opening = Mock(side_effect=fail_open)
    monkeypatch.setattr(sd, "OutputStream", opening)
    code, message = invoke("file")
    recorder.stop.assert_not_called()
    recorder.close.assert_not_called()
    assert opening.call_count == 1
    assert code == error_code.INVALID_PLAY and "another audio operation" in message
    assert controller._playback_stream is None
    assert controller.get_current_playing_file() is None
    assert not controller.is_audio_playing()


@pytest.mark.parametrize("owned_previous", [False, True])
def test_real_file_mono_retry_allows_no_prior_or_owned_stream(real_convenience_backend, monkeypatch, owned_previous):
    sd, controller, invoke = real_convenience_backend
    if owned_previous:
        previous = convenience_stream()
        controller._playback_stream = previous
        sd._last_callback = SimpleNamespace(stream=previous)
    current = convenience_stream()
    current.start = Mock()

    def open_stream(**kwargs):
        if kwargs["channels"] == 2:
            raise RuntimeError("stereo rejected")
        return current

    opening = Mock(side_effect=open_stream)
    monkeypatch.setattr(sd, "OutputStream", opening)
    assert invoke("file")[0] == error_code.OK
    assert [(c.kwargs["device"], c.kwargs["channels"]) for c in opening.call_args_list] == [(2, 2), (2, 1)]
    assert controller._playback_stream is current
    assert controller.is_audio_playing()
    current.start.assert_called_once()
    if owned_previous:
        assert previous.closed


@pytest.mark.parametrize("entry", ["play", "playrec"])
def test_real_convenience_calls_allow_no_prior_stream(real_convenience_backend, monkeypatch, entry):
    sd, _, invoke = real_convenience_backend
    current = convenience_stream()

    def open_stream(**kwargs):
        def complete():
            if entry == "playrec":
                inputs, outputs = kwargs["channels"]
                kwargs["callback"](np.zeros((4, inputs)), np.zeros((4, outputs)),
                                   4, None, sd.CallbackFlags())
            else:
                kwargs["callback"](np.zeros((4, kwargs["channels"])), 4, None, sd.CallbackFlags())
            kwargs["finished_callback"]()

        current.start = Mock(side_effect=complete)
        return current

    opening = Mock(side_effect=open_stream)
    monkeypatch.setattr(sd, "OutputStream", opening)
    monkeypatch.setattr(sd, "Stream", opening)
    assert invoke(entry)[0] == error_code.OK
    assert opening.call_args.kwargs["device"] == (2 if entry == "play" else (7, 2))
    current.start.assert_called_once()
    assert current.closed


@pytest.mark.parametrize("entry, guard_number", [("file", 1), ("play", 1), ("playrec", 1), ("file", 2)])
def test_project_recording_cannot_enter_between_output_guard_and_start_or_retry(
        real_convenience_backend, monkeypatch, entry, guard_number):
    sd, _, invoke = real_convenience_backend
    guarded, resume, recording_done = threading.Event(), threading.Event(), threading.Event()
    output_result, record_result = [], []
    original_guard = module.SoundDeviceManager.output_would_interrupt
    calls = 0

    def paused_guard(owned_stream=None):
        nonlocal calls
        answer = original_guard(owned_stream)
        if threading.current_thread() is output_thread:
            calls += 1
            if calls == guard_number:
                guarded.set()
                assert resume.wait(3), "test did not release output guard"
        return answer

    monkeypatch.setattr(module.SoundDeviceManager, "output_would_interrupt", paused_guard)
    opening = Mock(side_effect=RuntimeError("stream rejected"))
    monkeypatch.setattr(sd, "OutputStream", opening)
    monkeypatch.setattr(sd, "Stream", opening)
    recorder = convenience_stream()

    def input_stream(**kwargs):
        recorder.start = Mock(side_effect=lambda: kwargs["callback"](
            np.zeros((4, 1), dtype=np.float32), 4, None, sd.CallbackFlags()))
        return recorder

    recording_open = Mock(side_effect=input_stream)
    monkeypatch.setattr(sd, "InputStream", recording_open)
    output_thread = threading.Thread(target=lambda: output_result.append(invoke(entry)))

    def record():
        record_result.append(SoundcardAudioProcessor.sd_rec(
            {"device": 7, "num_frames": 4, "blocking": False}))
        recording_done.set()

    recording_thread = threading.Thread(target=record)
    output_thread.start()
    try:
        assert guarded.wait(3)
        recording_thread.start()
        assert recording_done.wait(1), "competing recording must return promptly"
    finally:
        resume.set()
        output_thread.join(3)
        if recording_thread.ident is not None:
            recording_thread.join(3)
    assert not output_thread.is_alive() and not recording_thread.is_alive()
    recorder.stop.assert_not_called()
    recorder.close.assert_not_called()
    recording_open.assert_not_called()
    assert record_result[0][0] == error_code.INVALID_RECORD
    assert "busy" in record_result[0][1].lower()
    assert output_result[0][0] == error_code.INVALID_PLAY
    assert opening.call_count == (2 if entry == "file" else 1)


@pytest.mark.parametrize("entry", ["file", "play", "playrec"])
def test_project_output_returns_busy_during_blocking_record_start(real_convenience_backend, monkeypatch, entry):
    sd, _, invoke = real_convenience_backend
    started, finish = threading.Event(), threading.Event()
    record_result = []
    recorder = convenience_stream()

    def input_stream(**kwargs):
        def start():
            started.set()
            assert finish.wait(3)
            kwargs["callback"](np.zeros((4, 1), dtype=np.float32), 4, None, sd.CallbackFlags())
            kwargs["finished_callback"]()
        recorder.start = Mock(side_effect=start)
        return recorder

    monkeypatch.setattr(sd, "InputStream", input_stream)
    opening = Mock(side_effect=RuntimeError("output rejected"))
    monkeypatch.setattr(sd, "OutputStream", opening)
    monkeypatch.setattr(sd, "Stream", opening)
    recording_thread = threading.Thread(target=lambda: record_result.append(
        SoundcardAudioProcessor.sd_rec({"device": 7, "num_frames": 4})))
    recording_thread.start()
    try:
        assert started.wait(3)
        module.SoundDeviceManager.get_default_device.assert_not_called()
        code, message = invoke(entry)
        assert code == error_code.INVALID_PLAY and "busy" in message.lower()
        opening.assert_not_called()
        recorder.stop.assert_not_called()
        recorder.close.assert_not_called()
    finally:
        finish.set()
        recording_thread.join(3)
    assert not recording_thread.is_alive()
    assert record_result[0][0] == error_code.OK


@pytest.mark.parametrize("previous_entry", ["play", "rec"])
@pytest.mark.parametrize("next_entry", ["file", "play", "playrec", "rec"])
def test_completed_nonblocking_stream_allows_next_project_operation(
        real_convenience_backend, monkeypatch, previous_entry, next_entry):
    sd, _, invoke = real_convenience_backend
    opened = []

    def factory(kind):
        def open_stream(**kwargs):
            stream = convenience_stream()

            def complete():
                channels = kwargs["channels"]
                if kind == "duplex":
                    kwargs["callback"](np.zeros((4, channels[0])), np.zeros((4, channels[1])),
                                       4, None, sd.CallbackFlags())
                else:
                    kwargs["callback"](np.zeros((4, channels), dtype=np.float32),
                                       4, None, sd.CallbackFlags())
                stream.active = False
                kwargs["finished_callback"]()

            stream.start = Mock(side_effect=complete)
            opened.append(stream)
            return stream
        return open_stream

    monkeypatch.setattr(sd, "OutputStream", factory("output"))
    monkeypatch.setattr(sd, "InputStream", factory("input"))
    monkeypatch.setattr(sd, "Stream", factory("duplex"))
    if previous_entry == "play":
        result = SoundcardAudioProcessor.sd_play({"data": np.ones(4), "amplitude": 1,
                                                 "sr": 8000, "blocking": False})
    else:
        result = SoundcardAudioProcessor.sd_rec({"device": 7, "num_frames": 4, "blocking": False})
    assert result[0] == error_code.OK
    previous = sd.get_stream()
    assert not previous.active and not previous.closed
    assert invoke(next_entry)[0] == error_code.OK
    assert len(opened) == 2
    assert previous.closed
