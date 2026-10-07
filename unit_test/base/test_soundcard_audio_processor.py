import mock
import numpy as np
import pytest

from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code
from base.sound_device_manager import SoundDeviceManager


@pytest.fixture(autouse=True)
def default_output(monkeypatch):
    resolver = mock.Mock(return_value=(error_code.OK, {"index": 2, "hostapi": 0, "max_output_channels": 2}))
    monkeypatch.setattr(SoundDeviceManager, "get_default_device", resolver)
    return resolver


@pytest.mark.parametrize("entry", ["play", "playrec"])
def test_output_calls_follow_current_default_and_preserve_input(monkeypatch, default_output, entry):
    from base import soundcard_audio_processor as module
    default_output.side_effect = [(error_code.OK, {"index": index}) for index in (2, 3)]
    output = mock.Mock(return_value=np.ones((3, 2)))
    monkeypatch.setattr(module.sd, entry, output)
    monkeypatch.setattr(module, "save_audio_simple", mock.Mock())
    monkeypatch.setattr(module.sd, "stop", mock.Mock())
    monkeypatch.setattr(module.sd, "_terminate", mock.Mock())
    monkeypatch.setattr(module.sd, "_initialize", mock.Mock())
    processor = SoundcardAudioProcessor()
    monkeypatch.setattr(processor, "calculate_alignment", lambda *args: 0)
    stimulus = {"data": np.ones(3), "amplitude": 1, "sr": 8000, "device": 1}
    record = {"input_device": {"index": 7}, "input_channels": [1]}
    for index in (2, 3):
        result = (processor.sd_play(stimulus) if entry == "play"
                  else processor.sd_play_rec(record, stimulus, "unused.wav"))
        assert result[0] == error_code.OK
        assert output.call_args.kwargs["device"] == (index if entry == "play" else (7, index))
        if entry == "playrec":
            assert output.call_args.kwargs["channels"] == 2
            assert record["_recorded_multi"].shape == (3, 1)
    assert default_output.call_count == 2
    assert all(call == mock.call("speaker", refresh=False) for call in default_output.call_args_list)
    module.sd.stop.assert_not_called()
    module.sd._terminate.assert_not_called()
    module.sd._initialize.assert_not_called()


@pytest.mark.parametrize("entry", ["play", "playrec"])
@pytest.mark.parametrize("failure", [None, "device removed", "stream rejected", "incompatible Host APIs"])
def test_output_failure_never_falls_back_or_stops_recording(monkeypatch, default_output, entry, failure):
    from base import soundcard_audio_processor as module
    if failure is None:
        default_output.return_value = (error_code.MISSING_HARDWARE_DEVICE, None)
    output = mock.Mock(side_effect=RuntimeError(failure))
    monkeypatch.setattr(module.sd, entry, output)
    monkeypatch.setattr(module.sd, "stop", mock.Mock())
    monkeypatch.setattr(module.sd, "_terminate", mock.Mock())
    processor = SoundcardAudioProcessor()
    stimulus = {"data": np.ones(3), "amplitude": 1, "sr": 8000, "device": 1}
    record = {"device": 7, "input_channels": [0]}
    code, message = (processor.sd_play(stimulus) if entry == "play"
                     else processor.sd_play_rec(record, stimulus, "unused.wav"))
    assert code == error_code.INVALID_PLAY
    assert "default output" in message.lower()
    assert default_output.call_count == 1
    assert "_recorded_multi" not in record
    if failure is None:
        output.assert_not_called()
    else:
        assert failure in message
        assert output.call_args.kwargs["device"] == (2 if entry == "play" else (7, 2))
        assert output.call_count == 1
    module.sd.stop.assert_not_called()
    module.sd._terminate.assert_not_called()


def test_pure_recording_does_not_resolve_output(monkeypatch, default_output):
    from base import soundcard_audio_processor as module
    monkeypatch.setattr(module.sd, "rec", mock.Mock(return_value=np.ones((3, 1))))
    assert SoundcardAudioProcessor.sd_rec({"device": 7})[0] == error_code.OK
    default_output.assert_not_called()


def test_play_rec_returns_saved_pcm24_after_alignment(tmp_path, monkeypatch):
    import soundfile as sf
    from base import soundcard_audio_processor as module
    source = np.array([[9, 9], [0.5, 2], [-3, 0.5], [2**-24, -2**-24]], dtype=np.float32)
    original = source.copy()
    monkeypatch.setattr(module.sd, "playrec", lambda *a, **kw: source)
    processor = SoundcardAudioProcessor()
    monkeypatch.setattr(processor, "calculate_alignment", lambda *a: 1)
    record = {"input_channels": [0, 1]}
    path = tmp_path / "aligned.wav"
    code, mono = processor.sd_play_rec(record, {"data": np.ones(3), "amplitude": 1, "sr": 8000}, str(path))
    decoded, rate = sf.read(path, dtype="float32", always_2d=True)
    expected = np.array([[0.5, 8388607 / 8388608], [-1, 0.5], [0, -1 / 8388608]], dtype=np.float32)
    assert code == error_code.OK
    assert rate == 8000
    assert sf.info(path).subtype == "PCM_24"
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(record["_recorded_multi"], expected)
    np.testing.assert_array_equal(mono, decoded.mean(axis=1))
    np.testing.assert_array_equal(source, original)


def test_play_rec_failed_save_does_not_publish_multi(tmp_path, monkeypatch):
    from base import soundcard_audio_processor as module
    monkeypatch.setattr(module.sd, "playrec", lambda *a, **kw: np.ones((3, 1), dtype=np.float32))
    processor = SoundcardAudioProcessor()
    monkeypatch.setattr(processor, "calculate_alignment", lambda *a: 0)
    monkeypatch.setattr(module, "save_audio_simple", mock.Mock(side_effect=OSError("disk full")))
    record = {}
    with pytest.raises(OSError, match="disk full"):
        processor.sd_play_rec(record, {"data": np.ones(3), "amplitude": 1, "sr": 8000}, str(tmp_path / "failed.wav"))
    assert "_recorded_multi" not in record


class TestSoundcardAudioProcessor(object):

    test_path = "base.soundcard_audio_processor.SoundcardAudioProcessor"

    @pytest.mark.parametrize("corr_ret, stimulus_signal, recorded_signal, result_set", [
        (1, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], -4),
        (0, [1, 1, 1], [1, 1, 1], -2),
        (0, [], [], 1),
    ])
    @mock.patch("scipy.signal.correlate")
    def test_calculate_alignment(self, mock_corr, corr_ret, stimulus_signal, recorded_signal, result_set):
        mock_corr.return_value = corr_ret
        result = SoundcardAudioProcessor().calculate_alignment(stimulus_signal, recorded_signal)
        assert result == result_set
