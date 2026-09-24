import mock
import numpy as np
import pytest

from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


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
