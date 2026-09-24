from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from ui.custom_ui_widget.audio_clip_extraction_dialog import AudioClipExtractionDialog


@pytest.mark.parametrize("save", [True, False])
def test_clip_handoff_matches_actual_save(tmp_path, monkeypatch, save):
    source = np.array([9, 0.5, 2, -3, 2**-24, -2**-24, 9], dtype=np.float32)
    original = source.copy()
    path = str(tmp_path / "clip.wav") if save is True else ""
    host = SimpleNamespace(selected_region_time=(1 / 8000, 6 / 8000), sample_rate=8000,
                           audio_data=source, save_clip=bool(save), save_clip_path=path,
                           accept=mock.Mock(), close=mock.Mock())
    monkeypatch.setattr("ui.custom_ui_widget.audio_clip_extraction_dialog.QFileDialog.getSaveFileName", lambda *a: ("", ""))
    AudioClipExtractionDialog.on_click_ok_btn(host)
    actual, _, length = host.return_value
    assert length == 5
    if save is True:
        decoded, _ = sf.read(path, dtype="float32")
        assert sf.info(path).subtype == "PCM_24"
        np.testing.assert_array_equal(decoded, np.array([0.5, 8388607 / 8388608, -1, 0, -1 / 8388608], dtype=np.float32))
        np.testing.assert_array_equal(actual, decoded)
    else:
        np.testing.assert_array_equal(actual, source[1:6])
    np.testing.assert_array_equal(source, original)
