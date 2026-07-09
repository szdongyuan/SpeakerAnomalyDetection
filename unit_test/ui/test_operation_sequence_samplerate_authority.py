import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui import operation_sequence


class _FakeDialog:
    def __init__(self, captured, signal_len):
        self._captured = captured
        self._captured.append(signal_len)

    def setWindowTitle(self, _title):
        pass

    def exec_(self):
        return 0


def _selector(mic, speaker):
    seq = SimpleNamespace(
        detail={"stimulus_info": {"sample_rate": 44100}, "recording_start_delay_ms": 100.0},
        analysis_list={
            "display_sequence": ["FR"],
            "FR": {"type": "FR", "golden_sample_checked": True},
        },
    )
    data_struct = SimpleNamespace(sample_rate=None, stimulus_data=np.zeros(4), store_wave_data=None)
    return SimpleNamespace(
        select_list=SimpleNamespace(config=[seq], data_struct=data_struct),
        mic=mic,
        speaker=speaker,
        using_config_path="sequence.json",
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None),
    )


def test_record_golden_sample_uses_duplex_samplerate_not_config(monkeypatch):
    selector = _selector({"samplerate": 48000, "hostapi": 1}, {"samplerate": 48000, "hostapi": 1})
    calls = []
    monkeypatch.setattr(
        operation_sequence.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(
            lambda data_struct,
            detail,
            using_config_path=None,
            logger=None,
            runtime_sample_rate=None: calls.append(runtime_sample_rate) or True
        ),
    )
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda data_struct, recording_start_delay_ms=None: (
            {"sr": data_struct.sample_rate, "data": np.zeros(4)},
            {"sample_rate": data_struct.sample_rate},
        ),
    )
    monkeypatch.setattr(operation_sequence.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(operation_sequence.QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    monkeypatch.setattr(
        operation_sequence.SoundcardAudioProcessor,
        "sd_play_rec",
        lambda self, record, stimulus, path, calibration_metadata=None: (1, None),
    )
    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args: None)
    monkeypatch.setattr(operation_sequence.LoadUiConfig, "save_sequence_config_to_json", lambda *args: True)
    selector.set_data_struct_stimulus_signal = operation_sequence.AnalysisModelSelect.set_data_struct_stimulus_signal

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)

    assert calls == [48000]
    assert selector.select_list.data_struct.sample_rate is None


def test_record_golden_sample_blocks_mismatched_samplerates(monkeypatch):
    selector = _selector({"samplerate": 44100, "hostapi": 1}, {"samplerate": 48000, "hostapi": 1})
    warnings = []
    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(
        operation_sequence.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load stimulus"))),
    )

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)

    assert warnings


def test_record_golden_sample_passes_wav_calibration_metadata(monkeypatch):
    selector = _selector({"samplerate": 48000, "hostapi": 1, "hardware_id": "mic-1"}, {"samplerate": 48000, "hostapi": 1})
    selector.select_list.mic_channels = [2, 4]
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    build_calls = []
    record_calls = []

    monkeypatch.setattr(
        operation_sequence.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(lambda data_struct, detail, using_config_path=None, logger=None, runtime_sample_rate=None: True),
    )
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda data_struct, recording_start_delay_ms=None: (
            {"sr": data_struct.sample_rate, "data": np.zeros(4, dtype=np.float32), "amplitude": 1.0},
            {"sample_rate": data_struct.sample_rate, "sr": data_struct.sample_rate},
        ),
    )
    monkeypatch.setattr(operation_sequence.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        operation_sequence,
        "build_recording_wav_calibration_metadata",
        lambda input_channels, hardware_id=None, logger=None: build_calls.append((list(input_channels), hardware_id))
        or metadata,
    )

    def fake_sd_play_rec(self, recorded_dict, stimulus_dict, path, calibration_metadata=None):
        record_calls.append((list(recorded_dict.get("input_channels", [])), calibration_metadata))
        return 0, np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(operation_sequence.SoundcardAudioProcessor, "sd_play_rec", fake_sd_play_rec)
    monkeypatch.setattr(operation_sequence, "get_class_mapping", lambda: {})
    monkeypatch.setattr(operation_sequence.QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args: None)
    selector.set_data_struct_stimulus_signal = operation_sequence.AnalysisModelSelect.set_data_struct_stimulus_signal

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)

    assert build_calls == [([2], "mic-1")]
    assert record_calls == [([2], metadata)]


def test_record_golden_sample_cancel_save_restores_shared_runtime_state(monkeypatch):
    seeded_stimulus = np.array([0.25, 0.5], dtype=np.float32)
    seeded_wave = np.array([0.75, 0.125], dtype=np.float32)
    seeded_info = {"sample_rate": 44100, "stimulus_method": "seeded"}
    selector = _selector({"samplerate": 48000, "hostapi": 1}, {"samplerate": 48000, "hostapi": 1})
    data_struct = selector.select_list.data_struct
    data_struct.sample_rate = 44100
    data_struct.stimulus_data = seeded_stimulus
    data_struct.store_wave_data = seeded_wave
    data_struct.stimulus_info = seeded_info
    data_struct.alignment_sample_count = 17
    analysis_cfg = selector.select_list.config[0].analysis_list

    def fake_setup(data_struct, detail, using_config_path=None, logger=None, runtime_sample_rate=None):
        data_struct.sample_rate = runtime_sample_rate
        data_struct.stimulus_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data_struct.stimulus_info = {"sample_rate": runtime_sample_rate, "stimulus_method": "runtime"}
        data_struct.alignment_sample_count = 3
        return True

    monkeypatch.setattr(
        operation_sequence.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(fake_setup),
    )
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda data_struct, recording_start_delay_ms=None: (
            {
                "sr": data_struct.sample_rate,
                "data": np.zeros(3, dtype=np.float32),
                "alignment_sample_count": getattr(data_struct, "alignment_sample_count", None),
            },
            {"sample_rate": data_struct.sample_rate},
        ),
    )
    monkeypatch.setattr(operation_sequence.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        operation_sequence.SoundcardAudioProcessor,
        "sd_play_rec",
        lambda self, record, stimulus, path, calibration_metadata=None: (0, np.zeros(3, dtype=np.float32)),
    )
    monkeypatch.setattr(operation_sequence, "get_class_mapping", lambda: {})
    monkeypatch.setattr(operation_sequence.QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args: None)
    selector.set_data_struct_stimulus_signal = operation_sequence.AnalysisModelSelect.set_data_struct_stimulus_signal

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)

    assert data_struct.sample_rate == 44100
    assert data_struct.stimulus_data is seeded_stimulus
    assert data_struct.store_wave_data is seeded_wave
    assert data_struct.stimulus_info is seeded_info
    assert data_struct.alignment_sample_count == 17
    assert "golden_sample_result_path" not in analysis_cfg


def test_record_golden_sample_success_saves_runtime_payload_then_restores_shared_state(monkeypatch):
    seeded_stimulus = np.array([0.25, 0.5], dtype=np.float32)
    seeded_wave = np.array([0.75, 0.125], dtype=np.float32)
    seeded_info = {"sample_rate": 44100, "stimulus_method": "seeded"}
    runtime_info = {"sample_rate": 48000, "stimulus_method": "runtime", "total_time": 0.25}
    saved_path = Path("unit_test/ui/.golden_baseline_regression.json").resolve()
    if saved_path.exists():
        saved_path.unlink()
    selector = _selector({"samplerate": 48000, "hostapi": 1}, {"samplerate": 48000, "hostapi": 1})
    data_struct = selector.select_list.data_struct
    data_struct.sample_rate = 44100
    data_struct.stimulus_data = seeded_stimulus
    data_struct.store_wave_data = seeded_wave
    data_struct.stimulus_info = seeded_info
    data_struct.alignment_sample_count = 17
    analysis_cfg = selector.select_list.config[0].analysis_list

    def fake_setup(data_struct, detail, using_config_path=None, logger=None, runtime_sample_rate=None):
        data_struct.sample_rate = runtime_sample_rate
        data_struct.stimulus_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data_struct.stimulus_info = dict(runtime_info)
        data_struct.alignment_sample_count = 3
        return True

    monkeypatch.setattr(
        operation_sequence.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(fake_setup),
    )
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda data_struct, recording_start_delay_ms=None: (
            {
                "sr": data_struct.sample_rate,
                "data": np.zeros(3, dtype=np.float32),
                "alignment_sample_count": getattr(data_struct, "alignment_sample_count", None),
            },
            {"sample_rate": data_struct.sample_rate},
        ),
    )
    monkeypatch.setattr(operation_sequence.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        operation_sequence.SoundcardAudioProcessor,
        "sd_play_rec",
        lambda self, record, stimulus, path, calibration_metadata=None: (0, np.zeros(3, dtype=np.float32)),
    )
    monkeypatch.setattr(operation_sequence, "get_class_mapping", lambda: {})
    monkeypatch.setattr(
        operation_sequence.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(saved_path), ""),
    )
    monkeypatch.setattr(operation_sequence.MessageBox, "warning", lambda *args: None)
    selector.set_data_struct_stimulus_signal = operation_sequence.AnalysisModelSelect.set_data_struct_stimulus_signal

    try:
        operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        assert payload["sample_rate"] == 48000
        assert payload["stimulus_info"] == runtime_info
        assert data_struct.sample_rate == 44100
        assert data_struct.stimulus_data is seeded_stimulus
        assert data_struct.store_wave_data is seeded_wave
        assert data_struct.stimulus_info is seeded_info
        assert data_struct.alignment_sample_count == 17
        assert analysis_cfg["golden_sample_result_path"] == str(saved_path).replace("\\", "/")
    finally:
        if saved_path.exists():
            saved_path.unlink()


def _option_list_with_loaded_config(monkeypatch, sequence_mode, detail, mic=None, speaker=None):
    config_info = [
        {
            "seq1": {
                "acq": {
                    "name": {
                        "PLAY_AND_RECORD": "播放与录制",
                        "RECORD_ONLY": "录制音频",
                        "IMPORT_AUDIO": "导入音频",
                        "IMPORT_STIMULUS_AUDIO": "导入激励与音频",
                    }[sequence_mode],
                    "mode": sequence_mode,
                    "detail": detail,
                },
                "analysis_list": {
                    "display_sequence": ["AI1"],
                    "AI1": {"type": "AI"},
                },
            }
        }
    ]
    obj = SimpleNamespace(
        config=[],
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        mic=mic,
        speaker=speaker,
        mic_channels=[0],
        signal_len=0,
        add_config=lambda *args, **kwargs: None,
    )
    obj._normalize_record_only_detail = operation_sequence.OptionList._normalize_record_only_detail
    obj.create_config_dialog = (
        lambda model, config_manager, name, type, signal_len: operation_sequence.OptionList.create_config_dialog(
            obj, model, config_manager, name, type, signal_len
        )
    )
    monkeypatch.setattr(operation_sequence.LoadUiConfig, "load_data_from_json", lambda _path: (0, config_info))

    operation_sequence.OptionList.init_config_info(obj, "sequence.json")
    return obj


def test_record_only_ai_dialog_uses_mic_samplerate_not_stale_config_samplerate(monkeypatch):
    captured_signal_lens = []
    option_list = _option_list_with_loaded_config(
        monkeypatch,
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
    )
    monkeypatch.setattr(
        operation_sequence,
        "AIConfigWindow",
        lambda config_manager, name, signal_len, available_channels=None: _FakeDialog(
            captured_signal_lens, signal_len
        ),
    )

    operation_sequence.OptionList.show_dialog(option_list, "AI1")

    assert captured_signal_lens == [48000]


def test_play_and_record_ai_dialog_uses_duplex_samplerate_not_stale_config_samplerate(monkeypatch):
    captured_signal_lens = []
    option_list = _option_list_with_loaded_config(
        monkeypatch,
        "PLAY_AND_RECORD",
        {"sample_rate": 44100, "total_time": 1.0, "stimulus_info": {"sample_rate": 44100, "total_time": 1.0}},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
    )
    monkeypatch.setattr(
        operation_sequence,
        "AIConfigWindow",
        lambda config_manager, name, signal_len, available_channels=None: _FakeDialog(
            captured_signal_lens, signal_len
        ),
    )

    operation_sequence.OptionList.show_dialog(option_list, "AI1")

    assert captured_signal_lens == [48000]


def test_import_stimulus_audio_ai_dialog_defers_model_length_filtering(monkeypatch):
    captured_signal_lens = []
    option_list = _option_list_with_loaded_config(
        monkeypatch,
        "IMPORT_STIMULUS_AUDIO",
        {"sample_rate": 44100, "total_time": 1.0, "stimulus_info": {"sample_rate": 44100, "total_time": 1.0}},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
    )
    monkeypatch.setattr(
        operation_sequence,
        "AIConfigWindow",
        lambda config_manager, name, signal_len, available_channels=None: _FakeDialog(
            captured_signal_lens, signal_len
        ),
    )

    operation_sequence.OptionList.show_dialog(option_list, "AI1")

    assert captured_signal_lens == [0]
