import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from base import stimulus_resolver


def _install_ui_import_shims():
    import base.load_audio

    if not hasattr(base.load_audio, "load_audio_preserve_rate"):
        base.load_audio.load_audio_preserve_rate = lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError("load_audio_preserve_rate is outside this focused test worktree")
        )

    if "base.audio_sample_rate" in sys.modules:
        return

    module_name = "base.audio_sample_rate"

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != module_name:
            raise

    def _device_rate(device):
        try:
            return int(device.get("samplerate"))
        except (AttributeError, TypeError, ValueError):
            return None

    def resolve_duplex_sample_rate(mic, speaker):
        mic_rate = _device_rate(mic)
        speaker_rate = _device_rate(speaker)
        if mic_rate and speaker_rate and mic_rate == speaker_rate:
            return SimpleNamespace(ok=True, sample_rate=mic_rate, message="")
        return SimpleNamespace(ok=False, sample_rate=None, message="采样率不一致")

    def resolve_input_sample_rate(mic):
        mic_rate = _device_rate(mic)
        if mic_rate:
            return SimpleNamespace(ok=True, sample_rate=mic_rate, message="")
        return SimpleNamespace(ok=False, sample_rate=None, message="未找到麦克风采样率")

    def resolve_output_sample_rate(speaker):
        speaker_rate = _device_rate(speaker)
        if speaker_rate:
            return SimpleNamespace(ok=True, sample_rate=speaker_rate, message="")
        return SimpleNamespace(ok=False, sample_rate=None, message="未找到扬声器采样率")

    module = types.ModuleType("base.audio_sample_rate")
    module.resolve_duplex_sample_rate = resolve_duplex_sample_rate
    module.resolve_input_sample_rate = resolve_input_sample_rate
    module.resolve_output_sample_rate = resolve_output_sample_rate
    sys.modules["base.audio_sample_rate"] = module


def _chirp_detail(config_rate=44100):
    return {
        "stimulus_info": {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "sample_rate": config_rate,
            "start_freq": 100,
            "stop_freq": 1000,
            "total_time": 0.01,
            "repeat_times": 1,
            "amplitude": 0.5,
            "voltage": 1.0,
            "voltage_type": "RMS",
        },
        "sample_rate": config_rate,
    }


def test_ui_import_shim_prefers_importable_real_audio_sample_rate(monkeypatch):
    module_name = "base.audio_sample_rate"
    missing = object()
    original_module = sys.modules.pop(module_name, missing)
    original_import_module = importlib.import_module
    real_module = types.ModuleType("base.audio_sample_rate")
    real_module.real_audio_sample_rate_module = True

    def fake_import_module(name):
        if name != module_name:
            return original_import_module(name)
        sys.modules[name] = real_module
        return real_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        _install_ui_import_shims()

        assert sys.modules[module_name] is real_module
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not missing:
            sys.modules[module_name] = original_module


def test_ui_import_shim_fallback_includes_output_sample_rate(monkeypatch):
    module_name = "base.audio_sample_rate"
    missing = object()
    original_module = sys.modules.pop(module_name, missing)
    original_import_module = importlib.import_module

    def fake_import_module(name):
        if name == module_name:
            raise ModuleNotFoundError(name=module_name)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        _install_ui_import_shims()

        module = sys.modules[module_name]
        assert hasattr(module, "resolve_duplex_sample_rate")
        assert hasattr(module, "resolve_input_sample_rate")
        assert hasattr(module, "resolve_output_sample_rate")
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not missing:
            sys.modules[module_name] = original_module


def test_ui_import_shim_reraises_internal_audio_sample_rate_module_not_found(monkeypatch):
    module_name = "base.audio_sample_rate"
    missing = object()
    original_module = sys.modules.pop(module_name, missing)
    original_import_module = importlib.import_module
    internal_error = ModuleNotFoundError("internal dependency unavailable", name="base.audio_backend")

    def fake_import_module(name):
        if name == module_name:
            raise internal_error
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            _install_ui_import_shims()

        assert exc_info.value is internal_error
        assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not missing:
            sys.modules[module_name] = original_module


def test_ui_import_shim_reraises_internal_audio_sample_rate_import_error(monkeypatch):
    module_name = "base.audio_sample_rate"
    missing = object()
    original_module = sys.modules.pop(module_name, missing)
    original_import_module = importlib.import_module
    internal_error = ImportError("internal dependency unavailable", name="base.audio_backend")

    def fake_import_module(name):
        if name == module_name:
            raise internal_error
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        with pytest.raises(ImportError) as exc_info:
            _install_ui_import_shims()

        assert exc_info.value is internal_error
        assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not missing:
            sys.modules[module_name] = original_module


def test_ui_import_shim_reraises_plain_audio_sample_rate_import_error(monkeypatch):
    module_name = "base.audio_sample_rate"
    missing = object()
    original_module = sys.modules.pop(module_name, missing)
    original_import_module = importlib.import_module
    import_error = ImportError("module raised import error", name=module_name)

    def fake_import_module(name):
        if name == module_name:
            raise import_error
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        with pytest.raises(ImportError) as exc_info:
            _install_ui_import_shims()

        assert exc_info.value is import_error
        assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not missing:
            sys.modules[module_name] = original_module


def test_generate_and_save_uses_runtime_sample_rate_not_config(monkeypatch, tmp_path):
    monkeypatch.setattr(stimulus_resolver.model_consts, "STORED_STIMULUS_PATH", str(tmp_path))
    detail = _chirp_detail(config_rate=44100)

    data, sample_rate, _ = stimulus_resolver.generate_and_save_stimulus(
        detail,
        runtime_sample_rate=48000,
    )

    assert sample_rate == 48000
    assert detail["stimulus_info"]["sample_rate"] == 48000
    assert len(data) == int(0.01 * 48000)


def test_set_data_struct_requires_runtime_sample_rate():
    with pytest.raises(TypeError):
        stimulus_resolver.set_data_struct_stimulus_signal(SimpleNamespace(), _chirp_detail())


def test_existing_wav_loads_with_runtime_sample_rate(monkeypatch):
    calls = []

    def fake_load(path, sr):
        calls.append((path, sr))
        return np.zeros(3), np.arange(3)

    monkeypatch.setattr(stimulus_resolver, "load_audio_simple", fake_load)
    monkeypatch.setattr(stimulus_resolver.os.path, "exists", lambda path: True)
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_signal_path"] = "existing.wav"
    data_struct = SimpleNamespace()

    stimulus_resolver.set_data_struct_stimulus_signal(
        data_struct,
        detail,
        runtime_sample_rate=48000,
    )

    assert calls[-1][1] == 48000
    assert data_struct.sample_rate == 48000


def test_temporary_analysis_reference_does_not_write_back(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    before = {k: v.copy() if isinstance(v, dict) else v for k, v in detail.items()}
    data_struct = SimpleNamespace()

    stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        runtime_sample_rate=32000,
    )

    assert data_struct.sample_rate == 32000
    assert data_struct.stimulus_info["sample_rate"] == 32000
    assert len(data_struct.stimulus_data) == int(0.01 * 32000)
    assert detail == before


def test_analysis_reference_resolves_config_relative_external_wav(monkeypatch, tmp_path):
    calls = []
    config_dir = tmp_path / "sequences"
    config_dir.mkdir()
    config_path = config_dir / "sequence.json"
    reference_path = config_dir / "refs" / "reference.wav"
    detail = _chirp_detail(config_rate=44100)
    detail["load_stimulus_signal_path"] = "refs/reference.wav"
    data_struct = SimpleNamespace()

    def fake_exists(path):
        return path.replace("\\", "/") == str(reference_path).replace("\\", "/")

    def fake_load(path, sr):
        calls.append((path.replace("\\", "/"), sr))
        return np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32)

    monkeypatch.setattr(stimulus_resolver.os.path, "exists", fake_exists)
    monkeypatch.setattr(stimulus_resolver, "load_audio_simple", fake_load)

    assert stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        using_config_path=str(config_path),
        runtime_sample_rate=32000,
    ) is True

    assert calls == [(str(reference_path).replace("\\", "/"), 32000)]
    assert data_struct.sample_rate == 32000
    assert np.array_equal(data_struct.stimulus_data, np.arange(4, dtype=np.float32))


def test_analysis_reference_uses_load_stimulus_path_as_external_authority(monkeypatch, tmp_path):
    calls = []
    config_dir = tmp_path / "sequences"
    config_dir.mkdir()
    config_path = config_dir / "sequence.json"
    artifact_path = config_dir / "refs" / "generated-artifact.wav"
    reference_path = config_dir / "refs" / "external-reference.wav"
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_signal_path"] = "refs/generated-artifact.wav"
    detail["load_stimulus_signal_path"] = "refs/external-reference.wav"
    data_struct = SimpleNamespace()

    existing = {
        str(artifact_path).replace("\\", "/"),
        str(reference_path).replace("\\", "/"),
    }

    def fake_exists(path):
        return path.replace("\\", "/") in existing

    def fake_load(path, sr):
        calls.append((path.replace("\\", "/"), sr))
        return np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32)

    monkeypatch.setattr(stimulus_resolver.os.path, "exists", fake_exists)
    monkeypatch.setattr(stimulus_resolver, "load_audio_simple", fake_load)

    assert stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        using_config_path=str(config_path),
        runtime_sample_rate=32000,
    ) is True

    assert calls == [(str(reference_path).replace("\\", "/"), 32000)]
    assert data_struct.sample_rate == 32000
    assert np.array_equal(data_struct.stimulus_data, np.arange(5, dtype=np.float32))


def test_analysis_reference_external_wav_updates_runtime_metadata(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_info"]["total_time"] = 0.5
    detail["load_stimulus_signal_path"] = "refs/external-reference.wav"
    data_struct = SimpleNamespace()

    monkeypatch.setattr(stimulus_resolver.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        stimulus_resolver,
        "load_audio_simple",
        lambda path, sr: (np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32)),
    )

    assert stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        runtime_sample_rate=32000,
    ) is True

    assert data_struct.sample_rate == 32000
    assert data_struct.stimulus_info["sample_rate"] == 32000
    assert data_struct.stimulus_info["total_time"] == pytest.approx(7 / 32000)
    assert detail["stimulus_info"]["sample_rate"] == 44100
    assert detail["stimulus_info"]["total_time"] == 0.5


def test_analysis_reference_ignores_generated_stimulus_artifact_and_regenerates(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    detail["stimulus_signal_path"] = "missing-generated-artifact.wav"
    before = {k: v.copy() if isinstance(v, dict) else v for k, v in detail.items()}
    data_struct = SimpleNamespace()

    monkeypatch.setattr(stimulus_resolver.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        stimulus_resolver,
        "load_audio_simple",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generated artifact must not be loaded")),
    )

    assert stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        runtime_sample_rate=32000,
    ) is True

    assert data_struct.sample_rate == 32000
    assert data_struct.stimulus_info["sample_rate"] == 32000
    assert len(data_struct.stimulus_data) == int(0.01 * 32000)
    assert detail == before


def test_analysis_reference_missing_configured_external_wav_fails_without_generation(monkeypatch):
    detail = _chirp_detail(config_rate=44100)
    detail["load_stimulus_signal_path"] = "missing-reference.wav"
    data_struct = SimpleNamespace()

    monkeypatch.setattr(stimulus_resolver.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        stimulus_resolver,
        "_generate_stimulus_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate substitute reference")),
    )

    assert stimulus_resolver.set_data_struct_analysis_reference_signal(
        data_struct,
        detail,
        runtime_sample_rate=32000,
    ) is False
    assert not hasattr(data_struct, "stimulus_data")

def test_sequence_init_play_and_record_resolves_duplex_runtime_rate(monkeypatch):
    _install_ui_import_shims()
    from ui.sequence import sequence_widget

    calls = []

    def fake_set_data_struct_stimulus_signal(
        data_struct,
        detail,
        using_config_path=None,
        *,
        runtime_sample_rate,
        logger=None,
    ):
        calls.append((data_struct, detail, using_config_path, runtime_sample_rate))
        return False

    monkeypatch.setattr(
        sequence_widget.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(fake_set_data_struct_stimulus_signal),
    )
    win = SimpleNamespace(
        sequence_config=[
            {"seq1": {"acq": {"mode": "PLAY_AND_RECORD", "detail": _chirp_detail(config_rate=44100)}}}
        ],
        data_struct=SimpleNamespace(sample_rate=None),
        using_config_path="sequence.json",
        mic={"samplerate": 48000},
        speaker={"samplerate": 48000},
    )

    sequence_widget.SequenceWindow.init_data_struct_stimulus_config(win)

    assert win.data_struct.sample_rate == 48000
    assert calls[-1][3] == 48000


def test_sequence_init_play_and_record_mismatch_clears_stale_runtime_stimulus_state(monkeypatch):
    _install_ui_import_shims()
    from ui.sequence import sequence_widget

    calls = []
    win = SimpleNamespace(
        sequence_config=[
            {"seq1": {"acq": {"mode": "PLAY_AND_RECORD", "detail": _chirp_detail(config_rate=44100)}}}
        ],
        data_struct=SimpleNamespace(
            sample_rate=48000,
            stimulus_data=np.ones(8, dtype=np.float32),
            stimulus_info={"stimulus_method": "chirp", "amplitude": 0.5, "sample_rate": 48000},
            alignment_sample_count=3,
            clear_data=lambda: None,
        ),
        using_config_path="sequence.json",
        mic={"samplerate": 44100, "hostapi": 1, "name": "mic"},
        speaker={"samplerate": 48000, "hostapi": 1, "name": "speaker"},
        lineedit_type=SimpleNamespace(text=lambda: "model"),
        lineedit_count=SimpleNamespace(text=lambda: "1"),
        lineedit_s_or_n=SimpleNamespace(text=lambda: ""),
        _excel_export_cache="old",
        _excel_exported_record_id="old",
        mic_channels=[0],
    )
    monkeypatch.setattr(
        sequence_widget.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("out.wav", {}))
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args, **kwargs: None)

    sequence_widget.SequenceWindow.init_data_struct_stimulus_config(win)
    reset_result = sequence_widget.SequenceWindow.reset_work_pram(win, "not_labeled")

    assert calls == []
    assert win.data_struct.sample_rate is None
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert reset_result == (None, None, None)


def test_sequence_init_import_stimulus_audio_defers_reference_generation(monkeypatch):
    _install_ui_import_shims()
    from ui.sequence import sequence_widget

    calls = []
    monkeypatch.setattr(
        sequence_widget.AnalysisModelSelect,
        "set_data_struct_stimulus_signal",
        staticmethod(lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    win = SimpleNamespace(
        sequence_config=[
            {"seq1": {"acq": {"mode": "IMPORT_STIMULUS_AUDIO", "detail": _chirp_detail(config_rate=44100)}}}
        ],
        data_struct=SimpleNamespace(
            sample_rate=48000,
            stimulus_data=np.ones(8, dtype=np.float32),
            stimulus_info={"stimulus_method": "chirp", "sample_rate": 48000},
            alignment_sample_count=4,
        ),
        streaming_stimulus_data=np.ones(8, dtype=np.float32),
        using_config_path="sequence.json",
        mic={"samplerate": 48000},
        speaker={"samplerate": 48000},
    )

    sequence_widget.SequenceWindow.init_data_struct_stimulus_config(win)

    assert calls == []
    assert win.data_struct.sample_rate == 48000
    assert win.data_struct.stimulus_data is None
    assert win.data_struct.stimulus_info is None
    assert not hasattr(win.data_struct, "alignment_sample_count")
    assert win.streaming_stimulus_data is None


def _golden_sample_window(mic, speaker, data_struct=None):
    _install_ui_import_shims()
    from ui.operation_sequence import normalize_play_record_detail

    detail = normalize_play_record_detail(_chirp_detail(config_rate=44100))
    seq = SimpleNamespace(
        detail=detail,
        analysis_list={
            "display_sequence": ["fr"],
            "fr": {"type": "FR", "golden_sample_checked": True},
        },
    )
    return SimpleNamespace(
        mic=mic,
        speaker=speaker,
        using_config_path="sequence.json",
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None),
        select_list=SimpleNamespace(
            config=[seq],
            data_struct=data_struct or SimpleNamespace(sample_rate=None),
        ),
    )


def test_record_golden_sample_resolves_duplex_runtime_rate(monkeypatch):
    _install_ui_import_shims()
    from ui import operation_sequence

    calls = []
    warnings = []
    win = _golden_sample_window(
        {"index": 0, "samplerate": 48000},
        {"index": 1, "samplerate": 48000},
    )

    def fake_setup(data_struct, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        calls.append((data_struct.sample_rate, detail, using_config_path, runtime_sample_rate))
        return False

    win.set_data_struct_stimulus_signal = fake_setup
    monkeypatch.setattr(
        operation_sequence.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after setup")),
    )
    monkeypatch.setattr(
        operation_sequence.MessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(win)

    assert win.select_list.data_struct.sample_rate is None
    assert calls[-1][0] == 48000
    assert calls[-1][3] == 48000


def test_record_golden_sample_warns_on_duplex_mismatch(monkeypatch):
    _install_ui_import_shims()
    from ui import operation_sequence

    calls = []
    warnings = []
    win = _golden_sample_window(
        {"index": 0, "samplerate": 44100},
        {"index": 1, "samplerate": 48000},
    )
    win.set_data_struct_stimulus_signal = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setattr(
        operation_sequence.MessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(win)

    assert calls == []
    assert warnings
    assert "不一致" in warnings[-1][2]
