
import copy
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = (
        _ConcurrentRotatingFileHandler
    )
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from ui.sequence.channel_plot_workspace import (
    ChannelPlotSubWindow,
    ChannelPlotWorkspace,
)
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadResult,
    WavCalibrationMetadataReadStatus,
)


@pytest.fixture(autouse=True)
def _default_wav_metadata_is_absent():
    with patch(
        "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
        return_value=WavCalibrationMetadataReadResult(
            WavCalibrationMetadataReadStatus.ABSENT,
            None,
        ),
    ):
        yield


class _ButtonSpy:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)


class _ImportCycleHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "IMPORT_AUDIO",
                        "detail": {"sample_rate": 44100},
                    }
                }
            }
        ]
        self.analysis_config = {"auto_analysis": False}
        self.product_test_pdf_report_config = {"enabled": False}
        self.product_test_condition_configs = [
            {
                "key": "01",
                "condition_name": "6000",
                "test_queue": "queue_6000",
            }
        ]
        self._active_product_condition_key = "01"
        self._active_product_condition_config = {
            "condition_name": "6000"
        }
        self._manual_product_condition_group_id = "group-import"
        self._displayed_manual_product_condition_group_id = "group-import"
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._condition_record_cache = {
            "00": {"recorded_path": "already-completed.wav"}
        }
        self._waveform_display_override_direction = "01"
        self._current_trigger_direction = "01"
        self._record_workflow_busy = False
        self.default_logger = logging.getLogger(__name__)
        self._current_recent_session_id = "old-recording"
        self.recent_test_session_by_id = {
            "old-recording": {"recorded_path": "old.wav"}
        }
        self.recorded_path = "old.wav"
        self.recorded_signal_info = {"file_path": "old.wav"}
        self.data_struct = SimpleNamespace(
            store_wave_data=np.asarray([1.0], dtype=np.float32),
            store_wave_data_multi=np.asarray([[1.0]], dtype=np.float32),
            sample_rate=48000,
            audio_lenth=1,
            analysis_result_dict={},
            wav_calibration_metadata={"recorded_channels": [{"wav_channel_index": 0}]},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
            clear_data=self._clear_data_struct,
        )
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.data_btn = _ButtonSpy()
        self.replayer_btn = _ButtonSpy()
        self.lineedit_type = SimpleNamespace(text=lambda: "S004-1")
        self.lineedit_s_or_n = SimpleNamespace(
            text=lambda: (_ for _ in ()).throw(
                AssertionError("导入音频不应读取 S/N 输入框")
            )
        )
        self.run_calls = []
        self.advance_calls = 0
        self.clear_calls = 0
        self.player_refreshes = 0
        self.pdf_calls = []
        self.condition_result_calls = []
        self.stage_calls = []
        self.left_panel = SimpleNamespace(
            set_condition_result=lambda *args, **kwargs: self.condition_result_calls.append(
                (args, kwargs)
            ),
            set_current_stage=lambda *args, **kwargs: self.stage_calls.append(
                (args, kwargs)
            ),
            set_final_result=lambda *_args, **_kwargs: None,
        )

    def _clear_data_struct(self):
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None
        self.data_struct.wav_calibration_metadata = None
        self.data_struct.wav_calibration_metadata_authoritative = False
        self.data_struct.wav_calibration_warning_shown = False

    def _decode_audio_file(self, file_path, sample_rate=None, **_kwargs):
        return np.asarray([0.1, -0.1], dtype=np.float32), 44100

    def _apply_audio_to_data_struct(self, audio_mono, sample_rate):
        audio_mono = np.asarray(audio_mono, dtype=np.float32)
        self.data_struct.store_wave_data = audio_mono
        self.data_struct.store_wave_data_multi = audio_mono.reshape(-1, 1)
        self.data_struct.sample_rate = sample_rate
        self.data_struct.audio_lenth = audio_mono.size
        self._condition_record_cache["01"] = {
            "recorded_path": self.recorded_path,
            "recorded_signal_info": dict(self.recorded_signal_info),
            "session_id": "",
        }

    def run(self, show_windows=True, **kwargs):
        self.run_calls.append((show_windows, kwargs))
        self.data_struct.analysis_result_dict = {"SPL": (True, 0.0)}
        self._manual_product_condition_results["01"] = "OK"

    def _advance_manual_product_condition_cycle_after_recording(self):
        self.advance_calls += 1
        self._active_product_condition_key = ""

    def _clear_plot_area(self):
        self.clear_calls += 1
        self._condition_record_cache.pop("01", None)

    def update_player_btn_is_paused(self):
        self.player_refreshes += 1

    def _maybe_export_product_test_pdf(self, group_id, overall_result):
        self.pdf_calls.append((group_id, overall_result))

    def _try_export_product_test_pdf(self, group_id):
        self.pdf_calls.append((group_id, "try"))


def test_product_condition_import_caches_result_and_advances_once():
    host = _ImportCycleHost()

    with patch(
        "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
        return_value=("D:/audio/6000.wav", "WAV Files (*.wav)"),
    ):
        assert host.import_audio_and_analyze()

    assert host.run_calls == [
        (
            True,
            {"capture_product_report": False},
        )
    ]
    assert host.advance_calls == 1
    assert host._manual_product_condition_completed_keys == {"01"}
    assert host._condition_record_cache["00"]["recorded_path"] == (
        "already-completed.wav"
    )
    imported = host._condition_record_cache["01"]
    assert imported["source_type"] == "imported"
    assert imported["group_id"] == "group-import"
    assert imported["recorded_path"] == "D:/audio/6000.wav"
    assert imported["result_label"] == "OK"
    assert imported["barcode"] == ""
    assert imported["recorded_signal_info"]["barcode"] is None
    assert imported["session_id"] == ""
    assert host.recent_test_session_by_id == {
        "old-recording": {"recorded_path": "old.wav"}
    }


def test_cancelled_product_condition_import_does_not_advance():
    host = _ImportCycleHost()

    with patch(
        "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ):
        assert not host.import_audio_and_analyze()

    assert host.advance_calls == 0
    assert host.run_calls == []
    assert host._active_product_condition_key == ""
    assert host._condition_record_cache["00"]["recorded_path"] == (
        "already-completed.wav"
    )


def test_decode_failure_warns_and_restores_previous_import_state():
    host = _ImportCycleHost()
    host._decode_audio_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("音频文件为空")
    )

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/empty.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
        ) as warning,
    ):
        assert not host.import_audio_and_analyze()

    warning.assert_called_once()
    assert host.advance_calls == 0
    assert host.recorded_path == "old.wav"
    assert host.recorded_signal_info == {"file_path": "old.wav"}
    np.testing.assert_array_equal(host.data_struct.store_wave_data, [1.0])
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        [[1.0]],
    )
    assert host.data_struct.wav_calibration_metadata == {
        "recorded_channels": [{"wav_channel_index": 0}]
    }
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is True
    assert host.run_calls == []
    assert host.data_btn.enabled is None
    assert host.replayer_btn.enabled is None
    assert host._condition_record_cache["00"]["recorded_path"] == (
        "already-completed.wav"
    )


def test_analysis_exception_preserves_import_and_keeps_condition_index():
    host = _ImportCycleHost()

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("analysis failed")

    host.run = fail_analysis

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/6000.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
        ) as warning,
    ):
        assert not host.import_audio_and_analyze()

    warning.assert_called_once()
    assert "分析失败" in warning.call_args.args[2]
    assert host.advance_calls == 0
    assert host._active_product_condition_key == "01"
    assert host._condition_record_cache["01"]["recorded_path"] == (
        "D:/audio/6000.wav"
    )
    assert host.recorded_path == "D:/audio/6000.wav"
    assert host.recorded_signal_info["source_type"] == "imported"
    assert host.data_struct.store_wave_data is not None
    assert host.data_struct.store_wave_data_multi is not None
    assert host.data_btn.enabled is True


def test_auto_analysis_false_returns_failure_and_retains_imported_audio():
    host = _ImportCycleHost()
    host._active_product_condition_key = ""
    host._active_product_condition_config = None
    host.analysis_config = {"auto_analysis": True}
    host.run = Mock(return_value=False)
    metadata = {"recorded_channels": [{"wav_channel_index": 0}]}

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/retry.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                WavCalibrationMetadataReadStatus.VALID,
                metadata["recorded_channels"],
            ),
        ),
    ):
        assert host.import_audio_and_analyze() is False

    host.run.assert_called_once_with(
        show_windows=True,
        capture_product_report=False,
    )
    assert host.recorded_path == "D:/audio/retry.wav"
    assert host.data_struct.store_wave_data is not None
    assert host.data_struct.store_wave_data_multi is not None
    assert host.data_struct.wav_calibration_metadata == metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_btn.enabled is True


def test_product_condition_analysis_false_aborts_without_capture_or_advance():
    host = _ImportCycleHost()
    host.run = Mock(return_value=False)
    host._capture_imported_product_condition_record = Mock()
    host._complete_imported_product_condition_step = Mock()
    metadata = {"recorded_channels": [{"wav_channel_index": 0}]}

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/retry-product.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                WavCalibrationMetadataReadStatus.VALID,
                metadata["recorded_channels"],
            ),
        ),
    ):
        assert host.import_audio_and_analyze() is False

    host._capture_imported_product_condition_record.assert_not_called()
    host._complete_imported_product_condition_step.assert_not_called()
    assert host.advance_calls == 0
    assert host._manual_product_condition_completed_keys == set()
    assert host._active_product_condition_key == ""
    assert host._active_product_condition_config is None
    assert host.condition_result_calls[-1][0][:2] == ("01", "待检测")
    assert host.recorded_path == "D:/audio/retry-product.wav"
    assert host.data_struct.store_wave_data is not None
    assert host.data_struct.store_wave_data_multi is not None
    assert host.data_struct.wav_calibration_metadata == metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True


@pytest.mark.parametrize("run_result", [True, None])
def test_product_condition_true_and_legacy_none_keep_completion_path(run_result):
    host = _ImportCycleHost()
    original_run = host.run

    def run_with_result(*args, **kwargs):
        original_run(*args, **kwargs)
        return run_result

    host.run = run_with_result

    with patch(
        "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
        return_value=("D:/audio/6000.wav", "WAV Files (*.wav)"),
    ):
        assert host.import_audio_and_analyze() is True

    assert host.advance_calls == 1
    assert host._manual_product_condition_completed_keys == {"01"}
    assert host._condition_record_cache["01"]["source_type"] == "imported"


class _AnalysisInstance:
    _sequence_analysis_key = "spl"
    _channel_mismatch = False
    _channel_mismatch_info = None

    def __init__(self, host):
        self.host = host

    def calculate_spl(self):
        self.host.data_struct.analysis_result_dict["SPL"] = (True, 0.0)
        return True

    def hide(self):
        return None


class _AIJudgmentAnalysisInstance:
    _sequence_analysis_key = "ai"
    _channel_mismatch = False
    _channel_mismatch_info = None

    def __init__(self, host, label):
        self.host = host
        self.label = label
        self.result = None
        self.export_detail = {}

    def calculate_ai_scores(self, *_args):
        if self.label not in ("OK", "NG"):
            return
        is_ok = self.label == "OK"
        self.host.data_struct.analysis_result_dict["AI"] = (is_ok, 0.2)
        self.result = self.label
        self.export_detail = {"label": self.label}

    def hide(self):
        return None


class _RuleJudgmentAnalysisInstance(_AnalysisInstance):
    def __init__(self, host, is_ok):
        super().__init__(host)
        self.is_ok = is_ok

    def calculate_spl(self):
        self.host.data_struct.analysis_result_dict["SPL"] = (self.is_ok, 0.0)
        return True


class _ImportAnalysisHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "IMPORT_AUDIO",
                        "detail": {"sample_rate": 44100},
                    }
                }
            }
        ]
        self.analysis_config = {
            "display_sequence": ["spl"],
            "spl": {"type": "SPL"},
        }
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self.count_board = SimpleNamespace(
            mode="test",
            set_test_result_file=lambda _label: (_ for _ in ()).throw(
                AssertionError("导入分析不应写生产计数")
            ),
            set_test_text=lambda: None,
        )
        self._active_product_condition_key = "01"
        self._manual_product_condition_results = {}
        self._current_recent_session_id = "old-recording"
        self.recorded_signal_info = {
            "source_type": "imported",
            "labels": "not_labeled",
        }
        self.data_btn = _ButtonSpy()
        self.replayer_btn = _ButtonSpy()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self.excel_calls = 0
        self.product_results = []

    def screen(self):
        size = SimpleNamespace(width=lambda: 1600, height=lambda: 900)
        return SimpleNamespace(size=lambda: size)

    def instance_analysis_class(self, _key, _type, _params):
        self.analysis_window.append(_AnalysisInstance(self))

    def _capture_excel_export_cache(self):
        self.excel_calls += 1

    def _maybe_export_excel_results(self):
        self.excel_calls += 1

    def _can_output_ok_ng(self):
        return True, ""

    def _summarize_ok_ng(self):
        return True, "OK"

    def _sync_left_panel_analysis_details(self, _state):
        return None

    def _is_directional_cycle_active(self):
        return False

    def _update_manual_product_condition_result_after_analysis(self, label):
        self.product_results.append(label)
        self._manual_product_condition_results["01"] = label
        return "OK"

    def _persist_current_test_audio_label(self, *_args, **_kwargs):
        raise AssertionError("导入分析不应移动或改标 WAV")

    def _finalize_test_run(self, *_args, **_kwargs):
        raise AssertionError("导入分析不应执行生产测试收口")

    def update_player_btn_is_paused(self):
        return None

    def _capture_analysis_report_failure(self, *_args):
        raise AssertionError("本测试不应产生分析异常")


def test_import_analysis_keeps_outputs_but_skips_production_side_effects():
    host = _ImportAnalysisHost()

    with patch(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
    ):
        host.run(show_windows=False)

    assert host.data_struct.analysis_result_dict == {"SPL": (True, 0.0)}
    assert host.product_results == ["OK"]
    assert host.excel_calls == 2


class _CombinedJudgmentHost(_ImportAnalysisHost):
    _summarize_ok_ng = SequenceWidgetStreamingOpsMixin._summarize_ok_ng
    _can_output_ok_ng = SequenceWidgetStreamingOpsMixin._can_output_ok_ng

    def __init__(self, ai_label, threshold_ok):
        super().__init__()
        self.ai_label = ai_label
        self.threshold_ok = threshold_ok
        self.analysis_config = {
            "display_sequence": ["ai", "spl"],
            "ai": {"type": "AI", "analyse_model_name": "demo"},
            "spl": {"type": "SPL", "limit_checked": True},
        }

    def instance_analysis_class(self, key, _type, _params):
        if key == "ai":
            self.analysis_window.append(
                _AIJudgmentAnalysisInstance(self, self.ai_label)
            )
        else:
            self.analysis_window.append(
                _RuleJudgmentAnalysisInstance(self, self.threshold_ok)
            )

    def _update_manual_product_condition_result_after_analysis(self, label):
        self.product_results.append(label)
        return label


def test_ai_and_rule_results_use_the_same_overall_judgment():
    scenarios = [
        ("OK", False, "NG"),
        ("NG", True, "NG"),
        ("OK", True, "OK"),
        (None, True, "OK"),
    ]

    for ai_label, threshold_ok, expected in scenarios:
        host = _CombinedJudgmentHost(ai_label, threshold_ok)

        with patch(
            "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
        ):
            host.run(show_windows=False)

        assert host.product_results == [expected]


class _RecordedAudioLoadHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
                        "detail": {"sample_rate": 48000},
                    }
                }
            }
        ]
        self.data_struct = SimpleNamespace(
            store_wave_data=None,
            store_wave_data_multi=None,
            sample_rate=0,
            audio_lenth=0,
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        )
        self._active_product_condition_key = ""
        self._active_input_channels = [0, 1]
        self.clear_calls = 0
        self.plotted = None
        self.plot_mapping = None

    def clear_all_direction_waveforms(self):
        self.clear_calls += 1

    def plot_waveform_to_workspace(
        self,
        wave,
        sample_rate,
        *,
        channel_mapping=None,
    ):
        self.plotted = (np.asarray(wave), sample_rate)
        self.plot_mapping = (
            None
            if channel_mapping is None
            else tuple(channel_mapping)
        )


def test_recorded_audio_loader_preserves_multichannel_shape():
    host = _RecordedAudioLoadHost()
    channels_first = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    with patch(
        "ui.sequence.sequence_widget_analysis_ops.librosa.load",
        return_value=(channels_first, 48000),
    ) as load:
        host._load_audio_file_to_data_struct("D:/audio/recorded.wav")

    load.assert_called_once_with(
        "D:/audio/recorded.wav",
        sr=48000,
        mono=False,
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        channels_first.T,
    )
    np.testing.assert_allclose(
        host.data_struct.store_wave_data,
        channels_first.T.mean(axis=1),
    )
    assert host._active_input_channels == [0, 1]
    assert host.clear_calls == 1
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is False
    assert host.data_struct.wav_calibration_warning_shown is False


def test_imported_audio_loader_replaces_metadata_and_preserves_channels():
    host = _RecordedAudioLoadHost()
    host.sequence_config[0]["seq1"]["acq"]["mode"] = "IMPORT_AUDIO"
    host.default_logger = logging.getLogger(__name__)
    host.data_struct.wav_calibration_metadata = {"old": True}
    host.data_struct.wav_calibration_metadata_authoritative = True
    host.data_struct.wav_calibration_warning_shown = True
    channels_first = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    metadata = {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            },
            {
                "wav_channel_index": 1,
                "v2pa_factor": None,
                "standard_spl": None,
                "calibrated": False,
            },
        ]
    }

    def decode(*_args, **_kwargs):
        assert host.data_struct.wav_calibration_metadata == {"old": True}
        assert host.data_struct.wav_calibration_metadata_authoritative is True
        assert host.data_struct.wav_calibration_warning_shown is True
        return channels_first, 48000

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.librosa.load",
            side_effect=decode,
        ) as load,
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                WavCalibrationMetadataReadStatus.VALID,
                metadata["recorded_channels"],
            ),
        ) as inspect_metadata,
    ):
        host._load_audio_file_to_data_struct("D:/audio/imported.wav")

    load.assert_called_once_with(
        "D:/audio/imported.wav",
        sr=48000,
        mono=False,
    )
    inspect_metadata.assert_called_once_with(
        "D:/audio/imported.wav",
        logger=host.default_logger,
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        channels_first.T,
    )
    assert host._active_input_channels == [0, 1]
    assert host.data_struct.wav_calibration_metadata == metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is False


def test_recent_session_view_restores_imported_wav_metadata_state():
    from unit_test.test_recent_session_view import _DummySequenceWidget

    host = _DummySequenceWidget()
    original_metadata = {"recorded_channels": [{"wav_channel_index": 0}]}
    host.data_struct.wav_calibration_metadata = original_metadata
    host.data_struct.wav_calibration_metadata_authoritative = True
    host.data_struct.wav_calibration_warning_shown = True
    original_skip = object()
    host._analysis_preflight_warning_shown = True
    host._analysis_preflight_skips = {"current": original_skip}
    host._analysis_channel_local_columns = {"current": 1}
    host._imported_wav_channel_v2pa_factors = {"current": 2.5}
    original_loader = host._load_audio_file_to_data_struct

    def replace_audio(file_path, sample_rate=None, **kwargs):
        original_loader(file_path, sample_rate)
        host.data_struct.wav_calibration_metadata = {"history": True}
        host.data_struct.wav_calibration_metadata_authoritative = False
        host.data_struct.wav_calibration_warning_shown = False
        host._analysis_preflight_warning_shown = False
        host._analysis_preflight_skips = {"history": object()}
        host._analysis_channel_local_columns = {"history": 0}
        host._imported_wav_channel_v2pa_factors = {"history": 7.0}

    host._load_audio_file_to_data_struct = replace_audio

    host._show_recent_session_analysis_by_id("recent_1")

    assert host.data_struct.wav_calibration_metadata == original_metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is True
    assert host._analysis_preflight_warning_shown is True
    assert host._analysis_preflight_skips == {"current": original_skip}
    assert host._analysis_channel_local_columns == {"current": 1}
    assert host._imported_wav_channel_v2pa_factors == {"current": 2.5}


class _DirectImportHost(_ImportCycleHost):
    _decode_audio_file = SequenceWidgetAnalysisOpsMixin._decode_audio_file
    _apply_audio_to_data_struct = SequenceWidgetAnalysisOpsMixin._apply_audio_to_data_struct

    def __init__(self):
        super().__init__()
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self.analysis_config = {"auto_analysis": False}
        self._active_input_channels = [7]
        self._waveform_presentation_owner = "hardware"
        self.plotted = None
        self.plot_mapping = None

    def plot_waveform_to_workspace(
        self,
        wave,
        sample_rate,
        *,
        channel_mapping=None,
    ):
        self.plotted = (np.asarray(wave), sample_rate)
        self.plot_mapping = (
            None
            if channel_mapping is None
            else tuple(channel_mapping)
        )


def _metadata_result(status, channels=None):
    return WavCalibrationMetadataReadResult(
        status,
        None if channels is None else {"recorded_channels": channels},
    )


def _metadata_channel(wav_index, physical=...):
    channel = {
        "wav_channel_index": wav_index,
        "v2pa_factor": None,
        "standard_spl": None,
        "calibrated": False,
    }
    if physical is not ...:
        channel["physical_input_channel"] = physical
    return channel


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _RealDirectImportWorkspaceHost(
    _DirectImportHost,
    SequenceWidgetStreamingOpsMixin,
):
    plot_waveform_to_workspace = (
        SequenceWidgetStreamingOpsMixin.plot_waveform_to_workspace
    )

    def __init__(self):
        super().__init__()
        self.channel_workspace = ChannelPlotWorkspace()
        self.channel_workspace.set_channels((7,))


def test_direct_import_rebuilds_real_workspace_before_committing_mapping(
    qapp,
    monkeypatch,
):
    host = _RealDirectImportWorkspaceHost()
    channels_first = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    render_events = []
    original_set_data = ChannelPlotSubWindow.set_data

    def observe_set_data(window, x_data, y_data):
        render_events.append(
            (
                window.channel_index,
                tuple(host._active_input_channels),
                host._waveform_presentation_owner,
                np.asarray(y_data).copy(),
            )
        )
        return original_set_data(window, x_data, y_data)

    monkeypatch.setattr(ChannelPlotSubWindow, "set_data", observe_set_data)
    try:
        with (
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
                return_value=("D:/audio/imported.wav", "WAV Files (*.wav)"),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.librosa.load",
                return_value=(channels_first, 44100),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
            ),
        ):
            assert host.import_audio_and_analyze()

        assert [
            window.channel_index
            for window in host.channel_workspace.all_subwindows()
        ] == [0, 1]
        assert [event[0] for event in render_events] == [0, 1]
        assert all(event[1] == (7,) for event in render_events)
        assert all(event[2] == "hardware" for event in render_events)
        np.testing.assert_allclose(render_events[0][3], channels_first[0])
        np.testing.assert_allclose(render_events[1][3], channels_first[1])
        assert host._active_input_channels == [0, 1]
        assert host._waveform_presentation_owner == "direct_import"
    finally:
        host.channel_workspace.close()
        qapp.processEvents()


def test_direct_import_real_plot_failure_restores_previous_workspace(
    qapp,
    monkeypatch,
):
    host = _RealDirectImportWorkspaceHost()
    previous_multi = host.data_struct.store_wave_data_multi.copy()
    previous_window = host.channel_workspace.all_subwindows()[0]
    previous_x = np.asarray([0.0], dtype=np.float32)
    previous_y = np.asarray([0.75], dtype=np.float32)
    previous_window.set_data(previous_x, previous_y)
    original_set_data = ChannelPlotSubWindow.set_data

    def fail_second_column(window, x_data, y_data):
        if window.channel_index == 1:
            raise RuntimeError("second column plot failed")
        return original_set_data(window, x_data, y_data)

    monkeypatch.setattr(ChannelPlotSubWindow, "set_data", fail_second_column)
    try:
        with (
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
                return_value=("D:/audio/replacement.wav", "WAV Files (*.wav)"),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.librosa.load",
                return_value=(
                    np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                    44100,
                ),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
            ) as warning,
        ):
            assert host.import_audio_and_analyze() is False

        warning.assert_called_once()
        assert host._active_input_channels == [7]
        assert host._waveform_presentation_owner == "hardware"
        np.testing.assert_array_equal(
            host.data_struct.store_wave_data_multi,
            previous_multi,
        )
        restored_windows = host.channel_workspace.all_subwindows()
        assert [window.channel_index for window in restored_windows] == [7]
        restored_x, restored_y = restored_windows[0].snapshot_plot_state()
        np.testing.assert_array_equal(restored_x, previous_x)
        np.testing.assert_array_equal(restored_y, previous_y)
    finally:
        host.channel_workspace.close()
        qapp.processEvents()


def test_direct_import_preserves_all_channels_and_reads_metadata_once():
    host = _DirectImportHost()
    channels_first = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/imported.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.librosa.load",
            return_value=(channels_first, 44100),
        ) as load,
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                WavCalibrationMetadataReadStatus.ABSENT
            ),
            create=True,
        ) as inspect_metadata,
    ):
        assert host.import_audio_and_analyze()

    load.assert_called_once_with(
        "D:/audio/imported.wav",
        sr=44100,
        mono=False,
    )
    inspect_metadata.assert_called_once_with(
        "D:/audio/imported.wav",
        logger=host.default_logger,
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        channels_first.T,
    )
    np.testing.assert_allclose(
        host.data_struct.store_wave_data,
        channels_first.T.mean(axis=1),
    )
    assert host.data_struct.sample_rate == 44100
    assert host.data_struct.audio_lenth == 3
    assert host._active_input_channels == [0, 1]
    assert host.plot_mapping == (0, 1)
    assert host._waveform_presentation_owner == "direct_import"
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is False


def test_recent_audio_loader_uses_exact_saved_mapping_for_valid_old_metadata():
    host = _RecordedAudioLoadHost()
    host.default_logger = logging.getLogger(__name__)
    channels_first = np.asarray(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float32,
    )
    diagnostic = _metadata_result(
        WavCalibrationMetadataReadStatus.VALID,
        [_metadata_channel(0), _metadata_channel(1)],
    )

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.librosa.load",
            return_value=(channels_first, 48000),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=diagnostic,
        ),
    ):
        host._load_audio_file_to_data_struct(
            "D:/audio/recent.wav",
            saved_active_input_channels=[1, 3],
            presentation_owner="recent_view",
        )

    assert host._active_input_channels == [1, 3]
    assert host.plot_mapping == (1, 3)
    assert host._waveform_presentation_owner == "recent_view"


def test_direct_import_uses_complete_physical_metadata_in_wav_column_order():
    host = _DirectImportHost()
    channels_first = np.asarray(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float32,
    )
    diagnostic = _metadata_result(
        WavCalibrationMetadataReadStatus.VALID,
        [_metadata_channel(1, 4), _metadata_channel(0, 2)],
    )

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/mapped.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.librosa.load",
            return_value=(channels_first, 44100),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=diagnostic,
            create=True,
        ) as inspect_metadata,
    ):
        assert host.import_audio_and_analyze()

    inspect_metadata.assert_called_once_with(
        "D:/audio/mapped.wav",
        logger=host.default_logger,
    )
    assert host._active_input_channels == [2, 4]
    assert host.plot_mapping == (2, 4)
    assert host._waveform_presentation_owner == "direct_import"


def test_invalid_authoritative_import_restores_prior_presentation_and_data():
    host = _DirectImportHost()
    previous_multi = host.data_struct.store_wave_data_multi.copy()
    previous_mono = host.data_struct.store_wave_data.copy()
    previous_metadata = host.data_struct.wav_calibration_metadata
    host.plotted = (np.asarray([9.0], dtype=np.float32), 48000)
    previous_plot = host.plotted

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/invalid.wav", "WAV Files (*.wav)"),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.librosa.load",
            return_value=(np.asarray([[0.1, 0.2]], dtype=np.float32), 44100),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                WavCalibrationMetadataReadStatus.INVALID
            ),
            create=True,
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
        ) as warning,
    ):
        assert host.import_audio_and_analyze() is False

    warning.assert_called_once()
    assert "导入音频失败" in warning.call_args.args[2]
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        previous_multi,
    )
    np.testing.assert_array_equal(host.data_struct.store_wave_data, previous_mono)
    assert host.data_struct.wav_calibration_metadata == previous_metadata
    assert host._active_input_channels == [7]
    assert host.plotted is previous_plot
    assert host._waveform_presentation_owner == "hardware"


class _ImportWorkspaceSpy:
    def __init__(self):
        self.channel_calls = []
        self.clear_count = 0

    def set_channels(self, channels):
        self.channel_calls.append(tuple(channels))

    def clear_plots(self):
        self.clear_count += 1


class _DirectImportRefreshHost(_DirectImportHost):
    refresh_channel_windows = SequenceWidgetUiOpsMixin.refresh_channel_windows
    _apply_input_channel_workspace_mapping = (
        SequenceWidgetUiOpsMixin._apply_input_channel_workspace_mapping
    )
    _end_direct_import_presentation = (
        SequenceWidgetUiOpsMixin._end_direct_import_presentation
    )

    def __init__(self):
        super().__init__()
        self.mic_channels = [0, 2]
        self.mic = {"max_input_channels": 4}
        self._configured_input_channels = (0, 2)
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self.channel_workspace = _ImportWorkspaceSpy()


def test_hardware_refresh_ends_direct_import_ownership_and_clears_plots():
    host = _DirectImportRefreshHost()
    host._active_input_channels = [2, 4]
    host._waveform_presentation_owner = "direct_import"

    host.refresh_channel_windows()

    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


class _RecentOwnershipHost(SequenceWidgetAnalysisOpsMixin):
    refresh_channel_windows = SequenceWidgetUiOpsMixin.refresh_channel_windows
    _apply_input_channel_workspace_mapping = (
        SequenceWidgetUiOpsMixin._apply_input_channel_workspace_mapping
    )

    def __init__(self, *, fail_load=False, refresh_during_load=False):
        from unit_test.test_recent_session_view import _DummySequenceWidget

        base = _DummySequenceWidget()
        self.__dict__.update(base.__dict__)
        self._session_record = {
            "recorded_signal_info": {},
            "recorded_path": "history.wav",
            "sample_rate": 48000,
            "config_snapshot": {
                "sequence_config": copy.deepcopy(self.sequence_config),
                "analysis_config": copy.deepcopy(self.analysis_config),
                "using_config_path": self.using_config_path,
                "active_input_channels": [3],
            },
        }
        self.default_logger = logging.getLogger(__name__)
        self._configured_input_channels = (0,)
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self._waveform_presentation_owner = "hardware"
        self.mic_channels = [0]
        self.mic = {"max_input_channels": 5}
        self.channel_workspace = _ImportWorkspaceSpy()
        self._fail_load = fail_load
        self._refresh_during_load = refresh_during_load
        self.loader_kwargs = None
        self.warning_messages = []

    def _resolve_recent_session(self, session_id):
        return self._session_record if session_id == "recent_1" else None

    def _resolve_recent_session_path(self, _session_record):
        return "history.wav"

    def _close_analysis_windows(self):
        return None

    def _load_audio_file_to_data_struct(self, file_path, sample_rate=None, **kwargs):
        self.loader_kwargs = {
            "saved_active_input_channels": getattr(
                self,
                "_pending_recent_saved_active_input_channels",
                None,
            ),
            "presentation_owner": getattr(
                self,
                "_pending_recent_presentation_owner",
                None,
            ),
        }
        self.data_struct.store_wave_data = np.asarray([3.0, 4.0], dtype=np.float32)
        self.data_struct.store_wave_data_multi = np.asarray(
            [[3.0], [4.0]], dtype=np.float32
        )
        self.data_struct.sample_rate = sample_rate
        self.data_struct.audio_lenth = 2
        self._active_input_channels = [3]
        if self._refresh_during_load:
            self.mic_channels = [0, 2]
            self.refresh_channel_windows()
        if self._fail_load:
            raise RuntimeError("recent load failed")

    def run(self, *args, **kwargs):
        return None

    def plot_waveform_to_workspace(self, data, sample_rate, *, channel_mapping=None):
        self.plot_calls.append(
            (
                np.asarray(data).copy(),
                sample_rate,
                None if channel_mapping is None else tuple(channel_mapping),
            )
        )

    def _clear_plot_area(self):
        self.channel_workspace.clear_plots()


class _RealRecentWorkspaceHost(
    _RecentOwnershipHost,
    SequenceWidgetStreamingOpsMixin,
):
    _decode_audio_file = SequenceWidgetAnalysisOpsMixin._decode_audio_file
    _apply_audio_to_data_struct = (
        SequenceWidgetAnalysisOpsMixin._apply_audio_to_data_struct
    )
    _load_audio_file_to_data_struct = (
        SequenceWidgetAnalysisOpsMixin._load_audio_file_to_data_struct
    )
    plot_waveform_to_workspace = (
        SequenceWidgetStreamingOpsMixin.plot_waveform_to_workspace
    )

    def init_fft_and_stft_flag(self):
        return None

    def __init__(self):
        super().__init__()
        self._session_record["config_snapshot"]["active_input_channels"] = [2, 4]
        self._active_input_channels = [7]
        self.channel_workspace = ChannelPlotWorkspace()
        self.channel_workspace.set_channels((7,))


def test_recent_view_rebuilds_real_workspace_before_committing_mapping(
    qapp,
    monkeypatch,
):
    host = _RealRecentWorkspaceHost()
    channels_first = np.asarray(
        [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        dtype=np.float32,
    )
    render_events = []
    original_set_data = ChannelPlotSubWindow.set_data

    def observe_set_data(window, x_data, y_data):
        if window.channel_index in (2, 4):
            render_events.append(
                (
                    window.channel_index,
                    tuple(host._active_input_channels),
                    host._waveform_presentation_owner,
                    np.asarray(y_data).copy(),
                )
            )
        return original_set_data(window, x_data, y_data)

    monkeypatch.setattr(ChannelPlotSubWindow, "set_data", observe_set_data)
    diagnostic = _metadata_result(
        WavCalibrationMetadataReadStatus.VALID,
        [_metadata_channel(0), _metadata_channel(1)],
    )
    try:
        with (
            patch(
                "ui.sequence.sequence_widget_analysis_ops.librosa.load",
                return_value=(channels_first, 48000),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
                return_value=diagnostic,
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
            ) as warning,
        ):
            host._show_recent_session_analysis_by_id("recent_1")

        warning.assert_not_called()
        assert [event[0] for event in render_events] == [2, 4]
        assert all(event[1] == (7,) for event in render_events)
        assert all(event[2] == "recent_view" for event in render_events)
        np.testing.assert_allclose(render_events[0][3], channels_first[0])
        np.testing.assert_allclose(render_events[1][3], channels_first[1])
        assert host._active_input_channels == [7]
        assert host._waveform_presentation_owner == "hardware"
        assert [
            window.channel_index
            for window in host.channel_workspace.all_subwindows()
        ] == [7]
    finally:
        host.channel_workspace.close()
        qapp.processEvents()


def test_recent_view_restores_prior_presentation_without_pending_refresh():
    host = _RecentOwnershipHost()
    previous_multi = host.data_struct.store_wave_data_multi.copy()

    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host._show_recent_session_analysis_by_id("recent_1")

    assert host.loader_kwargs == {
        "saved_active_input_channels": [3],
        "presentation_owner": "recent_view",
    }
    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0]
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, previous_multi)
    np.testing.assert_array_equal(host.plot_calls[-1][0], previous_multi)


@pytest.mark.parametrize("fail_load", [False, True])
def test_recent_view_pending_hardware_refresh_wins_once_on_exit(fail_load):
    host = _RecentOwnershipHost(
        fail_load=fail_load,
        refresh_during_load=True,
    )

    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host._show_recent_session_analysis_by_id("recent_1")

    assert host._waveform_presentation_owner == "hardware"
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


def test_replacement_import_decode_failure_preserves_previous_state():
    host = _DirectImportHost()
    host._analysis_preflight_warning_shown = True
    host._analysis_preflight_skips = {"old": object()}
    host._analysis_channel_local_columns = {"old": 1}
    host._imported_wav_channel_v2pa_factors = {"old": 2.5}

    def fail_decode(*_args, **_kwargs):
        assert host.data_struct.wav_calibration_metadata == {
            "recorded_channels": [{"wav_channel_index": 0}]
        }
        assert host.data_struct.wav_calibration_metadata_authoritative is True
        assert host.data_struct.wav_calibration_warning_shown is True
        raise ValueError("decode failed")

    host._decode_audio_file = fail_decode

    with (
        patch(
            "ui.sequence.sequence_widget_analysis_ops.QFileDialog.getOpenFileName",
            return_value=("D:/audio/replacement.wav", "WAV Files (*.wav)"),
        ),
        patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"),
    ):
        assert not host.import_audio_and_analyze()

    assert host.data_struct.wav_calibration_metadata == {
        "recorded_channels": [{"wav_channel_index": 0}]
    }
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is True
    assert host._analysis_preflight_warning_shown is True
    assert set(host._analysis_preflight_skips) == {"old"}
    assert host._analysis_channel_local_columns == {"old": 1}
    assert host._imported_wav_channel_v2pa_factors == {"old": 2.5}
