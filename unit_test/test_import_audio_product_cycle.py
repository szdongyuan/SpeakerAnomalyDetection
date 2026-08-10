
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
        self.left_panel = SimpleNamespace(
            set_condition_result=lambda *_args, **_kwargs: None,
            set_current_stage=lambda *_args, **_kwargs: None,
            set_final_result=lambda *_args, **_kwargs: None,
        )

    def _clear_data_struct(self):
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None

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


def test_decode_failure_warns_and_clears_failed_import_state():
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
    assert host.recorded_path is None
    assert host.recorded_signal_info == {}
    assert host.data_struct.store_wave_data is None
    assert host.data_struct.store_wave_data_multi is None
    assert host.data_btn.enabled is False
    assert host.replayer_btn.enabled is False
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

    host.run(show_windows=False)

    assert host.data_struct.analysis_result_dict == {"SPL": (True, 0.0)}
    assert host.product_results == ["OK"]
    assert host.excel_calls == 2


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
        )
        self._active_product_condition_key = ""
        self._active_input_channels = [0, 1]
        self.clear_calls = 0
        self.plotted = None

    def clear_all_direction_waveforms(self):
        self.clear_calls += 1

    def plot_waveform_to_workspace(self, wave, sample_rate):
        self.plotted = (np.asarray(wave), sample_rate)


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
