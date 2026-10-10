from types import SimpleNamespace
from unittest.mock import Mock, patch

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _ButtonSpy:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)


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


class _RecordingAnalysisHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
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
        self._prepare_live_mic_calibration_batch = Mock()
        self.count_board = SimpleNamespace(
            mode="test",
            set_test_result_file=Mock(),
            set_test_text=lambda: None,
        )
        self._active_product_condition_key = "01"
        self._manual_product_condition_results = {}
        self._current_recent_session_id = ""
        self.recorded_signal_info = {
            "source_type": "recorded",
            "labels": "not_labeled",
        }
        self.data_btn = _ButtonSpy()
        self.replayer_btn = _ButtonSpy()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self.product_results = []

    def screen(self):
        size = SimpleNamespace(width=lambda: 1600, height=lambda: 900)
        return SimpleNamespace(size=lambda: size)

    def instance_analysis_class(self, _key, _type, _params):
        self.analysis_window.append(_AnalysisInstance(self))

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
        return True

    def _finalize_test_run(self, *_args, **_kwargs):
        return None

    def update_player_btn_is_paused(self):
        return None

    def _capture_analysis_report_failure(self, *_args):
        raise AssertionError("本测试不应产生分析异常")


class _CombinedJudgmentHost(_RecordingAnalysisHost):
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
