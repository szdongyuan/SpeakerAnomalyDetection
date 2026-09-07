from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication

from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
from unit_test.test_manual_product_condition_cycle import _DummyManualCycleWidget


@pytest.fixture
def host():
    app = QApplication.instance() or QApplication([])
    widget = _DummyManualCycleWidget()
    for index, condition in enumerate(widget.product_test_condition_configs):
        condition["group_name"] = "A口" if index < 2 else "B口"
        condition["display_name"] = f"{condition['group_name']} / {condition['condition_name']}"
    widget.left_panel = MotorAiResultPanel(
        condition_configs=widget.product_test_condition_configs
    )
    widget.left_panel.set_channels(list(range(5)))
    widget.recorded_signal_info = {"labels": "not_labeled"}
    widget._capture_analysis_report_failure = Mock()
    yield widget
    widget.left_panel.close()
    app.processEvents()


def row_text(host, key="q6000"):
    return host.left_panel.rows[key]["labels"]["result"].text()


def test_recording_title_claims_started_only_after_process_callback(host):
    host.on_clicked_player_btn()
    assert row_text(host) == "待检测"
    assert host.left_panel.stage_label.text() == "等待开始"

    host._append_recent_session_from_current_run = Mock()
    host._begin_recent_session_for_current_run()
    assert host.left_panel.stage_label.text() == "等待开始"
    assert row_text(host) == "待检测"

    host.default_logger = Mock()
    host._recording_process_id = "request-1"
    session = SimpleNamespace(
        request=SimpleNamespace(request_id="request-1"),
        worker_pid=1234,
    )
    host._on_process_recording_started(session)

    assert host.left_panel.stage_label.text() == ""
    assert row_text(host) == "采集中"
    assert row_text(host, "q7000") == "待检测"
    assert host.left_panel.port_result_value.text() == "待判定"
    assert host.left_panel.round_result_value.text() == "待判定"


def test_next_recording_keeps_previous_automatic_analysis_in_task_title(host):
    host._manual_product_condition_index = 1
    host.on_clicked_player_btn()
    host.left_panel.set_current_stage("分析中", tone="running")
    host._analysis_has_pending_tasks = Mock(return_value=True)
    host.default_logger = Mock()
    host._recording_process_id = "request-2"
    session = SimpleNamespace(
        request=SimpleNamespace(request_id="request-2"),
        worker_pid=1234,
    )

    host._on_process_recording_started(session)

    assert host.left_panel.stage_label.text() == "分析中"
    assert row_text(host, "q7000") == "采集中"


@pytest.mark.parametrize("mode", ["mark", "test"])
def test_recorded_condition_waits_for_judgement_without_claiming_five_channels(host, mode):
    host.count_board.mode = mode
    host.on_clicked_player_btn()
    host._mark_manual_product_condition_recording_completed()
    host._advance_manual_product_condition_cycle_after_recording()

    assert row_text(host) == "待判定"
    assert host.left_panel.stage_label.text() == "等待下一档位"
    assert host.left_panel.rows["q6000"]["labels"]["progress"].text() == "通道判定：0/5"
    assert host.left_panel.port_result_value.text() == "待判定"
    assert host.left_panel.round_result_value.text() == "待判定"


def test_active_automatic_analysis_is_not_replaced_by_waiting_stage(host):
    host.count_board.mode = "test"
    host.on_clicked_player_btn()
    host._mark_manual_product_condition_recording_completed()
    host.left_panel.set_current_stage("A口 / 6000 分析中", tone="running")
    host._analysis_has_pending_tasks = Mock(return_value=True)

    host._advance_manual_product_condition_cycle_after_recording()

    assert host.left_panel.stage_label.text() == "分析中"


def test_saved_placeholder_does_not_overwrite_active_recording(host):
    host.count_board.mode = "mark"
    host.on_clicked_player_btn()
    host._set_active_product_condition_stage("采集中")
    host.player_status_flag = True
    host._manual_product_group_raw_results = Mock(return_value={"q6000": "not_labeled"})
    host._refresh_manual_product_condition_results_from_group(host._manual_product_condition_group_id)
    assert row_text(host) == "采集中"


@pytest.mark.parametrize("mode", ["mark", "test"])
def test_analysis_phase_survives_session_refresh_then_returns_to_result(host, mode):
    host.count_board.mode = mode
    host.on_clicked_player_btn()
    host._mark_manual_product_condition_recording_completed()
    host._manual_product_group_raw_results = Mock(return_value={"q6000": "not_labeled"})

    def analyze(**_kwargs):
        assert host.left_panel.stage_label.text() == "分析中"
        assert row_text(host) == "分析中"
        host._refresh_manual_product_condition_results_from_group(host._manual_product_condition_group_id)
        assert row_text(host) == "分析中"
        return "analysis-return-value"

    host._run_analysis_impl = analyze
    assert host.run(show_windows=False) == "analysis-return-value"
    assert row_text(host) == "未判定"
    assert host.left_panel.stage_label.text() == "等待下一档位"
    assert not host._manual_product_analysis_key


def test_analysis_failure_is_not_an_ng_judgement_and_can_be_retried(host):
    host.on_clicked_player_btn()
    host._mark_manual_product_condition_recording_completed()
    host._run_analysis_impl = Mock(side_effect=ValueError("test failure"))
    with pytest.raises(ValueError, match="test failure"):
        host.run(show_windows=False)
    assert row_text(host) == "分析失败"
    assert host.left_panel.stage_label.text() == "测试异常"
    assert host.left_panel.port_result_value.text() == "待判定"
    assert host.left_panel.progress_label.text() == "档位进度：0/2"
    assert not host._manual_product_analysis_key
    assert host._active_product_condition_key == "q6000"
    host._run_analysis_impl = Mock(return_value=None)
    host.run(show_windows=False)
    assert row_text(host) == "未判定"


def test_history_analysis_does_not_change_live_task_status(host):
    host.on_clicked_player_btn()
    before = (host.left_panel.stage_label.text(), row_text(host))
    host.count_board.mode = "view"
    host._run_analysis_impl = Mock()
    host.run(show_windows=True, capture_product_report=False)
    assert (host.left_panel.stage_label.text(), row_text(host)) == before


@pytest.mark.parametrize("result", ["OK", "NG", "not_labeled"])
def test_round_completion_keeps_flow_and_verdict_separate(host, result):
    expected = result if result in ("OK", "NG") else "待判定"
    for condition in host.product_test_condition_configs:
        host.on_clicked_player_btn()
        host._mark_manual_product_condition_recording_completed()
        if result in ("OK", "NG"):
            host._update_manual_product_condition_result_after_analysis(result)
        host._advance_manual_product_condition_cycle_after_recording()
    assert host.left_panel.round_result_value.text() == expected
    assert host.left_panel.stage_label.text() == "本轮完成"
    assert host._manual_product_condition_group_id == ""


def test_import_preparation_does_not_claim_recording(host):
    host.queue_modes["queue_6000"] = "IMPORT_AUDIO"
    host.on_clicked_player_btn()
    assert host.left_panel.stage_label.text() == "等待开始"
    assert row_text(host) == "等待导入"


@pytest.mark.parametrize("text", ["检测中", "未标记", "完成"])
def test_summary_only_displays_judgement_for_nonfinal_states(host, text):
    host.left_panel.set_final_result(text)
    assert host.left_panel.round_result_value.text() == "待判定"


def test_channel_progress_counts_only_real_judgements(host):
    panel = host.left_panel
    panel.set_condition_channel_results("q6000", [
        {"result": "OK"}, {"result": "待检测"}, {"result": "NG"},
        {"SPL": "OK"}, {"result": "未标记"}, {"result": "OK"},
    ])
    assert panel.rows["q6000"]["completed_channels"] == 2
    assert panel.rows["q6000"]["labels"]["progress"].text() == "通道判定：2/5"
    panel.set_condition_result("q6000", "OK")
    assert panel.rows["q6000"]["completed_channels"] == 2
    panel.set_condition_result("q6000", "采集中", tone="running")
    assert panel.rows["q6000"]["completed_channels"] == 0
    assert panel.rows["q6000"]["channel_results"] == []


def test_port_summary_uses_verdict_not_status_color(host):
    panel = host.left_panel
    panel.set_condition_result("q6000", "完成", tone="ok")
    panel.set_condition_result("q7000", "OK", tone="ok")
    assert panel.progress_label.text() == "档位进度：1/2"
    panel.set_condition_result("q7000", "分析失败", tone="ng")
    assert panel.port_result_value.text() == "待判定"
    assert panel.progress_label.text() == "档位进度：0/2"
    panel.set_condition_result("q6000", "NG", tone="ng")
    assert panel.port_result_value.text() == "NG"
    panel.current_port_combo.setCurrentIndex(1)
    assert panel.port_result_value.text() == "待判定"


def test_channel_results_keep_physical_identity_and_spl_details(host):
    host.on_clicked_player_btn()
    host.left_panel.set_channels([0, 7])
    host.analysis_config = {"pressure": {"type": "SPL", "limit_checked": True}}
    host.analysis_window = [
        SimpleNamespace(
            _sequence_analysis_key="pressure",
            _sequence_runtime_key=f"pressure--通道{channel + 1}",
            _analysis_raw_channel=channel,
            title_name=f"pressure--通道{channel + 1}",
            result={"overall_spl": pressure},
        )
        for channel, pressure in [(7, 82.0), (0, 42.0)]
    ]
    host.data_struct = SimpleNamespace(analysis_result_dict={
        "pressure--通道1": (True, 0), "pressure--通道8": (False, 2),
    })

    host._sync_left_panel_analysis_details()

    row = host.left_panel.rows["q6000"]
    results = {item["raw_channel"]: item for item in row["channel_results"]}
    assert results[0]["SPL"] == results[0]["result"] == "OK"
    assert results[7]["SPL"] == results[7]["result"] == "NG"
    assert "42.00" in results[0]["details"]["SPL"]
    assert "82.00" in results[7]["details"]["SPL"]
    assert row["labels"]["progress"].text() == "通道判定：2/2"
@pytest.mark.parametrize("second_result, expected", [(None, "待判定"), (True, "OK"), (False, "NG")])
def test_repeated_analysis_items_do_not_hide_pending_or_failed_judgements(host, second_result, expected):
    host.analysis_config = {
        key: {"type": "FFT", "limit_checked": True} for key in ("first", "second")
    }
    host.analysis_window = [
        SimpleNamespace(_sequence_analysis_key=key, _analysis_raw_channel=0, title_name=key)
        for key in ("first", "second")
    ]
    host.data_struct = SimpleNamespace(analysis_result_dict={
        "first": (True, 0), "second": (second_result, 0),
    })
    result, = host._build_left_panel_channel_results()
    assert result["FFT"] == result["result"] == expected


def test_channel_progress_ignores_unselected_and_duplicate_channels(host):
    panel = host.left_panel
    panel.set_channels([2, 7])
    panel.set_condition_channel_results("q6000", [
        {"raw_channel": 7, "result": "OK"},
        {"raw_channel": 7, "result": "NG"},
        {"raw_channel": 0, "result": "OK"},
        {"raw_channel": 2, "result": "待判定"},
    ])
    assert panel.rows["q6000"]["labels"]["progress"].text() == "通道判定：1/2"
    assert panel.channel_labels == ["CH3", "CH8"]
    panel.set_channels([2])
    assert panel.rows["q6000"]["labels"]["progress"].text() == "通道判定：0/1"
