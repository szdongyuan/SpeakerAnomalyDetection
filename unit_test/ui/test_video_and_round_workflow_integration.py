"""Integration guards for video, recording admission, and round reset."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget

import main_window as main_window_module
from main_window import MainWindow
from ui.sequence import sequence_widget_test_metadata_ops as metadata
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from unit_test.test_test_round_metadata import _MetadataHost


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("delete_failed", [False, True])
def test_manual_start_and_button_require_both_guards(
    ui_qapp, monkeypatch, admitted, delete_failed
):
    monkeypatch.setattr(metadata.LoadUiConfig, "load_last_recorded_info", lambda logger: None)
    monkeypatch.setattr(metadata, "save_recorded_data_to_json", Mock())
    monkeypatch.setattr(metadata.QMessageBox, "warning", Mock())
    host = _MetadataHost()
    host.player_btn = host.toolsbar.player_btn
    host._next_manual_product_condition_display_name = Mock(return_value="condition")
    host._can_prepare_recording_workflow = Mock(return_value=admitted)
    host._round_reset_delete_failed = delete_failed
    host.toolsbar.sample_number_lineedit.setText("sample-1")
    host.toolsbar.current_round_spinbox.setValue(1)
    try:
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
        expected = admitted and not delete_failed
        assert host.player_btn.isEnabled() is expected
        result = host._prepare_next_manual_product_condition_recording()
        assert result is (True if expected else None)
        if not expected:
            host._load_sequence_config_for_product_condition.assert_not_called()
            assert not host._manual_product_condition_group_id
    finally:
        host.toolsbar.close()
        host.close()


@pytest.mark.parametrize("video_finished", [False, True])
def test_close_preserves_discovery_cleanup_and_waits_for_video(ui_qapp, video_finished):
    calls = []
    video = SimpleNamespace(
        is_shutdown_complete=video_finished,
        closed=SimpleNamespace(connect=Mock()),
        shutdown=Mock(side_effect=lambda: calls.append("video")),
    )
    bridge = SimpleNamespace(
        service=SimpleNamespace(closed=SimpleNamespace(is_set=lambda: False)),
        shutdown=Mock(side_effect=lambda callback: calls.append("audio")),
    )
    host = SimpleNamespace(
        sequence_window=None,
        ve_discovery=object(),
        _close_ve_discovery=Mock(side_effect=lambda: calls.append("discovery")),
        video_controller=video,
        recording_bridge=bridge,
        _finish_recording_shutdown=Mock(),
        setEnabled=Mock(),
        close=Mock(),
    )
    event = QCloseEvent()
    MainWindow.closeEvent(host, event)
    assert not event.isAccepted()
    assert calls == ["discovery", "audio" if video_finished else "video"]
    if video_finished:
        video.shutdown.assert_not_called()
        bridge.shutdown.assert_called_once_with(host._finish_recording_shutdown)
    else:
        bridge.shutdown.assert_not_called()
        video.closed.connect.assert_called_once_with(host.close)
        MainWindow.closeEvent(host, QCloseEvent())
        video.shutdown.assert_called_once()
        video.closed.connect.assert_called_once()


def test_sequence_initialization_keeps_prewarm_stores_and_video_panel(ui_qapp, monkeypatch):
    sequence = QWidget()
    panel = object()
    sequence.left_panel = SimpleNamespace(video_monitor_panel=panel)
    sequence.update_v2pa_factor = Mock()
    factory = Mock(return_value=sequence)
    monkeypatch.setattr(main_window_module, "SequenceWindow", factory)
    host = SimpleNamespace(
        recording_bridge=object(),
        ve_prewarm_lifetime=object(),
        ve_profile_store=object(),
        ve_calibration_store=object(),
        video_controller=SimpleNamespace(attach_panel=Mock()),
        init_menu=Mock(return_value=QWidget()),
        set_title=Mock(return_value=QWidget()),
        _create_menu_row=Mock(return_value=QWidget()),
        setCentralWidget=Mock(),
        mic=None, speaker=None, mic_channels=[], speaker_channels=[],
    )
    try:
        MainWindow.init_sequence_widget(host)
        factory.assert_called_once_with(
            recording_bridge=host.recording_bridge,
            ve_prewarm_lifetime=host.ve_prewarm_lifetime,
        )
        assert sequence.ve_profile_store is host.ve_profile_store
        assert sequence.ve_calibration_store is host.ve_calibration_store
        host.video_controller.attach_panel.assert_called_once_with(panel)
        sequence.update_v2pa_factor.assert_called_once()
    finally:
        if host.setCentralWidget.called:
            host.setCentralWidget.call_args.args[0].close()
        sequence.close()
