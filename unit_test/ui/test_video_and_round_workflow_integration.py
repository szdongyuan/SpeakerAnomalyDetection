"""Integration guards for video, recording admission, and round reset."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QComboBox, QWidget

import main_window as main_window_module
from main_window import MainWindow

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
    sequence.left_panel = SimpleNamespace(
        video_monitor_panel=panel, station_panel=object(), set_daily_records=Mock()
    )
    sequence.plot_grid = object()
    sequence.show_workbench_round = Mock()
    sequence.set_workbench_active_product = Mock()
    sequence.show_workbench_products = Mock()
    sequence.set_workbench_product_selection_enabled = Mock()
    sequence.using_file_combobox = QComboBox(sequence)
    sequence.update_v2pa_factor = Mock()
    factory = Mock(return_value=sequence)
    monkeypatch.setattr(main_window_module, "SequenceWindow", factory)
    workbench = Mock()
    monkeypatch.setattr(main_window_module, "MotorWorkbenchController", Mock(return_value=workbench))
    card = SimpleNamespace(settings_requested=SimpleNamespace(connect=Mock()))
    host = SimpleNamespace(
        _workbench_scanner_admission_available=Mock(),
        _workbench_configuration_busy=Mock(),
        recording_bridge=object(),
        raw_audio_csv_bridge=object(),
        ve_prewarm_lifetime=object(),
        ve_profile_store=object(),
        ve_calibration_store=object(),
        video_controller=SimpleNamespace(attach_panel=Mock(return_value=card), show_settings=Mock()),
        init_menu=Mock(return_value=QWidget()),
        set_title=Mock(return_value=QWidget()),
        _create_menu_row=Mock(return_value=QWidget()),
        setCentralWidget=Mock(),
        mic=None, speaker=None, mic_channels=[], speaker_channels=[],
    )
    try:
        MainWindow.init_sequence_widget(host)
        factory.assert_called_once_with(can_commit=host._workbench_scanner_admission_available)
        host.video_controller.attach_panel.assert_called_once_with(panel)
        card.settings_requested.connect.assert_called_once_with(workbench.open_video_settings)
        workbench.refresh_configuration.assert_called_once_with()
    finally:
        if host.setCentralWidget.called:
            host.setCentralWidget.call_args.args[0].close()
        sequence.close()
