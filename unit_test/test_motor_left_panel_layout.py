from PyQt5.QtWidgets import QLabel, QWidget

from ui.sequence.motor_left_panel import MotorDetectionLeftPanel

# Current station, summary and video integration.
def test_dashboard_integration_zero_stations_and_detached_video(qt_app):
    panel = MotorDetectionLeftPanel()
    panel.show()
    qt_app.processEvents()
    assert panel.get_station_count() == 0
    assert panel.dashboard_panel.summary_panel.isVisible()
    dashboard, video = panel.take_split_sections()
    assert dashboard is panel.dashboard_panel
    assert video is panel.video_monitor_panel
    assert not hasattr(panel, "result_panel")
    dashboard.close()
    video.close()
    dashboard.deleteLater()
    video.deleteLater()
    panel.close()
    panel.deleteLater()

def test_left_panel_preserves_video_and_hides_business_count_board(qt_app):
    board = QWidget()
    panel = MotorDetectionLeftPanel()
    panel.resize(340, 800)
    panel.show()
    qt_app.processEvents()
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert "视频监控" in labels
    assert "信息汇总" in labels
    assert panel.content_layout.itemAt(0).widget() is panel.dashboard_panel
    assert panel.content_layout.itemAt(1).widget() is panel.video_monitor_panel
    assert panel.get_station_count() == 0
    panel.close()
    panel.deleteLater()


def test_daily_records_forwarding_preserves_stations_conditions_and_video(qt_app):
    from datetime import date
    from PyQt5.QtCore import QCoreApplication, QEvent
    from unit_test.daily_record_cases import make_daily_records

    board = QWidget()
    panel = MotorDetectionLeftPanel()
    try:
        panel.set_station_count(1)
        panel.set_station_row_count(1, 1)
        panel.set_station_row(1, 0, ["existing", 12, 34])
        panel.set_station_results(1, "OK", "NG")
        station = panel.get_station_data(1)
        video = panel.video_monitor_panel
        day = date(2026, 11, 4)
        records = make_daily_records(3, day)
        records[0]["station_id"] = 99
        assert panel.set_daily_records(records, day) is None
        assert panel.dashboard_panel.summary_panel.findChild(QLabel, "summaryTotal").text() == "3"
        assert panel.get_station_count() == 1
        assert panel.get_station_data(1) == station
        assert panel.video_monitor_panel is video
        assert panel.dashboard_panel.summary_panel.findChild(QLabel, "summaryTotal").text() == "3"
    finally:
        panel.close()
        panel.deleteLater()
        board.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_daily_details_uses_detached_dashboard_host_and_is_released_with_host(qt_app):
    from PyQt5 import sip
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QPushButton, QVBoxLayout

    panel = MotorDetectionLeftPanel()
    host = QWidget()
    dashboard, video = panel.take_split_sections()
    QVBoxLayout(host).addWidget(dashboard)
    host.show()
    qt_app.processEvents()
    try:
        summary = dashboard.summary_panel
        summary.findChild(QPushButton, "summaryDetails").click()
        qt_app.processEvents()
        dialog = summary._details_dialog
        assert dialog.parentWidget() is host
        assert qt_app.activeModalWidget() is dialog
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()
        assert sip.isdeleted(dialog)
        assert sip.isdeleted(summary)
        assert qt_app.activeModalWidget() is None
    finally:
        if not sip.isdeleted(host):
            host.close()
            host.deleteLater()
        video.deleteLater()
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
