"""Render caller-supplied daily records, or host native modal-input verification.

Run from the repository root with --output-dir PATH; --interactive uses Windows
and leaves a live host for real mouse/keyboard input. Only synthetic data is saved.
"""

import argparse
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-dpi", type=int, default=96)
    parser.add_argument("--font", type=Path, action="append", default=[])
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.font_dpi <= 0:
        parser.error("--font-dpi must be positive")
    os.environ["QT_QPA_PLATFORM"] = "windows" if args.interactive else "offscreen"
    os.environ["QT_FONT_DPI"] = str(args.font_dpi)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from PyQt5.QtCore import QEvent, QObject, QPoint, Qt, QTimer
    from PyQt5.QtGui import QFont, QFontDatabase, QFontInfo, QImage, QKeySequence, QPainter
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import (
        QApplication, QLabel, QLineEdit, QPushButton, QShortcut, QVBoxLayout, QWidget,
    )
    from ui.sequence.motor_dashboard_panel import MotorDashboardPanel

    app = QApplication.instance() or QApplication([])
    families = []
    font_paths = args.font or [Path("C:/Windows/Fonts") / name for name in
                               ("msyh.ttc", "msyhbd.ttc", "seguisym.ttf")]
    for path in font_paths:
        if not path.is_file():
            raise RuntimeError(f"Verification font unavailable: {path}")
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            raise RuntimeError(f"Cannot register verification font: {path}")
        families.extend(QFontDatabase.applicationFontFamilies(font_id))
    QFont.insertSubstitution("Microsoft YaHei UI", "Segoe UI Symbol")
    font = QFont("Microsoft YaHei UI")
    font.setPointSizeF(9.75)
    app.setFont(font)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(name, value):
        target = args.output_dir / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def settle():
        app.processEvents()
        QTest.qWait(30)
        app.processEvents()

    def wav_records(count, long_name=False):
        day = date(2026, 11, 4)
        return [dict(completed_at=datetime.combine(day, datetime.min.time()) + timedelta(seconds=i),
                     station_id=i % 3 + 1,
                     file_name=("20261104_工位1_长文件名称_" * 8 if long_name else "20261104_CASE-")
                     + f"{i + 1:03}.wav",
                     condition=("正转", "反转", "低速保持", "负载测试")[i % 4],
                     result=("OK", "未标记", "NG")[i % 3]) for i in range(count)]

    def round_records(count):
        """Independent completed rounds, never inferred from the WAV fixture."""
        day = date(2026, 11, 4)
        return [dict(completed_at=datetime.combine(day, datetime.min.time()) + timedelta(seconds=i),
                     station_id=i % 3 + 1,
                     product_code=f"ROUND-{i + 1:03}", forward="OK", reverse="NG", result="NG")
                for i in range(count)]

    def rect(value):
        return [value.x(), value.y(), value.width(), value.height()]

    def labels(panel):
        return {name: getattr(panel, name + "_label").text()
                for name in ("date", "total", "ok", "ng", "unmarked", "rate")}

    def dialog_evidence(summary):
        dialog = summary._details_dialog
        if dialog is None:
            return None
        headers = [dialog.table.horizontalHeaderItem(c).text()
                   for c in range(dialog.table.columnCount())]
        assert headers == ["完成时间", "工位", "文件名称", "工况", "结果", "操作"], headers
        file_tooltips = [dialog.table.item(r, 2).toolTip()
                         for r in range(dialog.table.rowCount())]
        assert all(tooltip == dialog.table.item(r, 2).text()
                   for r, tooltip in enumerate(file_tooltips)), file_tooltips
        assert all(tooltip.endswith(".wav") for tooltip in file_tooltips), file_tooltips
        condition_tooltips = [dialog.table.item(r, 3).toolTip()
                              for r in range(dialog.table.rowCount())]
        assert all(tooltip == dialog.table.item(r, 3).text()
                   for r, tooltip in enumerate(condition_tooltips)), condition_tooltips
        assert dialog.table.columnWidth(4) >= dialog.fontMetrics().horizontalAdvance("未标记") + 24
        critical = {}
        for name in ("station_label", "station_combo", "result_label", "result_combo", "search_edit",
                     "record_count_label", "page_label", "prev_button", "next_button"):
            control = getattr(dialog, name)
            critical[name] = dict(geometry=rect(control.geometry()),
                                  visible_region=rect(control.visibleRegion().boundingRect()),
                                  enabled=control.isEnabled(),
                                  global_origin=[control.mapToGlobal(QPoint()).x(),
                                                 control.mapToGlobal(QPoint()).y()])
        return dict(summary=labels(dialog), row_count=dialog.table.rowCount(),
                    headers=headers, file_tooltips=file_tooltips, condition_tooltips=condition_tooltips,
                    column_widths=[dialog.table.columnWidth(c)
                                   for c in range(dialog.table.columnCount())],
                    horizontal_scroll_maximum=dialog.table.horizontalScrollBar().maximum(),
                    matched_count=dialog.record_count_label.text(), page=dialog.page_label.text(),
                    visible=dialog.isVisible(), modal=app.activeModalWidget() is dialog,
                    content_geometry=rect(dialog.geometry()), frame_geometry=rect(dialog.frameGeometry()),
                    screen_geometry=rect(dialog.screen().availableGeometry()), controls=critical,
                    font=dict(family=QFontInfo(dialog.font()).family(),
                              point_size=dialog.font().pointSizeF(),
                              pixel_height=dialog.fontMetrics().height()),
                    filters=dict(station=dialog.station_combo.currentData(),
                                 result=dialog.result_combo.currentData(), query=dialog.search_edit.text()),
                    rows=[[dialog.table.item(r, c).text() for c in range(dialog.table.columnCount())]
                          for r in range(dialog.table.rowCount())])

    def render(widget, name):
        image = QImage(widget.width() * 2, widget.height() * 2, QImage.Format_ARGB32)
        image.setDevicePixelRatio(2)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        widget.render(painter)
        painter.end()
        target = args.output_dir / name
        if not image.save(str(target)):
            raise RuntimeError(f"Cannot save verification image: {target}")

    if args.interactive:
        host = QWidget()
        host.setAttribute(Qt.WA_DeleteOnClose)
        host.setWindowTitle("Motor Daily Details Native Verification")
        layout = QVBoxLayout(host)
        main_button = QPushButton("Main action (Alt+M)", host)
        main_button.setObjectName("main_button")
        main_button.setShortcut("Alt+M")
        main_edit = QLineEdit(host)
        main_edit.setObjectName("main_edit")
        main_edit.setPlaceholderText("Main input (Alt+I): verify modal blocking here")
        input_shortcut = QShortcut(QKeySequence("Alt+I"), host)
        input_shortcut.activated.connect(main_edit.setFocus)
        counter = QLabel("click_count=0 heartbeat=0", host)
        for control in (main_button, main_edit, counter):
            layout.addWidget(control)
        dashboard = MotorDashboardPanel(host)
        layout.addWidget(dashboard, 1)
        dashboard.summary_panel.findChild(QPushButton, "summaryDetails").setShortcut("Alt+D")
        dashboard.summary_panel.set_daily_records(round_records(7), date(2026, 11, 4))
        dashboard.summary_panel.set_daily_wav_records(wav_records(41), date(2026, 11, 4))
        state = dict(click_count=0, heartbeat=0, main_text="", events=[])

        def event(name, **values):
            state["events"].append(dict(sequence=len(state["events"]) + 1,
                                        heartbeat=state["heartbeat"], event=name, **values))

        def clicked():
            state["click_count"] += 1
            event("main_button_clicked", count=state["click_count"])

        def edited(text):
            state["main_text"] = text
            event("main_edit_changed", text=text)

        main_button.clicked.connect(clicked)
        main_edit.textChanged.connect(edited)

        class Observer(QObject):
            def eventFilter(self, watched, ev):
                if ev.type() in (QEvent.Show, QEvent.Hide, QEvent.Close) and watched.objectName() == "motorDailyDetailsDialog":
                    event("dialog_lifecycle", type=int(ev.type()))
                return False

        observer = Observer(host)
        app.installEventFilter(observer)

        def tick():
            state["heartbeat"] += 1
            counter.setText(f"click_count={state['click_count']} heartbeat={state['heartbeat']}")
            details = dialog_evidence(dashboard.summary_panel)
            snapshot = dict(state, summary=labels(dashboard.summary_panel), dialog=details,
                            platform=app.platformName(), host_geometry=rect(host.frameGeometry()),
                            main_button_global=rect(main_button.rect().translated(main_button.mapToGlobal(QPoint()))),
                            main_edit_global=rect(main_edit.rect().translated(main_edit.mapToGlobal(QPoint()))))
            write_json("native-state.json", snapshot)

        timer = QTimer(host)
        timer.timeout.connect(tick)
        timer.start(250)
        host.destroyed.connect(lambda: write_json("native-final.json", dict(state, host_destroyed=True)))
        host.setGeometry(30, 40, 450, 800)
        host.show()
        event("host_started")
        tick()
        print(f"Native host ready: {args.output_dir.resolve()}", flush=True)
        return app.exec_()

    evidence = dict(font_dpi=args.font_dpi, registered_fonts=families,
                    font_files=[str(p) for p in font_paths], platform=app.platformName(), renders=[])
    for name, count in (("empty", 0), ("full-20", 20), ("last-21", 21),
                        ("last-39", 39), ("combined", 41), ("no-match", 41),
                        ("narrow", 41), ("long-file-name", 20), ("dark-host", 20),
                        ("separate-counts", 2)):
        host = MotorDashboardPanel()
        try:
            host.resize(450, 760)
            if name == "dark-host":
                host.setStyleSheet("QWidget { background: #17212e; color: #ffffff; }")
            summary = host.summary_panel
            rounds = round_records(0 if not count else 1 if name == "separate-counts" else 7)
            wavs = wav_records(count, name == "long-file-name")
            if name == "separate-counts":
                wavs[0].update(station_id=1, condition="正转", result="OK")
                wavs[1].update(station_id=1, condition="反转", result="NG")
            summary.set_daily_records(rounds, date(2026, 11, 4))
            summary.set_daily_wav_records(wavs, date(2026, 11, 4))
            host.show()
            settle()
            summary.findChild(QPushButton, "summaryDetails").click()
            settle()
            dialog = summary._details_dialog
            if name.startswith("last-"):
                dialog.next_button.click()
            elif name == "combined":
                dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(1))
                dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("OK"))
                dialog.search_edit.setText(" case-0 ")
            elif name == "no-match":
                dialog.search_edit.setText("not-a-file.wav")
            elif name == "narrow":
                dialog.resize(480, min(570, dialog.height()))
            settle()
            entry = dialog_evidence(summary)
            entry.update(name=name, main_summary=labels(summary))
            expected_rows = {"empty": 0, "last-21": 1, "last-39": 19,
                             "combined": 14, "no-match": 0, "separate-counts": 2}.get(name, 20)
            assert entry["row_count"] == expected_rows, entry
            for source, data in ((entry["summary"], wavs), (entry["main_summary"], rounds)):
                assert source["date"] == "2026-11-04", source
                assert source["total"] == str(len(data)), source
                for field, result in (("ok", "OK"), ("ng", "NG"), ("unmarked", "未标记")):
                    assert source[field] == str(sum(row["result"] == result for row in data)), source
                expected_rate = (f"{100 * sum(row['result'] == 'OK' for row in data) / len(data):.1f}%"
                                 if data else "--")
                assert source["rate"] == expected_rate, source
            if name == "separate-counts":
                assert entry["main_summary"]["ng"] == "1" and entry["main_summary"]["rate"] == "0.0%", entry
                assert entry["summary"]["ok"] == "1" and entry["summary"]["rate"] == "50.0%", entry
            assert dialog.screen().availableGeometry().contains(dialog.frameGeometry()), entry
            for control_name, control in entry["controls"].items():
                assert control["visible_region"][2] > 0 and control["visible_region"][3] > 0, control_name
            render(dialog, name + ".png")
            render(summary, name + "-summary.png")
            if name == "narrow":
                scrollbar = dialog.table.horizontalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                settle()
                for column in (3, 4, 5):
                    assert dialog.table.viewport().rect().intersects(
                        dialog.table.visualItemRect(dialog.table.item(0, column))), column
                render(dialog, name + "-right.png")
            evidence["renders"].append(entry)
            print(f"{name}: {entry['row_count']} rows, page {entry['page']}")
        finally:
            if host.summary_panel._details_dialog is not None:
                host.summary_panel._details_dialog.close()
            host.close()
            host.deleteLater()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            settle()
    write_json("render-evidence.json", evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
