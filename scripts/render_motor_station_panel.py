"""Render the real station dashboard; example data exists only in this CLI.

Run from the repository root, for example::

    python scripts/render_motor_station_panel.py --output-dir output/motor-station-panel
    python scripts/render_motor_station_panel.py --output-dir output/motor-station-panel/scaled --font-dpi 120
"""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-dpi", type=int, default=96,
                        help="Qt font DPI for a separate scaled-font verification run")
    parser.add_argument("--font", type=Path, action="append", default=[],
                        help="Optional real font file to register (repeatable)")
    args = parser.parse_args()
    if args.font_dpi <= 0:
        parser.error("--font-dpi must be positive")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_FONT_DPI"] = str(args.font_dpi)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPainter
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QTableWidget, QWidget

    from ui.sequence.motor_dashboard_panel import MotorDashboardPanel

    app = QApplication.instance() or QApplication([])
    font_paths = args.font or [Path("C:/Windows/Fonts/simsun.ttc"),
                               Path("C:/Windows/Fonts/seguisym.ttf")]
    registered = []
    for path in font_paths:
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id < 0:
                raise RuntimeError(f"Cannot register verification font: {path}")
            registered.extend(QFontDatabase.applicationFontFamilies(font_id))
    if not QFontDatabase().families():
        raise RuntimeError("Qt has no real fonts; supply --font with an installed font file")
    if "Segoe UI Symbol" in registered:
        # Offscreen's empty system font DB needs an explicit real-glyph fallback.
        QFont.insertSubstitution("SimSun", "Segoe UI Symbol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {"font_dpi": args.font_dpi, "registered_fonts": registered,
                "image_scale": 2, "renders": []}

    def settle():
        app.processEvents()
        QTest.qWait(20)
        app.processEvents()

    def expand(panel, station_id):
        button = panel.findChild(QPushButton, f"stationHeader{station_id}")
        panel.findChild(QScrollArea).ensureWidgetVisible(button)
        settle()
        QTest.mouseClick(button, Qt.LeftButton, pos=button.rect().center())
        settle()
        assert button.isChecked()

    def populate(panel, long_text=False):
        panel.set_station_count(3)
        for station_id, count in ((1, 2), (2, 5), (3, 0)):
            panel.set_station_row_count(station_id, count)
        panel.set_station_row(1, 0, ["SPL", "68.50 dB(A)", "63.60 dB(A)"])
        panel.set_station_row(1, 1, ["FFT", "分析已完成", "分析已完成"])
        panel.set_station_column(2, 0, ["SPL", "FFT", "FBA", "SPL 复测", "时频图"])
        panel.set_station_column(2, 1, ["61.20 dB(A)", "分析已完成", "0.00 dB(Z)", "62.00 dB(A)", "分析已完成"])
        panel.set_station_column(2, 2, ["62.40 dB(A)", "分析已完成", "64.10 dB(Z)", "--", "--"])
        if long_text:
            panel.set_station_cell(2, 4, 0, "很长的指标名称 / Long metric name")
        panel.set_station_results(1, " ok ", "NG", result="NG")
        panel.set_station_results(2, "--", "ok", result="OK")
        panel.set_station_results(3, "--", "--", result="--")

    def save(widget, name, requested_width):
        settle()
        image = QImage(widget.width() * 2, widget.height() * 2, QImage.Format_ARGB32)
        image.setDevicePixelRatio(2)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        widget.render(painter)
        painter.end()
        target = args.output_dir / f"{name}-{requested_width}.png"
        if not image.save(str(target)):
            raise RuntimeError(f"Cannot save {target}")
        panel = widget.station_panel
        scroll = panel.findChild(QScrollArea)
        general = panel.findChild(QWidget, "stationGeneralHeader")
        entry = {"file": target.name, "requested_width": requested_width,
                 "actual_size": [widget.width(), widget.height()],
                 "header_width": general.width(),
                 "label_contents_width": general.contentsRect().width(),
                 "viewport_width": scroll.viewport().width(),
                 "scroll_value": scroll.verticalScrollBar().value(),
                 "scroll_maximum": scroll.verticalScrollBar().maximum(),
                 "summary_visible": widget.summary_panel.visibleRegion().isEmpty() is False,
                 "stations": [panel.get_station_data(i) for i in range(1, panel.get_station_count() + 1)],
                 "expanded": [i for i in range(1, panel.get_station_count() + 1)
                              if panel.findChild(QPushButton, f"stationHeader{i}").isChecked()]}
        entry["critical_labels"] = {
            label.objectName(): {"text": label.text(), "size": [label.width(), label.height()],
                                 "text_height": label.fontMetrics().height()}
            for label in widget.findChildren(QLabel)
            if label.objectName().startswith(("summary", "stationGeneral"))
        }
        entry["table_headers"] = {
            table.objectName(): [table.horizontalHeaderItem(i).text() for i in range(3)]
            for table in panel.findChildren(QTableWidget)
        }
        evidence["renders"].append(entry)
        assert entry["summary_visible"], entry
        assert entry["label_contents_width"] == entry["viewport_width"], entry
        assert entry["header_width"] == panel.width() - 2, entry
        print(f"{target}: {widget.width()}x{widget.height()}, expanded={entry['expanded']}, scroll={entry['scroll_value']}/{entry['scroll_maximum']}")

    for width in (340, 600):
        for scenario in ("zero", "populated", "zero-row", "long-text", "scroll"):
            widget = MotorDashboardPanel()
            widget.resize(width, 760)
            widget.show()
            settle()
            panel = widget.station_panel
            if scenario != "zero":
                populate(panel, long_text=scenario == "long-text")
                settle()
                expand(panel, 1)
                expand(panel, 2)
                if scenario == "zero-row":
                    expand(panel, 3)
                    panel.findChild(QScrollArea).ensureWidgetVisible(
                        panel.findChild(QTableWidget, "stationTable3"))
                if scenario == "scroll":
                    panel.set_station_count(18)
                    panel.set_station_results(16, "OK", "--", result="OK")
                    panel.set_station_results(17, "OK", "NG", result="NG")
                    panel.set_station_results(18, "--", "--", result="--")
                    settle()
                    bar = panel.findChild(QScrollArea).verticalScrollBar()
                    bar.setValue(bar.maximum())
                    settle()
                    assert bar.maximum() > 0
            save(widget, scenario, width)
            widget.close()
            widget.deleteLater()
            settle()
    (args.output_dir / "render-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
