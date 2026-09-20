"""Read-only view of the images and scalar results saved beside one recording."""

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QStyle, QVBoxLayout,
    QStyledItemDelegate, QTableWidget, QTableWidgetItem,
)

from base.audio_analysis_result_source import item_channels
from ui.audio_analysis_result_loader import AudioAnalysisResultLoader
from ui.config_dialog_base import ConfigDialogBase
from ui.dialog_enter_policy import install_dialog_enter_policy


class AnalysisImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setDragMode(self.ScrollHandDrag)
        self.setBackgroundBrush(Qt.white)
        self.fitting = True

    def set_image(self, image):
        self.scene().clear()
        item = self.scene().addPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(item.boundingRect())
        self.fit_image()

    def clear_image(self):
        self.scene().clear()

    def fit_image(self):
        self.fitting = True
        if self.scene().items():
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def original_size(self):
        self.fitting = False
        self.resetTransform()

    def wheelEvent(self, event):
        if not self.scene().items():
            return
        self.fitting = False
        current = self.transform().m11()
        target = min(8.0, max(0.1, current * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2)))
        self.scale(target / current, target / current)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitting:
            self.fit_image()


class SegmentSummaryDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Draw this read-only summary explicitly so the shared table theme cannot
        # override the saved judgement colours or the subtle heading background.
        painter.save()
        background = index.data(Qt.BackgroundRole)
        if background is not None:
            painter.fillRect(option.rect, background)
        foreground = index.data(Qt.ForegroundRole)
        painter.setPen(foreground.color() if foreground is not None else QColor("#26364a"))
        painter.setFont(option.font)
        rect = option.rect.adjusted(10, 0, -10, 0)
        text = option.fontMetrics.elidedText(index.data(), Qt.ElideRight, rect.width())
        painter.drawText(rect, index.data(Qt.TextAlignmentRole), text)
        painter.setPen(QColor("#d5dfe9"))
        if index.row() < index.model().rowCount() - 1:
            painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        if index.column() == 0:
            painter.drawLine(option.rect.topRight(), option.rect.bottomRight())
        painter.restore()


class SegmentSummaryTable(QTableWidget):
    """Compact scalar or segment summary; extra segments scroll horizontally."""

    def __init__(self, parent=None):
        super().__init__(3, 0, parent)
        self.setObjectName("segmentSummaryTable")
        self.setFrameShape(QFrame.NoFrame)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().setMinimumSectionSize(0)
        self.verticalHeader().setMinimumSectionSize(0)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setItemDelegate(SegmentSummaryDelegate(self))
        self.setStyleSheet("""
            QTableWidget#segmentSummaryTable {
                background: transparent; border: none; color: #26364a;
                font-family: "Microsoft YaHei"; font-size: 14px;
            }
        """)
        self._label_width = 0
        self._value_width = 0

    def set_summary(self, rows):
        self.clearContents()
        self.setRowCount(len(rows))
        self.setColumnCount(len(rows[0]))
        metrics = self.fontMetrics()
        self._label_width = max(metrics.horizontalAdvance(row[0]) for row in rows) + 24
        self._value_width = max(92, max(metrics.horizontalAdvance(text)
                                      for row in rows for text in row[1:]) + 24)
        for row, texts in enumerate(rows):
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                item.setTextAlignment((Qt.AlignLeft if column == 0 else Qt.AlignHCenter) | Qt.AlignVCenter)
                if row == 0:
                    item.setBackground(QColor("#e8eff8"))
                if row == len(rows) - 1 and column > 0 and text in ("OK", "NG"):
                    item.setForeground(QColor("#27804b" if text == "OK" else "#bc3535"))
                self.setItem(row, column, item)
        self.horizontalScrollBar().setValue(0)
        self._fit_columns()

    def _fit_columns(self):
        count = self.columnCount() - 1
        if count < 1:
            return
        self.setColumnWidth(0, self._label_width)
        available = max(0, self.viewport().width() - self._label_width)
        width, remainder = divmod(max(available, count * self._value_width), count)
        for column in range(1, count + 1):
            self.setColumnWidth(column, width + (column <= remainder))
        overflow = available < count * self._value_width
        height = sum(self.rowHeight(row) for row in range(self.rowCount()))
        if overflow:
            height += self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.setFixedHeight(min(100, height))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_columns()


class ArchiveAudioAnalysisDialog(ConfigDialogBase):
    _MAX_SCALAR_HEIGHT = 100

    def __init__(self, wav_path, parent=None):
        super().__init__(parent)
        self.wav_path = str(wav_path)
        self.results = None
        self.scalars = ()
        self.scalar_issues = ()
        self.channel_issues = ()
        self.images = ()
        self.image_index = 0
        self._request_id = 0
        self._closing = False
        self._preferred_channel = None
        self.loader = AudioAnalysisResultLoader()
        self.loader.completed.connect(self._loaded)
        self.destroyed.connect(self.loader.cancel)
        self.setWindowTitle("分析结果")
        self.setWindowModality(Qt.WindowModal)
        self.resize(960, 680)
        screen = self.screen().availableGeometry()
        self.resize(min(self.width(), screen.width() - 40), min(self.height(), screen.height() - 60))
        self._build_ui()
        install_dialog_enter_policy(self, None)
        QTimer.singleShot(0, self._discover)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        self.file_label = QLabel()
        self.file_label.setTextFormat(Qt.PlainText)
        self.file_label.setToolTip(self.wav_path)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.file_label)
        layout.addSpacing(18)
        selectors = QHBoxLayout()
        self.item_combo = QComboBox()
        self.channel_combo = QComboBox()
        for title, combo in (("分析项", self.item_combo), ("通道", self.channel_combo)):
            selectors.addWidget(QLabel(title))
            selectors.addWidget(combo, 1)
            combo.setEnabled(False)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addLayout(selectors)
        self.stack = QStackedWidget()
        self.status_label = QLabel("正在加载…")
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.image_view = AnalysisImageView()
        self.stack.addWidget(self.status_label)
        self.stack.addWidget(self.image_view)
        layout.addWidget(self.stack, 1)
        self.scalar_label = QLabel()
        self.scalar_label.setContentsMargins(6, 4, 6, 4)
        self.scalar_label.setTextFormat(Qt.PlainText)
        self.scalar_label.setWordWrap(True)
        self.scalar_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.scalar_area = QScrollArea()
        self.scalar_area.setWidgetResizable(True)
        self.scalar_area.setWidget(self.scalar_label)
        self.scalar_area.setFrameShape(QFrame.NoFrame)
        self.scalar_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.scalar_area.viewport().setAutoFillBackground(False)
        self.scalar_label.setAutoFillBackground(False)
        self.scalar_area.setMaximumHeight(self._MAX_SCALAR_HEIGHT)
        self.scalar_area.hide()
        layout.addWidget(self.scalar_area)
        self.scalar_table = SegmentSummaryTable()
        self.scalar_table.hide()
        layout.addWidget(self.scalar_table)
        self.issue_label = QLabel()
        self.issue_label.setTextFormat(Qt.PlainText)
        self.issue_label.setWordWrap(True)
        self.issue_label.hide()
        layout.addWidget(self.issue_label)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        # Add to the outer layout's 12 px gap, separating results from actions.
        buttons.setContentsMargins(0, 8, 0, 0)
        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.image_counter = QLabel()
        self.folder_button = QPushButton("打开结果文件夹")
        self.fit_button = QPushButton("适应窗口")
        self.original_button = QPushButton("原图")
        for widget in (self.previous_button, self.image_counter, self.next_button):
            buttons.addWidget(widget)
            widget.hide()
        buttons.addStretch()
        for button in (self.folder_button, self.fit_button, self.original_button):
            button.setAutoDefault(False)
            button.setEnabled(False)
            buttons.addWidget(button)
            if button is self.folder_button:
                buttons.addSpacing(16)
        self.previous_button.setAutoDefault(False)
        self.next_button.setAutoDefault(False)
        layout.addLayout(buttons)
        self.item_combo.currentIndexChanged.connect(self._item_changed)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.fit_button.clicked.connect(self.image_view.fit_image)
        self.original_button.clicked.connect(self.image_view.original_size)
        self.folder_button.clicked.connect(self._open_folder)
        self.previous_button.clicked.connect(lambda: self._change_image(-1))
        self.next_button.clicked.connect(lambda: self._change_image(1))
        self._update_file_label()

    def _update_file_label(self):
        name = Path(self.wav_path).name
        self.file_label.setText(self.file_label.fontMetrics().elidedText(
            name, Qt.ElideMiddle, max(1, self.file_label.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "file_label"):
            self._update_file_label()

    def _discover(self):
        if not self._closing:
            self._request("discover", self.wav_path)

    def _request(self, kind, payload):
        self._request_id = self.loader.submit(kind, payload)

    def _show_status(self, text):
        self.image_view.clear_image()
        self.status_label.setText(text)
        self.stack.setCurrentWidget(self.status_label)
        self.fit_button.setEnabled(False)
        self.original_button.setEnabled(False)

    def _show_issues(self, issues):
        issues = tuple(dict.fromkeys(issues))
        self.issue_label.setText(issues[0] + (f"（共{len(issues)}项，悬停查看）" if len(issues) > 1 else "") if issues else "")
        self.issue_label.setToolTip("\n".join(issues))
        self.issue_label.setVisible(bool(issues))

    def _loaded(self, request_id, kind, result, error):
        if self._closing or request_id != self._request_id:
            return
        if kind == "discover":
            if error:
                self._show_status(f"结果读取失败：{error}")
                return
            self.results = result
            self.folder_button.setEnabled(bool(result.directories))
            if not result.items:
                self._show_issues(())
                self._show_status(result.issues[0] if result.issues else "暂无已保存的分析结果")
                self.status_label.setToolTip("\n".join(result.issues))
                return
            self._show_issues(result.issues)
            self.item_combo.blockSignals(True)
            for item in result.items:
                self.item_combo.addItem(item.name, item)
            self.item_combo.blockSignals(False)
            self.item_combo.setEnabled(len(result.items) > 1)
            self._item_changed()
        elif kind == "scalars":
            self.scalars = result.values if result else ()
            self.scalar_issues = (error,) if error else result.issues
            item = self.item_combo.currentData()
            channels, self.channel_issues = item_channels(item, self.scalars)
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            for channel, label in channels:
                self.channel_combo.addItem(label, channel)
            preferred = self.channel_combo.findData(self._preferred_channel)
            if preferred >= 0:
                self.channel_combo.setCurrentIndex(preferred)
            self.channel_combo.blockSignals(False)
            self.channel_combo.setEnabled(len(channels) > 1)
            self._channel_changed()
        elif kind == "image":
            if error:
                self._show_status(error)
            else:
                self.stack.setCurrentWidget(self.image_view)
                self.image_view.set_image(result)
                self.fit_button.setEnabled(True)
                self.original_button.setEnabled(True)

    def _item_changed(self, *_):
        item = self.item_combo.currentData()
        if item is None:
            return
        self._preferred_channel = self.channel_combo.currentData()
        self.channel_combo.setEnabled(False)
        self.scalars = ()
        self.scalar_area.hide()
        self.scalar_table.hide()
        self._show_issues(())
        self.images = ()
        self._update_navigation()
        self._show_status("正在加载…")
        self._request("scalars", item)

    def _channel_changed(self, *_):
        item = self.item_combo.currentData()
        channel = self.channel_combo.currentData()
        self.images = tuple(image for image in item.images if image.channel_id == channel)
        self.image_index = 0
        values = tuple(value for value in self.scalars if value.channel_id == channel)
        self._show_scalars(values)
        self._show_issues(self.results.issues + self.scalar_issues + self.channel_issues)
        self._load_image()

    @staticmethod
    def _display_scalar_value(value):
        if value.metric == "总体声压级" and value.value != "数值不可用":
            return f"{float(value.value):.2f}"
        return value.value

    def _show_scalars(self, values):
        segmented = (
            len(values) > 1 and any(value.segment_label for value in values)
            and len({(value.metric, value.unit) for value in values}) == 1
        )
        single = len(values) == 1 and not values[0].segment_label
        self.scalar_area.setVisible(bool(values) and not (segmented or single))
        self.scalar_table.setVisible(segmented or single)
        if segmented or single:
            self.scalar_table.setMinimumWidth(360 if single else 0)
            self.scalar_table.setMaximumWidth(360 if single else 16777215)
            self.layout().setAlignment(self.scalar_table, Qt.AlignLeft if single else Qt.Alignment())
            rows = self._segment_summary_rows(values)
            self.scalar_table.set_summary(rows[1:] if single else rows)
        else:
            self.scalar_label.setText(self._scalar_lines(values))
            height = len(values) * self.scalar_label.fontMetrics().lineSpacing() + 8
            self.scalar_area.setFixedHeight(min(self._MAX_SCALAR_HEIGHT, max(32, height)))

    def _scalar_lines(self, values):
        lines = []
        for value in values:
            title = f"{value.segment_label} · {value.metric}" if value.segment_label else value.metric
            display_value = self._display_scalar_value(value)
            text = f"{title}：{display_value} {value.unit}".rstrip()
            text += f"  {value.judgement or '—'}"
            lines.append(text)
        return "\n".join(lines)

    def _segment_summary_rows(self, values):
        prefix = next((name for name in ("输出负载", "时间")
                       if all(value.segment_label.startswith(name) for value in values)), "")
        heading = "时间范围" if prefix == "时间" else prefix or "分段"
        first = values[0]
        title = first.metric + (f"（{first.unit}）" if first.unit else "")
        rows = [[heading], [title], ["判定"]]
        for value in values:
            segment = value.segment_label[len(prefix):] if prefix else value.segment_label
            if prefix == "输出负载" and segment.endswith("A"):
                segment = segment[:-1].strip() + " A"
            cells = (segment or "整条录音", self._display_scalar_value(value), value.judgement or "—")
            for row, text in zip(rows, cells):
                row.append(text)
        return rows

    def _update_navigation(self):
        for widget in (self.previous_button, self.next_button, self.image_counter):
            widget.setVisible(len(self.images) > 1)
        self.previous_button.setEnabled(self.image_index > 0)
        self.next_button.setEnabled(self.image_index + 1 < len(self.images))
        self.image_counter.setText(f"{self.image_index + 1}/{len(self.images)}" if self.images else "")

    def _change_image(self, offset):
        self.image_index += offset
        self._load_image()

    def _load_image(self):
        self._update_navigation()
        if not self.images:
            # Invalidate an in-flight image when switching to a channel without one.
            self._request_id = self.loader.invalidate()
            self._show_status("未保存分析图片")
            return
        self._show_status("正在加载…")
        self._request("image", self.images[self.image_index].path)

    def _open_folder(self):
        if self.images:
            self._open_directory(str(Path(self.images[self.image_index].path).parent))
            return
        item = self.item_combo.currentData()
        if item is not None and item.csv_files:
            self._open_directory(str(Path(item.csv_files[0][1]).parent))
            return
        # Empty or unsupported results still allow inspecting the recording's
        # known directory, without inventing another recording's location.
        directories = dict(self.results.directories)
        self._open_directory(directories.get("分析数据") or next(iter(directories.values())))

    def _open_directory(self, path):
        if not Path(path).is_dir() or not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            QMessageBox.warning(self, "提示", "无法打开结果文件夹。")

    def done(self, result):
        if not self._closing:
            self._closing = True
            self.loader.cancel()
        super().done(result)

    def closeEvent(self, event):
        self.reject()
        event.accept()
