"""Native, modeless-to-the-event-loop child view for one saved record."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


class RecordAnalysisDialog(QDialog):
    item_selected = pyqtSignal(str)
    channel_selected = pyqtSignal(int)
    view_closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('recordAnalysisDialog')
        self.setWindowTitle('记录分析')
        self.setWindowModality(Qt.WindowModal)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._closed = False
        self._full_file_name = ''
        self.content = None
        # QObject.destroyed runs before Qt deletes its children. Keep this
        # callback independent of the dying dialog's C++ wrapper.
        self._content_owner = {}
        owner = self._content_owner

        def release_content(*_):
            content = owner.pop('content', None)
            if content is not None and hasattr(content, 'dispose'):
                content.dispose()

        self.destroyed.connect(release_content)
        self.setStyleSheet('''
            QDialog#recordAnalysisDialog {background:#eef3f8; color:#293b4d;}
            QDialog#recordAnalysisDialog QLabel {color:#293b4d; background:transparent;}
            QDialog#recordAnalysisDialog QListWidget {background:#e6eef8; border:0; border-radius:6px;}
            QDialog#recordAnalysisDialog QListWidget::item {padding:12px 8px; color:#46617d;}
            QDialog#recordAnalysisDialog QListWidget::item:selected {background:#3478c6; color:white;}
            QDialog#recordAnalysisDialog QScrollArea {background:white; border:1px solid #dce5ee;}
            QDialog#recordAnalysisDialog QComboBox {background:white; padding:4px; color:#293b4d;}
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 12)
        heading = QHBoxLayout()
        title = QLabel('记录分析')
        title.setStyleSheet('font-size:18px; font-weight:600; color:#2869af;')
        heading.addWidget(title)
        heading.addStretch()
        layout.addLayout(heading)
        self.file_name = QLabel()
        self.file_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_name.setMinimumWidth(0)
        layout.addWidget(self.file_name)
        record_line = QHBoxLayout()
        self.record_info = QLabel()
        self.record_info.setWordWrap(True)
        self.original_result = QLabel('原记录结果：--')
        record_line.addWidget(self.record_info, 1)
        record_line.addWidget(self.original_result)
        layout.addLayout(record_line)
        body = QHBoxLayout()
        self.navigation = QListWidget()
        self.navigation.setMinimumWidth(120)
        self.navigation.setMaximumWidth(200)
        self.navigation.currentItemChanged.connect(self._item_changed)
        body.addWidget(self.navigation, 1)
        right = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.item_title = QLabel()
        self.item_title.setWordWrap(True)
        self.item_title.setStyleSheet('font-size:15px; font-weight:600;')
        toolbar.addWidget(self.item_title, 1)
        self.channel_selector = QComboBox()
        self.channel_selector.setMinimumContentsLength(8)
        self.channel_selector.currentIndexChanged.connect(self._channel_changed)
        self.channel_selector.hide()
        toolbar.addWidget(self.channel_selector)
        right.addLayout(toolbar)
        self.status = QLabel('待分析')
        self.status_detail = QLabel()
        self.status_detail.setWordWrap(True)
        self.status_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.status)
        right.addWidget(self.status_detail)
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setMinimumWidth(0)
        right.addWidget(self.content_scroll, 1)
        body.addLayout(right, 4)
        layout.addLayout(body, 1)
        self.read_only_note = QLabel('按打开时当前配置分析，仅用于查看')
        self.read_only_note.setStyleSheet('color:#617183;')
        layout.addWidget(self.read_only_note)
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        available = screen.availableGeometry()
        self.resize(min(960, available.width()), min(680, available.height()))

    def show_record(self, record):
        self._full_file_name = record.file_name
        self.file_name.setToolTip(record.file_name)
        self._elide_filename()
        self.record_info.setText(
            f'工位 {record.station_id}  ·  {record.condition}  ·  '
            f'{record.completed_at:%Y-%m-%d %H:%M:%S}')
        self.original_result.setText(f'原记录结果：{record.result}')

    def _elide_filename(self):
        self.file_name.setText(self.file_name.fontMetrics().elidedText(
            self._full_file_name, Qt.ElideMiddle, max(0, self.file_name.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_filename()

    def set_analysis_items(self, items):
        self.navigation.clear()
        for item in items:
            row = QListWidgetItem(f'{item.config_key}  ·  {item.analysis_type}')
            row.setData(Qt.UserRole, item.config_key)
            row.setToolTip(row.text())
            self.navigation.addItem(row)
        if self.navigation.count():
            self.navigation.setCurrentRow(0)

    def _item_changed(self, current, previous):
        if current is not None:
            self.item_title.setText(current.text())
            self.item_selected.emit(current.data(Qt.UserRole))

    def set_channels(self, channels, selected_index=None):
        """Populate configured heatmap channels without generating selection intents."""
        self.channel_selector.blockSignals(True)
        self.channel_selector.clear()
        for channel in channels:
            self.channel_selector.addItem(
                f'CH{channel.wav_channel} · {channel.name}', channel.channel_index)
        index = self.channel_selector.findData(selected_index)
        self.channel_selector.setCurrentIndex(max(0, index) if channels else -1)
        self.channel_selector.blockSignals(False)
        self.channel_selector.setVisible(bool(channels))

    def _channel_changed(self, index):
        if index >= 0:
            self.channel_selected.emit(self.channel_selector.itemData(index))

    def show_status(self, text, detail=''):
        self.set_content(None)
        self.set_channels(())
        self.status.setText(text)
        self.status_detail.setText(detail)
        self.status_detail.setVisible(bool(detail))

    def set_content(self, widget):
        old = self.content_scroll.takeWidget()
        self.content = widget
        self._content_owner['content'] = widget
        if old is not None:
            if hasattr(old, 'dispose'):
                old.dispose()
            old.hide()
            old.setParent(None)
            old.deleteLater()
        if widget is not None:
            self.content_scroll.setWidget(widget)

    def _notify_closed(self):
        if not self._closed:
            self._closed = True
            self.set_content(None)
            self.view_closed.emit()

    def done(self, result):
        self._notify_closed()
        super().done(result)

    def closeEvent(self, event):
        self._notify_closed()
        super().closeEvent(event)
