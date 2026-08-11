import os
import time
import sys
import wave
import copy
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QStandardItem, QBrush, QColor
from PyQt5.QtWidgets import QProgressDialog, QFileDialog, QApplication, QWidget, QHBoxLayout
from scipy.io import wavfile as scipy_wavfile

from base.audio_sample_rate import resolve_output_sample_rate
from base.file_ops import FileOps
from base.log_manager import LogManager
from base.playback_controller import PlaybackController
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog, FilterAudioDialog
from ui.custom_ui_widget.widgets import Label, PushButton, MessageBox, SpinBox


class ArchiveAudioDataDialog(AudioDataManageDialog):
    PAGE_SIZE = 100

    def __init__(self, logger: LogManager, speaker=None):
        self._current_page_index = 0
        self.speaker = speaker
        self._selected_audio_data_by_id = dict()
        self.prev_page_btn = None
        self.next_page_btn = None
        self.page_status_label = None
        self.page_jump_spinbox = None
        self.page_jump_btn = None
        self._pagination_row_widget = None
        self._play_btn_col = 6
        self._play_text = "\u64ad\u653e"
        self._stop_text = "\u505c\u6b62"
        self._is_desc_order = False
        self._play_col_debounce_ms = 250
        self._play_col_last_action_ts = 0.0
        self._current_playing_row: int = None
        self._current_playing_file = None
        self._audio_duration_cache = dict()
        self._playback_started_monotonic = None
        self._current_playing_total_sec = 0.0
        self._last_remaining_display_sec = None
        self._status_row_widget = None
        self._click_order = False
        # Base __init__ will trigger load_audio_data_to_view(), which may reset this label.
        # Create it early to avoid attribute access before init_ui() runs.
        self.remaining_time_label = Label()
        self._is_switching_playback = False
        self._playback_poll_timer = None
        self.playback_controller = PlaybackController()
        # Hover hand cursor only; no hover color logic.
        super(ArchiveAudioDataDialog, self).__init__(logger)
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        # removed blink/flash logic for play column
        self.order_btn = PushButton(" 倒  序 ")
        self.package_delete_btn = PushButton(" 打包并删除 ")
        self.package_btn = PushButton(" 打  包 ")
        self.delete_btn = PushButton(" 删  除 ")
        self.prev_page_btn = PushButton(" 上一页 ")
        self.next_page_btn = PushButton(" 下一页 ")
        self.page_status_label = Label()
        self.page_jump_spinbox = SpinBox()
        self.page_jump_btn = PushButton(" 跳  转 ")

        self.set_h_header(["", "文件名称", "产品型号", "音频标签", "采样率", "录音时间", "播放"])

        # 而是在模型里放“播放/停止”文本项，并用 view.clicked 响应点击。
        self.data_view.clicked.connect(self._on_data_view_clicked)
        self.data_view.setMouseTracking(True)
        self.data_view.viewport().installEventFilter(self)

        self._rebuild_play_buttons()
        self._update_order_button_text()
        self.init_ui()
        self._update_page_buttons()

    def _on_data_view_clicked(self, index):
        if not index.isValid():
            return
        if index.column() != self._play_btn_col:
            return
        self._on_play_button_clicked(index.row())

    def init_ui(self):
        self.setWindowTitle("音频数据管理")

        self.set_bottom_layout()
        self._setup_status_row_with_remaining_label()
        self._setup_pagination_row()
        self._set_remaining_label_idle()

        window_width = ui_style_const.scale_size_px(1060)
        window_height = ui_style_const.scale_size_px(700)
        self.resize(window_width, window_height)

    def load_audio_data_to_view(self):
        self._stop_playback_if_needed()
        self._clamp_current_page()
        if self.is_filter_flag:
            self.load_audio_data_to_model(self.filter_audio_data, self.stimulus_name)
        else:
            result = self.load_audio_data_to_model(self.all_audio_data, self.stimulus_name)
            self.product_model_set, self.record_date_set = result
        self._rebuild_select_wave_data()
        self._restore_current_page_checked_states()
        self.set_select_wave_num_text(len(self.select_wave_data))
        self._update_page_buttons()
        self._rebuild_play_buttons()

    def show_all_wave(self):
        should_clear_selection = self.is_filter_flag is True and len(self.all_audio_data) != len(self.filter_audio_data)
        self._current_page_index = 0
        self._stop_playback_if_needed()
        super().show_all_wave()
        if should_clear_selection:
            self._clear_selected_audio_data()
        self._rebuild_select_wave_data()
        self._restore_current_page_checked_states()
        self.set_select_wave_num_text(len(self.select_wave_data))
        self._update_page_buttons()
        self._rebuild_play_buttons()

    def filter_audio_data_at_filter_config(self, filter_config: dict):
        self._current_page_index = 0
        super().filter_audio_data_at_filter_config(filter_config)

    def on_click_filter_btn(self):
        filter_config = copy.deepcopy(self.filter_config)
        dlg = FilterAudioDialog(self.product_model_set, self.record_date_set, filter_config)
        dlg.hide_select_not_label_check_box(self.is_hide_select_not_label)
        flag, filter_config = dlg.exec()
        if flag == 1:
            self.filter_config = filter_config
            self.filter_audio_data_at_filter_config(self.filter_config)
            self.is_filter_flag = True
            self._clear_selected_audio_data()
            self.load_audio_data_to_view()
            self.filter_signal.emit()
        elif flag == 2:
            self.show_all_wave()
            self.filter_signal.emit()

    def load_audio_data_to_model(self, audio_data, stimulus_name):
        self.setRowCount(0)
        if not audio_data:
            return set(), set()

        product_model_set = set()
        record_date_set = set()
        for item in audio_data:
            product_model_set.add(item[2])
            record_date_set.add(item[4])

        page_start, page_end = self._get_page_bounds(audio_data)
        for item in audio_data[page_start:page_end]:
            file_name = item[1].split("/")[-1]
            row_data_list = [None, file_name, item[2], item[5], item[3], item[4], None]
            self.add_row_data(row_data_list)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        return product_model_set, record_date_set

    def set_bottom_layout(self):
        all_show_btn = PushButton("全部显示")

        all_show_btn.clicked.connect(self.show_all_wave)
        self.order_btn.clicked.connect(self.on_clicked_order_btn)
        self.package_delete_btn.clicked.connect(self.on_clicked_package_delete_btn)
        self.package_btn.clicked.connect(self.on_clicked_package_btn)
        self.delete_btn.clicked.connect(self.on_clicked_delete_btn)

        self.bottom_layout.addWidget(all_show_btn)
        self.bottom_layout.addWidget(self.order_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(self.package_delete_btn)
        self.bottom_layout.addWidget(self.package_btn)
        self.bottom_layout.addWidget(self.delete_btn)

    def _setup_pagination_row(self):
        root_layout = self.layout()
        if root_layout is None:
            return
        if self._pagination_row_widget is not None:
            return
        if (
            self.page_status_label is None
            or self.page_jump_spinbox is None
            or self.page_jump_btn is None
            or self.prev_page_btn is None
            or self.next_page_btn is None
        ):
            return

        self.prev_page_btn.clicked.connect(self.on_clicked_prev_page_btn)
        self.next_page_btn.clicked.connect(self.on_clicked_next_page_btn)
        self.page_jump_btn.clicked.connect(self.on_clicked_jump_page_btn)

        self.page_jump_spinbox.setRange(1, 1)
        self.page_jump_spinbox.setValue(1)

        self._pagination_row_widget = QWidget(self)
        pagination_layout = QHBoxLayout(self._pagination_row_widget)
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.addWidget(self.page_status_label)
        pagination_layout.addStretch()
        pagination_layout.addWidget(Label("跳转到"))
        pagination_layout.addWidget(self.page_jump_spinbox)
        pagination_layout.addWidget(Label("页"))
        pagination_layout.addWidget(self.page_jump_btn)
        pagination_layout.addSpacing(12)
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addWidget(self.next_page_btn)

        bottom_layout_index = -1
        for idx in range(root_layout.count()):
            item = root_layout.itemAt(idx)
            if item is not None and item.layout() is self.bottom_layout:
                bottom_layout_index = idx
                break

        if bottom_layout_index >= 0:
            root_layout.insertWidget(bottom_layout_index, self._pagination_row_widget)
        else:
            root_layout.addWidget(self._pagination_row_widget)

    def _setup_status_row_with_remaining_label(self):
        root_layout = self.layout()
        if root_layout is None:
            return
        if self._status_row_widget is not None:
            return

        select_label_index = -1
        for idx in range(root_layout.count()):
            item = root_layout.itemAt(idx)
            if item is not None and item.widget() is self.select_data_num:
                select_label_index = idx
                root_layout.takeAt(idx)
                break

        self._status_row_widget = QWidget(self)
        status_row_layout = QHBoxLayout(self._status_row_widget)
        status_row_layout.setContentsMargins(0, 0, 0, 0)
        status_row_layout.addWidget(self.select_data_num, alignment=Qt.AlignLeft)
        status_row_layout.addStretch()
        status_row_layout.addWidget(self.remaining_time_label, alignment=Qt.AlignRight)

        if select_label_index >= 0:
            root_layout.insertWidget(select_label_index, self._status_row_widget)
        else:
            root_layout.addWidget(self._status_row_widget)

    def _set_remaining_label_idle(self):
        self.remaining_time_label.clear()
        self.remaining_time_label.hide()

    def _set_remaining_label_seconds(self, seconds: int):
        self.remaining_time_label.setText(f"正在播放-剩余时长：{self._format_hhmmss(seconds)}")
        self.remaining_time_label.show()

    def _update_order_button_text(self):
        self.order_btn.setText(" 正  序 " if self._is_desc_order else " 倒  序 ")

    def _get_active_audio_data(self):
        return self.filter_audio_data if self.is_filter_flag else self.all_audio_data

    def _get_page_count(self, audio_data=None):
        active_audio_data = self._get_active_audio_data() if audio_data is None else audio_data
        if not active_audio_data:
            return 1
        return ((len(active_audio_data) - 1) // self.PAGE_SIZE) + 1

    def _clamp_current_page(self):
        page_count = self._get_page_count()
        self._current_page_index = max(0, min(self._current_page_index, page_count - 1))

    def _get_page_bounds(self, audio_data=None):
        active_audio_data = self._get_active_audio_data() if audio_data is None else audio_data
        page_count = self._get_page_count(active_audio_data)
        current_page = max(0, min(self._current_page_index, page_count - 1))
        start = current_page * self.PAGE_SIZE
        end = min(start + self.PAGE_SIZE, len(active_audio_data))
        return start, end

    def _get_current_page_audio_data(self):
        audio_data = self._get_active_audio_data()
        start, end = self._get_page_bounds(audio_data)
        return audio_data[start:end]

    def _get_source_index_for_visible_row(self, row):
        if row is None:
            return None
        audio_data = self._get_active_audio_data()
        start, end = self._get_page_bounds(audio_data)
        source_index = start + row
        if row < 0 or source_index < start or source_index >= end:
            return None
        return source_index

    def _get_audio_data_by_visible_row(self, row):
        source_index = self._get_source_index_for_visible_row(row)
        if source_index is None:
            return None
        audio_data = self._get_active_audio_data()
        if source_index < 0 or source_index >= len(audio_data):
            return None
        return audio_data[source_index]

    def _get_audio_data_id(self, audio_data):
        if not audio_data:
            return None
        return str(audio_data[0])

    def _rebuild_select_wave_data(self):
        audio_data_by_id = {self._get_audio_data_id(item): item for item in self.all_audio_data}
        self._selected_audio_data_by_id = {
            audio_data_id: audio_data_by_id[audio_data_id]
            for audio_data_id in self._selected_audio_data_by_id
            if audio_data_id in audio_data_by_id
        }
        self.select_wave_data = dict(self._selected_audio_data_by_id)

    def _clear_selected_audio_data(self):
        self._selected_audio_data_by_id.clear()
        self.select_wave_data.clear()

    def _restore_current_page_checked_states(self):
        model = self.model()
        if model is None:
            return
        old_is_send_signal = self.data_view.is_send_signal
        self.data_view.is_send_signal = False
        try:
            page_audio_data = self._get_current_page_audio_data()
            for row, item_data in enumerate(page_audio_data):
                item = model.item(row, 0)
                if item is None:
                    continue
                item.setCheckState(
                    Qt.Checked
                    if self._get_audio_data_id(item_data) in self._selected_audio_data_by_id
                    else Qt.Unchecked
                )
        finally:
            self.data_view.is_send_signal = old_is_send_signal
        self._update_all_selected_checkbox_state()

    def _update_all_selected_checkbox_state(self):
        active_audio_data = self._get_active_audio_data()
        is_all_selected = bool(active_audio_data) and all(
            self._get_audio_data_id(item) in self._selected_audio_data_by_id for item in active_audio_data
        )
        self.all_select_flag = is_all_selected
        self.all_selected_checkbox.setChecked(is_all_selected)

    def _update_page_buttons(self):
        if self.prev_page_btn is None or self.next_page_btn is None:
            return
        self._clamp_current_page()
        page_count = self._get_page_count()
        has_multiple_pages = len(self._get_active_audio_data()) > self.PAGE_SIZE
        if self.page_status_label is not None:
            self.page_status_label.setText(f"第 {self._current_page_index + 1} / {page_count} 页")
        if self.page_jump_spinbox is not None:
            old_is_blocked = self.page_jump_spinbox.blockSignals(True)
            try:
                self.page_jump_spinbox.setRange(1, page_count)
                self.page_jump_spinbox.setValue(self._current_page_index + 1)
            finally:
                self.page_jump_spinbox.blockSignals(old_is_blocked)
        self.prev_page_btn.setEnabled(has_multiple_pages and self._current_page_index > 0)
        self.next_page_btn.setEnabled(has_multiple_pages and self._current_page_index < page_count - 1)

    def _reset_table_scroll_to_top_left(self):
        if self.data_view is None:
            return
        vertical_scroll_bar = self.data_view.verticalScrollBar()
        horizontal_scroll_bar = self.data_view.horizontalScrollBar()
        if vertical_scroll_bar is not None:
            vertical_scroll_bar.setValue(vertical_scroll_bar.minimum())
        if horizontal_scroll_bar is not None:
            horizontal_scroll_bar.setValue(horizontal_scroll_bar.minimum())

    def on_clicked_prev_page_btn(self):
        if self._current_page_index <= 0:
            return
        self._current_page_index -= 1
        self.load_audio_data_to_view()
        self._reset_table_scroll_to_top_left()

    def on_clicked_next_page_btn(self):
        if self._current_page_index >= self._get_page_count() - 1:
            return
        self._current_page_index += 1
        self.load_audio_data_to_view()
        self._reset_table_scroll_to_top_left()

    def on_clicked_jump_page_btn(self):
        page_number = self.page_jump_spinbox.value() if self.page_jump_spinbox is not None else 1
        page_count = self._get_page_count()
        if page_number < 1 or page_number > page_count:
            MessageBox.warning(self, "提示", "请输入有效页码")
            self._update_page_buttons()
            return
        self._current_page_index = page_number - 1
        self.load_audio_data_to_view()
        self._reset_table_scroll_to_top_left()

    def on_all_selected_changed(self):
        active_audio_data = self._get_active_audio_data()
        if self.all_select_flag:
            for item in active_audio_data:
                self._selected_audio_data_by_id.pop(self._get_audio_data_id(item), None)
        else:
            for item in active_audio_data:
                self._selected_audio_data_by_id[self._get_audio_data_id(item)] = item
        self._rebuild_select_wave_data()
        self._restore_current_page_checked_states()
        self.set_select_wave_num_text(len(self.select_wave_data))

    def set_select_wave_num_text(self, select_num):
        super().set_select_wave_num_text(select_num)

    def on_row_checkbox_toggled(self, item, is_checked):
        audio_data = self._get_audio_data_by_visible_row(item.row())
        audio_data_id = self._get_audio_data_id(audio_data)
        if audio_data_id is None:
            return

        if is_checked is True:
            self._selected_audio_data_by_id[audio_data_id] = audio_data
        elif is_checked is False:
            self._selected_audio_data_by_id.pop(audio_data_id, None)

        self._rebuild_select_wave_data()
        self._update_all_selected_checkbox_state()
        self.set_select_wave_num_text(len(self.select_wave_data))

    def _get_mirror_index(self, index, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if index < 0 or index >= capacity:
            raise IndexError("index out of range")
        return capacity - 1 - index

    def on_clicked_order_btn(self):
        self._click_order = True
        playing_audio_data = self._get_audio_data_by_visible_row(self._current_playing_row)
        playing_audio_data_id = self._get_audio_data_id(playing_audio_data)

        self.all_audio_data.reverse()
        self.filter_audio_data.reverse()
        if playing_audio_data_id is not None:
            for source_index, item in enumerate(self._get_active_audio_data()):
                if self._get_audio_data_id(item) == playing_audio_data_id:
                    self._current_page_index = source_index // self.PAGE_SIZE
                    self._current_playing_row = source_index % self.PAGE_SIZE
                    break
        else:
            self._current_page_index = 0
        self.load_audio_data_to_view()
        self._is_desc_order = not self._is_desc_order
        self._update_order_button_text()
        self._click_order = False

    def _apply_play_action_item_style(self, item: QStandardItem):
        font = item.font()
        font.setUnderline(True)
        font.setBold(False)
        item.setFont(font)
        item.setToolTip("")
        item.setData(QBrush(QColor(0, 123, 255)), Qt.ForegroundRole)
        item.setData(None, Qt.BackgroundRole)

    def _resolve_row_file_path(self, row):
        audio_data = self._get_audio_data_by_visible_row(row)
        if audio_data is None:
            return None
        raw_path = audio_data[1]
        if not raw_path:
            return None
        if os.path.isabs(raw_path):
            return os.path.abspath(raw_path)
        return os.path.abspath(os.path.join(DEFAULT_DIR, raw_path))

    @staticmethod
    def _format_hhmmss(seconds: int) -> str:
        total_seconds = max(0, int(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_audio_total_seconds(self, abs_path: str) -> float:
        if not abs_path:
            return 0.0
        normalized_path = os.path.abspath(abs_path)
        if normalized_path in self._audio_duration_cache:
            return float(self._audio_duration_cache.get(normalized_path, 0.0) or 0.0)
        duration = 0.0
        try:
            with wave.open(normalized_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate > 0:
                    duration = float(frames) / float(rate)
        except Exception:
            # Some WAV files are IEEE float (format=3), which wave.open cannot parse.
            # Fall back to scipy header/data parsing.
            try:
                sample_rate, audio_data = scipy_wavfile.read(normalized_path, mmap=True)
                if sample_rate > 0:
                    duration = float(audio_data.shape[0]) / float(sample_rate)
            except Exception:
                duration = 0.0
        self._audio_duration_cache[normalized_path] = duration
        return duration

    def _get_current_remaining_seconds(self) -> int:
        if self._current_playing_total_sec <= 0:
            return 0
        if self._playback_started_monotonic is None:
            return int(max(0.0, self._current_playing_total_sec))
        elapsed = max(0.0, time.monotonic() - float(self._playback_started_monotonic))
        remaining = max(0.0, float(self._current_playing_total_sec) - elapsed)
        return int(remaining)

    def _set_play_cell_text(self, row: int, is_current: bool) -> None:
        model = self.model()
        if model is None:
            return
        if row < 0 or row >= model.rowCount():
            return
        item = model.item(row, self._play_btn_col)
        if item is None:
            item = QStandardItem()
            model.setItem(row, self._play_btn_col, item)

        item.setText(self._stop_text if is_current else self._play_text)
        item.setTextAlignment(Qt.AlignCenter)
        self._apply_play_action_item_style(item)

    def _update_current_playing_remaining_text(self) -> None:
        if self._current_playing_row is None:
            return
        remaining_seconds = self._get_current_remaining_seconds()
        if self._last_remaining_display_sec == remaining_seconds:
            return
        self._last_remaining_display_sec = remaining_seconds
        self._set_remaining_label_seconds(remaining_seconds)

    def _rebuild_play_buttons(self):
        model = self.model()
        if model is None:
            return
        if model.columnCount() <= self._play_btn_col:
            return
        row_count = model.rowCount()

        for row in range(row_count):
            self._set_play_cell_text(row, False)
        self._refresh_play_button_states()

    def _refresh_play_button_states(self):
        model = self.model()
        if model is None:
            return
        if model.columnCount() <= self._play_btn_col:
            return
        row_count = model.rowCount()
        playing = self.playback_controller.is_audio_playing()
        for row in range(row_count):
            is_current = bool(playing and row == self._current_playing_row)
            self._set_play_cell_text(row, is_current)

    def eventFilter(self, watched, event):
        if watched is self.data_view.viewport():
            if event.type() == QEvent.MouseMove:
                index = self.data_view.indexAt(event.pos())
                if index.isValid() and index.column() == self._play_btn_col:
                    self.data_view.viewport().setCursor(Qt.PointingHandCursor)
                else:
                    self.data_view.viewport().unsetCursor()

            elif event.type() == QEvent.Leave:
                self.data_view.viewport().unsetCursor()

        return super().eventFilter(watched, event)

    def _show_switch_playback_blocked_popup(self):
        msg_box = MessageBox(self)
        msg_box.setWindowTitle("提示")
        msg_box.setText("当前有音频正在播放，请先停止后再播放其他音频。")
        msg_box.setIcon(MessageBox.NoIcon)
        msg_box.addButton("确定", MessageBox.AcceptRole)
        msg_box.exec_()

    def _clear_playing_state(self):
        self._current_playing_row = None
        self._current_playing_file = None
        self._playback_started_monotonic = None
        self._current_playing_total_sec = 0.0
        self._last_remaining_display_sec = None
        self._set_remaining_label_idle()
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        self._refresh_play_button_states()

    def _stop_playback_if_needed(self):
        is_stop = self.playback_controller.is_audio_playing() or self._current_playing_row is not None
        if is_stop and not self._click_order:
            self.playback_controller.stop_audio_playback()
            self._clear_playing_state()

    def _get_package_save_path(self):
        file_name = "audio_data_export_%s" % datetime.now().strftime("%Y%m%d")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存位置",
            os.path.join(model_consts.STORED_PACKAGE_PATH, file_name),
            "压缩文件 (*.zip)",
        )
        if not file_path:
            return None
        if not file_path.endswith(".zip"):
            file_path += ".zip"
        return os.path.normpath(file_path)

    def _resolve_audio_data_file_path(self, audio_data):
        raw_path = audio_data[1]
        if os.path.isabs(raw_path):
            return os.path.abspath(raw_path)
        return os.path.abspath(os.path.join(DEFAULT_DIR, raw_path))

    def _stop_playback_if_selected(self, selected_audio_data):
        current_file = self.playback_controller.get_current_playing_file()
        if not current_file:
            return
        current_file_abs = os.path.abspath(current_file)
        for item in selected_audio_data:
            if current_file_abs == self._resolve_audio_data_file_path(item):
                self._stop_playback_if_needed()
                return

    def _get_audio_data_file_name(self, audio_data):
        file_path = audio_data[1] if isinstance(audio_data, (list, tuple)) and len(audio_data) > 1 else audio_data
        return str(file_path).split("/")[-1].split("\\")[-1]

    def _build_package_delete_failure_report(
        self,
        output_zip_path,
        package_failed,
        source_delete_failed,
        archive_deleted_before_failure=None,
    ):
        if not package_failed and not source_delete_failed and not archive_deleted_before_failure:
            return None, None

        lines = [
            "失败清单",
            "生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
        ]
        if package_failed:
            lines.append("打包失败：%s 个" % len(package_failed))
            for item in package_failed:
                lines.append(self._get_audio_data_file_name(item))
            lines.append("")
        if source_delete_failed:
            lines.append("原文件删除失败：%s 个" % len(source_delete_failed))
            for item in source_delete_failed:
                lines.append(self._get_audio_data_file_name(item))
            lines.append("")
        if archive_deleted_before_failure:
            lines.append("压缩包最终写入失败前已删除源文件：%s 个" % len(archive_deleted_before_failure))
            for item in archive_deleted_before_failure:
                lines.append(self._get_audio_data_file_name(item))

        report_name = os.path.basename(os.path.splitext(output_zip_path)[0] + "_failures.txt")
        return report_name, "\n".join(lines)

    def _write_failure_report_to_file(self, output_zip_path, report_content):
        report_path = os.path.splitext(output_zip_path)[0] + "_failures.txt"
        try:
            FileOps.ensure_directory_exists(report_path)
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write(report_content)
            return report_path, None
        except Exception as e:
            self.logger.error(e)
            return None, e

    def _add_failure_report_to_zip(self, output_zip_path, report_name, report_content):
        if not report_name or not report_content:
            return None
        try:
            with ZipFile(output_zip_path, "a", compression=ZIP_DEFLATED) as zip_file:
                zip_file.writestr(report_name, report_content)
            return None
        except Exception as e:
            self.logger.error(e)
            return e

    def _show_package_delete_summary(
        self,
        packaged_deleted,
        package_failed,
        source_delete_failed,
        db_deleted_count,
        db_delete_error=None,
        database_package_error=None,
        archive_finalize_error=None,
        failure_report_path=None,
        failure_report_error=None,
        failure_report_zip_error=None,
        failure_report_zip_name=None,
    ):
        if database_package_error:
            title = "打包失败"
            lines = [
                "数据库快照写入失败，未删除任何音频或数据库记录。",
                "错误信息：%s" % str(database_package_error)[:120],
            ]
        elif archive_finalize_error:
            title = "打包并删除异常"
            if packaged_deleted:
                lines = [
                    "压缩包最终写入失败，压缩包可能不完整或不可用。",
                    "已删除源文件：%s 个" % len(packaged_deleted),
                    "数据库删除：0 条（数据库记录已保留）",
                ]
            else:
                lines = [
                    "压缩包创建或写入失败，未删除任何源音频。",
                    "数据库删除：0 条（数据库记录已保留）",
                ]
        else:
            title = "打包并删除结果"
            lines = [
                "打包并删除完成",
                "成功打包并删除原文件：%s 个" % len(packaged_deleted),
                "数据库删除：%s 条" % db_deleted_count,
            ]
        if package_failed:
            lines.append("打包失败：%s 个" % len(package_failed))
        if source_delete_failed:
            lines.append("原文件删除失败：%s 个" % len(source_delete_failed))
        if db_delete_error:
            lines.append("数据库删除失败：%s" % str(db_delete_error)[:120])
        if failure_report_zip_name:
            lines.append("失败清单已加入压缩包：%s" % failure_report_zip_name)
        if failure_report_path:
            lines.append("失败清单已保存：%s" % failure_report_path)
        if failure_report_error:
            lines.append("失败清单保存失败：%s" % str(failure_report_error)[:120])
        if failure_report_zip_error:
            lines.append("失败清单加入压缩包失败：%s" % str(failure_report_zip_error)[:120])

        msg_box = MessageBox(self)
        msg_box.setIcon(MessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText("\n".join(lines))
        msg_box.exec_()

    def _show_package_failure_summary(
        self,
        package_failed,
        failure_report_path=None,
        failure_report_error=None,
        failure_report_zip_error=None,
        failure_report_zip_name=None,
    ):
        if not package_failed and not failure_report_error and not failure_report_zip_error:
            return
        lines = ["打包完成", "打包失败：%s 个" % len(package_failed)]
        if failure_report_zip_name:
            lines.append("失败清单已加入压缩包：%s" % failure_report_zip_name)
        if failure_report_path:
            lines.append("失败清单已保存：%s" % failure_report_path)
        if failure_report_error:
            lines.append("失败清单保存失败：%s" % str(failure_report_error)[:120])
        if failure_report_zip_error:
            lines.append("失败清单加入压缩包失败：%s" % str(failure_report_zip_error)[:120])
        msg_box = MessageBox(self)
        msg_box.setIcon(MessageBox.Information)
        msg_box.setWindowTitle("打包结果")
        msg_box.setText("\n".join(lines))
        msg_box.exec_()

    def _refresh_after_audio_data_deleted(self, deleted_id_list):
        self._selected_audio_data_by_id.clear()
        self._rebuild_select_wave_data()
        self.delete_audio_data_with_id(deleted_id_list)
        self._clamp_current_page()
        self.update_filter_sets_after_deletion()
        self.load_audio_data_to_view()
        self.set_select_wave_num_text(len(self.select_wave_data))
        self._rebuild_play_buttons()

    def _on_play_button_clicked(self, row):
        now = time.monotonic()
        if now - float(self._play_col_last_action_ts or 0.0) < (float(self._play_col_debounce_ms) / 1000.0):
            return
        self._play_col_last_action_ts = now

        if self._is_switching_playback:
            return

        self._is_switching_playback = True
        try:
            row_path = self._resolve_row_file_path(row)
            if not row_path:
                MessageBox.warning(self, "提示", "未找到该行对应音频文件路径")
                self._clear_playing_state()
                return

            if self.playback_controller.is_audio_playing() and row == self._current_playing_row:
                self.playback_controller.stop_audio_playback()
                self._clear_playing_state()
                return

            if self.playback_controller.is_audio_playing() and row != self._current_playing_row:
                self._show_switch_playback_blocked_popup()
                return

            if not self.playback_controller.is_audio_playing() and self._current_playing_row is not None:
                self._clear_playing_state()

            code, msg = self._start_resolved_audio_playback(row_path)
            if code != error_code.OK:
                MessageBox.warning(self, "提示", msg)
                self._clear_playing_state()
                return

            self._current_playing_row = row
            self._current_playing_file = row_path
            self._playback_started_monotonic = time.monotonic()
            self._current_playing_total_sec = self._get_audio_total_seconds(row_path)
            self._last_remaining_display_sec = None
            self._set_remaining_label_seconds(self._get_current_remaining_seconds())
            if self._playback_poll_timer is not None and not self._playback_poll_timer.isActive():
                self._playback_poll_timer.start()
            self._refresh_play_button_states()
        except Exception as e:
            self.logger.error(e)
        finally:
            self._is_switching_playback = False

    def _start_resolved_audio_playback(self, path):
        if self.speaker is None:
            return (
                error_code.INVALID_PLAY,
                "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。",
            )
        raw_device = self.speaker.get("index") if isinstance(self.speaker, dict) else None
        if isinstance(raw_device, bool):
            device = None
        elif isinstance(raw_device, int):
            device = raw_device
        elif isinstance(raw_device, str):
            try:
                device = int(raw_device.strip(), 10)
            except ValueError:
                device = None
        else:
            device = None
        if device is None or device < 0:
            return (
                error_code.INVALID_PLAY,
                "输出设备信息无效，请在硬件管理中重新选择设备。",
            )
        result = resolve_output_sample_rate(self.speaker)
        if not result.ok:
            return error_code.INVALID_PLAY, result.message
        return self.playback_controller.start_audio_playback(
            path,
            device=device,
            output_sample_rate=result.sample_rate,
        )

    def _on_playback_poll_timeout(self):
        if not self.playback_controller.is_audio_playing():
            self._clear_playing_state()
            return

        playing_file = self.playback_controller.get_current_playing_file()
        if not playing_file:
            self._clear_playing_state()
            return

        if self._current_playing_file and os.path.abspath(playing_file) != os.path.abspath(self._current_playing_file):
            self._clear_playing_state()
            return

        self._update_current_playing_remaining_text()

    def _warn_cannot_close_while_playing(self):
        MessageBox.warning(self, "提示", "正在播放，请先停止播放后再退出")

    def on_clicked_package_delete_btn(self):
        selected_audio_data = list(self.select_wave_data.values())
        selected_count = len(selected_audio_data)
        if not selected_count:
            self.on_clicked_package_btn()
            return

        self._stop_playback_if_selected(selected_audio_data)

        file_path = self._get_package_save_path()
        if not file_path:
            return

        db_snapshot_path = os.path.join(DEFAULT_DIR, "database/audio_data.db")
        db_delete_step_count = max(1, (selected_count + 499) // 500)
        progress_total = 1 + selected_count + db_delete_step_count + 1
        package_delete_progress = QProgressDialog("正在打包并删除音频数据...", None, 0, progress_total, self)
        package_delete_progress.setWindowTitle("打包并删除进度")
        package_delete_progress.setWindowModality(Qt.WindowModal)
        package_delete_progress.setWindowFlags(package_delete_progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        package_delete_progress.show()
        QApplication.processEvents()

        packaged_deleted = list()
        package_failed = list()
        source_delete_failed = list()
        will_delete_in_db_list = list()
        db_deleted_count = 0
        db_delete_error = None
        database_package_error = None
        archive_finalize_error = None
        progress_value = 0
        FileOps.ensure_directory_exists(file_path)

        def update_package_delete_progress(value, text=None):
            nonlocal progress_value
            progress_value = min(value, progress_total)
            if text:
                package_delete_progress.setLabelText(text)
            package_delete_progress.setValue(progress_value)
            QApplication.processEvents()

        try:
            try:
                with ZipFile(file_path, "w", compression=ZIP_DEFLATED) as zip_file:
                    try:
                        zip_file.write(
                            db_snapshot_path,
                            FileOps.get_zip_arcname(db_snapshot_path, base_dir=DEFAULT_DIR),
                        )
                        update_package_delete_progress(1, "已写入数据库快照")
                    except Exception as e:
                        database_package_error = e
                        self.logger.error(e)

                    if not database_package_error:
                        for index, audio_data in enumerate(selected_audio_data, start=1):
                            result = FileOps.write_audio_to_zip_and_delete_source(
                                zip_file, audio_data, base_dir=DEFAULT_DIR
                            )
                            status = result.get("status")
                            if status == "packaged_deleted":
                                packaged_deleted.append(audio_data)
                                will_delete_in_db_list.append(result.get("db_delete_id") or audio_data[0])
                                self._audio_duration_cache.pop(self._resolve_audio_data_file_path(audio_data), None)
                            elif status == "source_delete_failed":
                                source_delete_failed.append(audio_data)
                                self.logger.error(result.get("error"))
                            else:
                                package_failed.append(audio_data)
                                self.logger.error(result.get("error"))

                            if index % 100 == 0 or index == selected_count:
                                update_package_delete_progress(
                                    1 + index,
                                    f"正在打包并删除源文件... {index}/{selected_count}",
                                )
            except Exception as e:
                if database_package_error is None:
                    archive_finalize_error = e
                    self.logger.error(e)
                else:
                    self.logger.error(e)

            if database_package_error is None and archive_finalize_error is None:
                db_progress_base = 1 + selected_count
                if will_delete_in_db_list:

                    def update_db_delete_progress(processed_count, total_count):
                        batch_index = min(db_delete_step_count, (processed_count + 499) // 500)
                        update_package_delete_progress(
                            db_progress_base + batch_index,
                            f"正在删除数据库记录... {processed_count}/{total_count}",
                        )

                    code, result = self.recording_manager.delete_audio_at_id_list(
                        will_delete_in_db_list, progress_callback=update_db_delete_progress
                    )
                    if code == error_code.OK:
                        db_deleted_count = len(will_delete_in_db_list)
                        self.logger.info("success delete audio with package delete workflow")
                        update_package_delete_progress(
                            progress_total - 1,
                            "正在刷新列表...",
                        )
                        self._refresh_after_audio_data_deleted(will_delete_in_db_list)
                    else:
                        db_delete_error = result
                        self.logger.error(result)
                        update_package_delete_progress(
                            progress_total - 1,
                            "数据库记录删除失败",
                        )
                else:
                    update_package_delete_progress(
                        progress_total - 1,
                        "无可删除的数据库记录",
                    )

                update_package_delete_progress(progress_total, "打包并删除完成")
        finally:
            package_delete_progress.close()

        failure_report_name, failure_report_content = self._build_package_delete_failure_report(
            file_path,
            package_failed,
            source_delete_failed,
            packaged_deleted if archive_finalize_error else None,
        )
        failure_report_path = None
        failure_report_error = None
        failure_report_zip_error = None
        failure_report_zip_name = None
        if failure_report_content:
            if archive_finalize_error:
                failure_report_path, failure_report_error = self._write_failure_report_to_file(
                    file_path, failure_report_content
                )
            else:
                failure_report_zip_error = self._add_failure_report_to_zip(
                    file_path, failure_report_name, failure_report_content
                )
                if failure_report_zip_error:
                    failure_report_path, failure_report_error = self._write_failure_report_to_file(
                        file_path, failure_report_content
                    )
                else:
                    failure_report_zip_name = failure_report_name

        self._show_package_delete_summary(
            packaged_deleted,
            package_failed,
            source_delete_failed,
            db_deleted_count,
            db_delete_error=db_delete_error,
            database_package_error=database_package_error,
            archive_finalize_error=archive_finalize_error,
            failure_report_path=failure_report_path,
            failure_report_error=failure_report_error,
            failure_report_zip_error=failure_report_zip_error,
            failure_report_zip_name=failure_report_zip_name,
        )

    def on_clicked_package_btn(self):
        self._stop_playback_if_needed()
        if not self.select_wave_data:
            msg_box = MessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("您未选择任何音频进行导出，程序将仅导出数据库，是否确定？")
            confirm_btn = msg_box.addButton(" 确  认 ", MessageBox.AcceptRole)
            cancel_btn = msg_box.addButton(" 取  消 ", MessageBox.RejectRole)
            msg_box.exec_()
            if msg_box.clickedButton() != confirm_btn:
                return

        file_path = self._get_package_save_path()
        if not file_path:
            return

        file_path_list = [i[1] for i in self.select_wave_data.values()]
        file_path_list.append("database/audio_data.db")
        self.packaging_progress = QProgressDialog("正在打包...", None, 0, len(file_path_list), self)
        self.packaging_progress.setWindowTitle("打包进度")
        self.packaging_progress.setWindowModality(Qt.WindowModal)
        self.packaging_progress.setWindowFlags(self.packaging_progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.packaging_progress.show()
        QApplication.processEvents()

        package_failed = list()

        def record_package_failure(path, error):
            package_failed.append(path)
            self.logger.error("Failed to package file [%s]. %s" % (path, error))

        FileOps.create_zip_with_files(
            file_path_list,
            file_path,
            progress_callback=self.update_packaging_progress,
            failure_callback=record_package_failure,
        )

        self._selected_audio_data_by_id.clear()
        self._rebuild_select_wave_data()
        self._restore_current_page_checked_states()
        self.set_select_wave_num_text(len(self.select_wave_data))
        self.packaging_progress.close()
        self.packaging_progress = None
        failure_report_name, failure_report_content = self._build_package_delete_failure_report(
            file_path,
            package_failed,
            [],
        )
        failure_report_path = None
        failure_report_error = None
        failure_report_zip_error = None
        failure_report_zip_name = None
        if failure_report_content:
            failure_report_zip_error = self._add_failure_report_to_zip(file_path, failure_report_name, failure_report_content)
            if failure_report_zip_error:
                failure_report_path, failure_report_error = self._write_failure_report_to_file(
                    file_path, failure_report_content
                )
            else:
                failure_report_zip_name = failure_report_name
        self._show_package_failure_summary(
            package_failed,
            failure_report_path=failure_report_path,
            failure_report_error=failure_report_error,
            failure_report_zip_error=failure_report_zip_error,
            failure_report_zip_name=failure_report_zip_name,
        )

    def on_clicked_delete_btn(self):
        selected_audio_data = list(self.select_wave_data.values())
        selected_count = len(selected_audio_data)
        if not selected_count:
            return

        batch_count = max(1, (selected_count + 499) // 500)
        progress_total = selected_count + batch_count + 4
        delete_progress = QProgressDialog("正在删除音频数据...", None, 0, progress_total, self)
        delete_progress.setWindowTitle("删除进度")
        delete_progress.setWindowModality(Qt.WindowModal)
        delete_progress.setWindowFlags(delete_progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        delete_progress.show()
        QApplication.processEvents()

        progress_value = 0

        def update_delete_progress(value, text=None):
            nonlocal progress_value
            progress_value = min(value, progress_total)
            if text:
                delete_progress.setLabelText(text)
            delete_progress.setValue(progress_value)
            QApplication.processEvents()

        try:
            self._stop_playback_if_selected(selected_audio_data)

            will_delete_in_db_list = list()
            for index, value in enumerate(selected_audio_data, start=1):
                file_path = self._resolve_audio_data_file_path(value)
                self._audio_duration_cache.pop(os.path.abspath(file_path), None)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        will_delete_in_db_list.append(value[0])
                    except Exception as e:
                        MessageBox.warning(self, "警告", "%s" % str(e)[:40])
                else:
                    will_delete_in_db_list.append(value[0])
                if index % 100 == 0 or index == selected_count:
                    update_delete_progress(index, f"正在处理文件... {index}/{selected_count}")

            db_progress_base = selected_count
            db_batch_count = max(1, (len(will_delete_in_db_list) + 499) // 500)

            def update_db_delete_progress(processed_count, total_count):
                batch_index = min(db_batch_count, (processed_count + 499) // 500)
                update_delete_progress(
                    db_progress_base + batch_index,
                    f"正在删除数据库记录... {processed_count}/{total_count}",
                )

            code, result = self.recording_manager.delete_audio_at_id_list(
                will_delete_in_db_list, progress_callback=update_db_delete_progress
            )
            if code == error_code.OK:
                self.logger.info("success delete audio with will_delete_in_db_list")
            else:
                self.logger.error(result)
                MessageBox.warning(self, "警告", result)
                return

            update_delete_progress(selected_count + db_batch_count + 1, "正在更新选择状态...")
            self._selected_audio_data_by_id.clear()
            self._rebuild_select_wave_data()
            self.delete_audio_data_with_id(will_delete_in_db_list)
            self._clamp_current_page()

            update_delete_progress(selected_count + db_batch_count + 2, "正在更新筛选条件...")
            self.update_filter_sets_after_deletion()

            update_delete_progress(selected_count + db_batch_count + 3, "正在刷新列表...")
            self.load_audio_data_to_view()
            self.set_select_wave_num_text(len(self.select_wave_data))
            self._rebuild_play_buttons()

            update_delete_progress(progress_total, "删除完成")
        finally:
            delete_progress.close()

    def closeEvent(self, event):
        if self.playback_controller.is_audio_playing():
            self._warn_cannot_close_while_playing()
            event.ignore()
            return
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        super().closeEvent(event)

    def reject(self):
        if self.playback_controller.is_audio_playing():
            self._warn_cannot_close_while_playing()
            return
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        super().reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
    dialog.exec()
