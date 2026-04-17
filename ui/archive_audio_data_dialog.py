import os
import time
import sys
import wave
from datetime import datetime

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QStandardItem, QBrush, QColor
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QFileDialog, QApplication, QWidget, QHBoxLayout
from scipy.io import wavfile as scipy_wavfile

from base.file_ops import FileOps
from base.log_manager import LogManager
from base.playback_controller import PlaybackController
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog
from ui.custom_ui_widget.widgets import Label, PushButton


class ArchiveAudioDataDialog(AudioDataManageDialog):

    def __init__(self, logger: LogManager):
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
        self.package_btn = PushButton(" 打  包 ")
        self.delete_btn = PushButton(" 删  除 ")

        self.set_h_header(["", "文件名称", "产品型号", "音频标签", "采样率", "录音时间", "播放"])

        # 而是在模型里放“播放/停止”文本项，并用 view.clicked 响应点击。
        self.data_view.clicked.connect(self._on_data_view_clicked)
        self.data_view.setMouseTracking(True)
        self.data_view.viewport().installEventFilter(self)

        self._rebuild_play_buttons()
        self._update_order_button_text()
        self.init_ui()

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
        self._set_remaining_label_idle()

        window_width = ui_style_const.scale_size_px(1060)
        window_height = ui_style_const.scale_size_px(700)
        self.resize(window_width, window_height)

    def load_audio_data_to_view(self):
        self._stop_playback_if_needed()
        super().load_audio_data_to_view()
        self._rebuild_play_buttons()

    def show_all_wave(self):
        self._stop_playback_if_needed()
        super().show_all_wave()
        self._rebuild_play_buttons()

    def load_audio_data_to_model(self, audio_data, stimulus_name):
        self.setRowCount(0)
        if not audio_data:
            return set(), set()

        product_model_set = set()
        record_date_set = set()
        for item in audio_data:
            product_model_set.add(item[2])
            record_date_set.add(item[4])

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
        self.package_btn.clicked.connect(self.on_clicked_package_btn)
        self.delete_btn.clicked.connect(self.on_clicked_delete_btn)

        self.bottom_layout.addWidget(all_show_btn)
        self.bottom_layout.addWidget(self.order_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(self.package_btn)
        self.bottom_layout.addWidget(self.delete_btn)

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

    def _get_mirror_index(self, index, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if index < 0 or index >= capacity:
            raise IndexError("index out of range")
        return capacity - 1 - index

    def on_clicked_order_btn(self):
        self._click_order = True
        if self._current_playing_row is not None:
            self._current_playing_row = self._get_mirror_index(self._current_playing_row, len(self.all_audio_data))

        self.all_audio_data.reverse()
        self.filter_audio_data.reverse()
        self.load_audio_data_to_view()
        self.all_selected_checkbox.setChecked(False)
        self.all_select_flag = False
        self.set_select_wave_num_text(0)
        self._is_desc_order = not self._is_desc_order
        self._update_order_button_text()
        self._click_order = False

    def _apply_play_action_item_style(self, item: QStandardItem):
        font = item.font()
        font.setUnderline(True)
        font.setBold(False)
        item.setFont(font)
        item.setToolTip("")
        item.setData(QBrush(QColor(193, 236, 250)), Qt.ForegroundRole)
        item.setData(None, Qt.BackgroundRole)

    def _resolve_row_file_path(self, row):
        audio_data = self.filter_audio_data if self.is_filter_flag else self.all_audio_data
        if row < 0 or row >= len(audio_data):
            return None
        raw_path = audio_data[row][1]
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
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("提示")
        msg_box.setText("当前有音频正在播放，请先停止后再播放其他音频。")
        msg_box.setIcon(QMessageBox.NoIcon)
        msg_box.addButton("确定", QMessageBox.AcceptRole)
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
                QMessageBox.warning(self, "提示", "未找到该行对应音频文件路径")
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

            code, msg = self.playback_controller.start_audio_playback(row_path)
            if code != error_code.OK:
                QMessageBox.warning(self, "提示", msg)
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
            print(e)
        finally:
            self._is_switching_playback = False

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
        QMessageBox.warning(self, "提示", "正在播放，请先停止播放后再退出")

    def on_clicked_package_btn(self):
        self._stop_playback_if_needed()
        if not self.select_wave_data:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("您未选择任何音频进行导出，程序将仅导出数据库，是否确定？")
            confirm_btn = msg_box.addButton(" 确  认 ", QMessageBox.AcceptRole)
            cancel_btn = msg_box.addButton(" 取  消 ", QMessageBox.RejectRole)
            msg_box.exec_()
            if msg_box.clickedButton() != confirm_btn:
                return

        file_name = "audio_data_export_%s" % datetime.now().strftime("%Y%m%d")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存位置",
            os.path.join(model_consts.STORED_PACKAGE_PATH, file_name),
            "压缩文件 (*.zip)",
        )

        if not file_path:
            return

        if not file_path.endswith(".zip"):
            file_path += ".zip"

        file_path_list = [i[1] for i in self.select_wave_data.values()]
        file_path_list.append("database/audio_data.db")
        self.packaging_progress = QProgressDialog("正在打包...", None, 0, len(file_path_list), self)
        self.packaging_progress.setWindowTitle("打包进度")
        self.packaging_progress.setWindowModality(Qt.WindowModal)
        self.packaging_progress.setWindowFlags(self.packaging_progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.packaging_progress.show()
        QApplication.processEvents()

        FileOps.create_zip_with_files(file_path_list, file_path, progress_callback=self.update_packaging_progress)

        self.set_all_checkboxes_checked([0], False)
        self.packaging_progress.close()
        self.packaging_progress = None

    def on_clicked_delete_btn(self):
        current_file = self.playback_controller.get_current_playing_file()
        if current_file and self.select_wave_data:
            selected_paths = []
            for value in self.select_wave_data.values():
                selected_path = value[1]
                if os.path.isabs(selected_path):
                    selected_paths.append(os.path.abspath(selected_path))
                else:
                    selected_paths.append(os.path.abspath(os.path.join(DEFAULT_DIR, selected_path)))
            if os.path.abspath(current_file) in selected_paths:
                self._stop_playback_if_needed()

        is_delete_item_list = list()
        will_delete_in_db_list = list()
        for key, value in self.select_wave_data.items():
            raw_path = value[1]
            if os.path.isabs(raw_path):
                file_path = os.path.abspath(raw_path)
            else:
                file_path = os.path.abspath(os.path.join(DEFAULT_DIR, raw_path))
            self._audio_duration_cache.pop(os.path.abspath(file_path), None)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    is_delete_item_list.append(int(key))
                    will_delete_in_db_list.append(value[0])
                except Exception as e:
                    QMessageBox.warning(self, "警告", "%s" % str(e)[:40])
            else:
                will_delete_in_db_list.append(value[0])
        code, result = self.recording_manager.delete_audio_at_id_list(will_delete_in_db_list)
        if code == error_code.OK:
            self.logger.info("success delete audio with will_delete_in_db_list")
        else:
            self.logger.error(result)
            return
        self.data_view.del_model_row_with_list(is_delete_item_list)
        self.select_wave_data = dict()
        self.delete_audio_data_with_id(will_delete_in_db_list)
        self.set_select_wave_num_text(0)
        self.update_filter_sets_after_deletion()
        self._rebuild_play_buttons()

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
