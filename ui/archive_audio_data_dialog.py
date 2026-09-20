import os
import time
import sys
import wave
from datetime import datetime

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import (
    QPushButton,
    QProgressDialog,
    QMessageBox,
    QFileDialog,
    QApplication,
    QLabel,
    QWidget,
    QHBoxLayout,
)
from scipy.io import wavfile as scipy_wavfile

from base.audio_record_delete import (
    count_audio_deletion_files, delete_audio_recordings, plan_audio_record_deletion,
)
from base.audio_record_filter import filter_audio_records, parse_audio_filter_metadata
from base.audio_record_package import collect_audio_package_files, select_audio_package_files
from base.log_manager import LogManager
from base.playback_controller import PlaybackController
from consts import error_code, model_consts
from consts.running_consts import DEFAULT_DIR
from ui.archive_audio_delete_dialog import ArchiveAudioDeleteDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from ui.archive_audio_package_dialog import ArchiveAudioPackageDialog
from ui.archive_audio_analysis_dialog import ArchiveAudioAnalysisDialog
from ui.audio_package_thread import AudioPackageThread
from ui.dialog_enter_policy import install_dialog_enter_policy
from ui.custom_ui_widget.audio_data_manage_dialog import (
    AudioDataManageDialog,
    extract_audio_file_name,
)


class _AudioPackageProgressDialog(QProgressDialog):
    """Keep progress visible until its owner observes the worker finish."""

    def reject(self):
        pass

    def closeEvent(self, event):
        event.ignore()


class ArchiveAudioDataDialog(AudioDataManageDialog):

    def __init__(self, logger: LogManager):
        self._audio_filter_cache = {}
        self._package_thread = None
        self._play_btn_col = 6
        self._analysis_btn_col = 7
        self._analysis_dialog = None
        self._play_text = "\u64ad\u653e"
        self._stop_text = "\u505c\u6b62"
        self._is_desc_order = False
        self._play_col_debounce_ms = 250
        self._play_col_last_action_ts = 0.0
        self._current_playing_row = None
        self._current_playing_file = None
        self._audio_duration_cache = dict()
        self._playback_started_monotonic = None
        self._current_playing_total_sec = 0.0
        self._last_remaining_display_sec = None
        self._status_row_widget = None
        # Base __init__ will trigger load_audio_data_to_view(), which may reset this label.
        # Create it early to avoid attribute access before init_ui() runs.
        self.remaining_time_label = QLabel()
        self._is_switching_playback = False
        self._playback_poll_timer = None
        self.playback_controller = PlaybackController()
        # Hover hand cursor only; no hover color logic.
        super(ArchiveAudioDataDialog, self).__init__(logger)
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        # removed blink/flash logic for play column
        self.order_btn = QPushButton(" 倒  序 ")
        self.package_btn = QPushButton(" 打  包 ")
        self.delete_btn = QPushButton(" 删  除 ")

        self.set_h_header(["", "文件名称", "产品型号", "音频标签", "采样率", "录音时间", "播放", "分析结果"])
        self.horizontalHeader().setSectionsClickable(False)

        # 而是在模型里放“播放/停止”文本项，并用 view.clicked 响应点击。
        self.data_view.clicked.connect(self._on_data_view_clicked)
        self.data_view.setMouseTracking(True)
        self.data_view.viewport().installEventFilter(self)

        self._rebuild_play_buttons()
        self._update_order_button_text()
        self.init_ui()
        install_dialog_enter_policy(self, None)

    def _refresh_audio_filter_metadata(self):
        previous = self._audio_filter_cache
        current = {}
        for row in self.all_audio_data:
            cached = previous.get(row[0])
            metadata = (
                cached[1] if cached is not None and cached[0] == row[1]
                else parse_audio_filter_metadata(row[1])
            )
            current[row[0]] = (row[1], metadata)
        self._audio_filter_cache = current
        return {identifier: entry[1] for identifier, entry in current.items()}

    def load_all_audio_data(self):
        super().load_all_audio_data()
        self._refresh_audio_filter_metadata()

    def create_filter_dialog(self, filter_config):
        return ArchiveAudioFilterDialog(
            self.all_audio_data, self._refresh_audio_filter_metadata(),
            filter_config, parent=self,
        )

    def filter_audio_data_at_filter_config(self, filter_config):
        self.filter_audio_data = filter_audio_records(
            self.all_audio_data, self._refresh_audio_filter_metadata(), filter_config,
        )

    def delete_audio_data_with_id(self, id_list):
        super().delete_audio_data_with_id(id_list)
        self._refresh_audio_filter_metadata()

    def _on_data_view_clicked(self, index):
        if not index.isValid():
            return
        if index.column() == self._analysis_btn_col:
            self._show_saved_analysis(index)
            return
        if index.column() != self._play_btn_col:
            return
        self._on_play_button_clicked(index.row())

    def _show_saved_analysis(self, index):
        if self._package_thread is not None:
            return
        if self._analysis_dialog is not None:
            self._analysis_dialog.raise_()
            self._analysis_dialog.activateWindow()
            return
        _, wav_path = index.data(Qt.UserRole)
        dialog = ArchiveAudioAnalysisDialog(wav_path, self)
        self._analysis_dialog = dialog
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.finished.connect(self._analysis_view_finished)
        dialog.open()

    def _analysis_view_finished(self, _result):
        self._analysis_dialog = None

    def init_ui(self):
        self.setWindowTitle("音频数据管理")

        self.set_bottom_layout()
        self._setup_status_row_with_remaining_label()
        self._set_remaining_label_idle()
        self.resize(880, 350)

    def mousePressEvent(self, a0):
        print(self.size())

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

            file_name = extract_audio_file_name(item[1])
            row_data_list = [None, file_name, item[2], item[5], item[3], item[4], None]
            self.add_row_data(row_data_list)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        return product_model_set, record_date_set

    def set_bottom_layout(self):
        all_show_btn = QPushButton("全部显示")

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

    def on_clicked_order_btn(self):
        self._stop_playback_if_needed()
        self.all_audio_data.reverse()
        self.filter_audio_data.reverse()
        self.load_audio_data_to_view()
        self.all_selected_checkbox.setChecked(False)
        self.all_select_flag = False
        self.set_select_wave_num_text(0)
        self._is_desc_order = not self._is_desc_order
        self._update_order_button_text()

    @staticmethod
    def _play_btn_style():
        return (
            "QPushButton {"
            " background-color: #eaf4ff;"
            " border: 1px solid #b7d3f1;"
            " border-bottom: 2px solid #79aee8;"
            " color: #1f3f63;"
            " border-radius: 2px;"
            " padding: 2px 8px;"
            " }"
            "QPushButton:hover {"
            " background-color: #d6ebff;"
            " border-color: #8fb9e8;"
            " border-bottom-color: #4f8fd4;"
            " }"
            "QPushButton:pressed {"
            " background-color: #c6e2ff;"
            " border-bottom-color: #3f7fc8;"
            " }"
            "QPushButton:disabled {"
            " background-color: #f0f7ff;"
            " color: #9a9a9a;"
            " }"
        )

    @staticmethod
    def _play_btn_blink_style():
        return """
            QPushButton {
                background-color: #ffe9e4;
                border: 1px solid #f2a897;
                border-bottom: 2px solid #e07e66;
                color: #7a2e1f;
                border-radius: 2px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #ffd9d0;
                border-color: #ea8d78;
                border-bottom-color: #d26f56;
            }
            QPushButton:pressed {
                background-color: #ffcbbf;
                border-bottom-color: #c85e47;
            }
            QPushButton:disabled {
                background-color: #ffe9e4;
                color: #9a9a9a;
            }
        """

    def _apply_play_action_item_style(self, item: QStandardItem):
        font = item.font()
        font.setUnderline(True)
        font.setBold(False)
        item.setFont(font)
        item.setToolTip("")
        item.setData(None, Qt.ForegroundRole)
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
        if model.columnCount() <= self._play_btn_col:
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
        rows = self.filter_audio_data if self.is_filter_flag else self.all_audio_data
        for row, record in enumerate(rows):
            item = QStandardItem("查看")
            item.setTextAlignment(Qt.AlignCenter)
            item.setData((record[0], record[1]), Qt.UserRole)
            self._apply_play_action_item_style(item)
            model.setItem(row, self._analysis_btn_col, item)
        model.setHeaderData(self._analysis_btn_col, Qt.Horizontal, "分析结果")
        self.data_view.resizeColumnToContents(self._analysis_btn_col)

    def eventFilter(self, watched, event):
        if watched is self.data_view.viewport():
            if event.type() == QEvent.MouseMove:
                index = self.data_view.indexAt(event.pos())
                if index.isValid() and index.column() in (self._play_btn_col, self._analysis_btn_col):
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
        if self.playback_controller.is_audio_playing() or self._current_playing_row is not None:
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
        if self._package_thread is not None:
            return
        self._stop_playback_if_needed()
        selected_paths = [row[1] for row in self.select_wave_data.values()]
        package_files = ()
        if not self.select_wave_data:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("您未选择任何音频进行导出，程序将仅导出数据库，是否确定？")
            confirm_btn = msg_box.addButton(" 确  认 ", QMessageBox.AcceptRole)
            cancel_btn = msg_box.addButton(" 取  消 ", QMessageBox.RejectRole)
            msg_box.exec_()
            if msg_box.clickedButton() != confirm_btn:
                return
        else:
            try:
                grouped = collect_audio_package_files(selected_paths)
                selected_wavs = {
                    os.path.normcase(os.path.abspath(os.path.join(DEFAULT_DIR, path)))
                    for path in selected_paths
                }
                available_wavs = {os.path.normcase(entry.source) for entry in grouped["wav"]}
                dialog = ArchiveAudioPackageDialog(
                    {kind: len(files) for kind, files in grouped.items()},
                    len(selected_paths), parent=self,
                    missing_wav_count=len(selected_wavs - available_wavs),
                )
                if dialog.exec() != dialog.Accepted:
                    return
                package_files = select_audio_package_files(grouped, dialog.selected_kinds())
            except (OSError, ValueError) as error:
                self.logger.error(f"failed to collect audio package files: {error}")
                QMessageBox.warning(self, "打包失败", f"无法收集打包文件：{error}")
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

        file_path_list = [entry.source for entry in package_files]
        archive_names = {entry.source: entry.archive_name for entry in package_files}
        file_path_list.append("database/audio_data.db")
        archive_names["database/audio_data.db"] = "audio_data.db"
        self._start_audio_package(file_path_list, file_path, archive_names)

    def _start_audio_package(self, file_paths, output_path, archive_names):
        thread = AudioPackageThread(file_paths, output_path, archive_names, self)
        self._package_thread = thread
        self.package_btn.setEnabled(False)
        self.packaging_progress = _AudioPackageProgressDialog(
            "正在打包...", None, 0, len(file_paths), self,
        )
        self.packaging_progress.setWindowTitle("打包进度")
        self.packaging_progress.setWindowModality(Qt.WindowModal)
        self.packaging_progress.setWindowFlags(self.packaging_progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.packaging_progress.setAutoClose(False)
        self.packaging_progress.setAutoReset(False)
        self.packaging_progress.show()
        thread.progress.connect(self.update_packaging_progress, Qt.QueuedConnection)
        thread.finished.connect(self._on_package_finished, Qt.QueuedConnection)
        thread.start()

    def _on_package_finished(self):
        thread = self._package_thread
        # finished is already emitted; join final thread cleanup before allowing
        # the dialog (and its child QThread) to be destroyed.
        thread.wait()
        error = thread.error_message
        self._package_thread = None
        thread.deleteLater()
        self.packaging_progress.hide()
        self.packaging_progress.deleteLater()
        self.packaging_progress = None
        self.package_btn.setEnabled(True)
        if error is not None:
            self.logger.error(f"failed to package audio data: {error}")
            QMessageBox.warning(self, "打包失败", f"音频数据打包失败：{error}")
            return
        self.set_all_checkboxes_checked([0], False)

    def on_clicked_delete_btn(self):
        if self._package_thread is not None:
            return
        selected_rows = tuple(self.select_wave_data.values())
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先勾选需要删除的录音。")
            return
        try:
            plans = plan_audio_record_deletion(selected_rows)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "无法删除", f"无法确认待删除文件：{error}")
            return
        dialog = ArchiveAudioDeleteDialog(count_audio_deletion_files(plans), len(plans), self)
        if dialog.exec() != dialog.Accepted:
            return

        self._stop_playback_if_needed()
        deleted_ids, errors = delete_audio_recordings(plans, self.recording_manager)
        for plan in plans:
            self._audio_duration_cache.pop(plan.wav_path, None)
        self.delete_audio_data_with_id(deleted_ids)
        self.product_model_set = {row[2] for row in self.all_audio_data}
        self.record_date_set = {row[4] for row in self.all_audio_data}
        if not self.all_audio_data:
            self.filter_config.clear()
            self.is_filter_flag = False
        self.load_audio_data_to_view()
        self.all_selected_checkbox.setChecked(False)
        self.all_select_flag = False
        self.set_select_wave_num_text(0)
        if errors:
            message = QMessageBox(self)
            message.setWindowTitle("部分录音未删除完成")
            message.setIcon(QMessageBox.Warning)
            message.setText(
                f"已完整删除 {len(deleted_ids)} 条录音，{len(errors)} 条未完成。\n"
                "未完成的记录已保留，部分文件可能已删除，可排除问题后重试。"
            )
            message.setDetailedText("\n".join(errors))
            message.exec_()

    def closeEvent(self, event):
        if self._package_thread is not None:
            event.ignore()
            return
        if self.playback_controller.is_audio_playing():
            self._warn_cannot_close_while_playing()
            event.ignore()
            return
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        super().closeEvent(event)

    def reject(self):
        if self._package_thread is not None:
            return
        if self.playback_controller.is_audio_playing():
            self._warn_cannot_close_while_playing()
            return
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        super().reject()

    def done(self, result):
        if self._package_thread is None:
            super().done(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
    dialog.exec()
