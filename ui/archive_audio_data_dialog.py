import os
import sys
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QPushButton, QProgressDialog, QMessageBox, QFileDialog, QApplication

from base.file_ops import FileOps
from base.log_manager import LogManager
from base.playback_controller import PlaybackController
from consts import error_code, model_consts
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_data_manage_dialog import audioDataManageDialog


class ArchiveAudioDataDialog(audioDataManageDialog):

    def __init__(self, logger: LogManager):
        self._play_col = 6
        self._play_text = "播放"
        self._stop_text = "停止"
        self._current_playing_row = None
        self._current_playing_file = None
        self.playback_controller = PlaybackController()
        self._playback_poll_timer = None
        super(ArchiveAudioDataDialog, self).__init__(logger)
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        self.data_view.clicked.connect(self._on_data_view_clicked)
        self.package_btn = QPushButton(" 打  包 ")
        self.delete_btn = QPushButton(" 删  除 ")
        self.init_ui()

    def get_table_column_count(self):
        return 8

    def get_header_labels(self):
        return ["", "文件名称", "产品型号", "音频标签", "采样率", "录音时间", "播放", "激励信号"]

    def init_ui(self):
        self.setWindowTitle("音频数据管理")

        self.set_bottom_layout()

    def load_audio_data_to_view(self):
        self._stop_playback_if_needed()
        super().load_audio_data_to_view()
        self._rebuild_play_cells()

    def show_all_wave(self):
        self._stop_playback_if_needed()
        super().show_all_wave()
        self._rebuild_play_cells()

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
            row_stimulus_name = stimulus_name.get(item[7], "无激励信号")
            row_data_list = [None, file_name, item[2], item[5], item[3], item[4], self._play_text, row_stimulus_name]
            self.add_row_data(row_data_list)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        return product_model_set, record_date_set

    def set_bottom_layout(self):
        all_show_btn = QPushButton("全部显示")

        all_show_btn.clicked.connect(self.show_all_wave)
        self.package_btn.clicked.connect(self.on_clicked_package_btn)
        self.delete_btn.clicked.connect(self.on_clicked_delete_btn)

        self.bottom_layout.addWidget(all_show_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(self.package_btn)
        self.bottom_layout.addWidget(self.delete_btn)

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

    def _rebuild_play_cells(self):
        model = self.model()
        if model is None or model.columnCount() <= self._play_col:
            return

        for row in range(model.rowCount()):
            item = model.item(row, self._play_col)
            if item is None:
                item = QStandardItem()
                model.setItem(row, self._play_col, item)
            item.setEditable(False)
            item.setTextAlignment(Qt.AlignCenter)
        self._refresh_play_cell_states()

    def _refresh_play_cell_states(self):
        model = self.model()
        if model is None or model.columnCount() <= self._play_col:
            return

        is_playing = self.playback_controller.is_audio_playing()
        for row in range(model.rowCount()):
            item = model.item(row, self._play_col)
            if item is None:
                item = QStandardItem()
                model.setItem(row, self._play_col, item)
            item.setText(self._stop_text if is_playing and row == self._current_playing_row else self._play_text)
            item.setTextAlignment(Qt.AlignCenter)

    def _clear_playing_state(self):
        self._current_playing_row = None
        self._current_playing_file = None
        if self._playback_poll_timer is not None:
            self._playback_poll_timer.stop()
        self._refresh_play_cell_states()

    def _stop_playback_if_needed(self):
        if self.playback_controller.is_audio_playing() or self._current_playing_row is not None:
            self.playback_controller.stop_audio_playback()
        self._clear_playing_state()

    def _on_data_view_clicked(self, index):
        if not index.isValid() or index.column() != self._play_col:
            return
        self._toggle_row_playback(index.row())

    def _toggle_row_playback(self, row):
        row_path = self._resolve_row_file_path(row)
        if not row_path:
            QMessageBox.warning(self, "提示", "未找到该行对应音频文件路径")
            self._clear_playing_state()
            return

        if self.playback_controller.is_audio_playing() and row == self._current_playing_row:
            self.playback_controller.stop_audio_playback()
            self._clear_playing_state()
            return

        if self.playback_controller.is_audio_playing():
            self.playback_controller.stop_audio_playback()
            self._clear_playing_state()

        code, msg = self.playback_controller.start_audio_playback(row_path)
        if code != error_code.OK:
            QMessageBox.warning(self, "提示", msg)
            self._clear_playing_state()
            return

        self._current_playing_row = row
        self._current_playing_file = row_path
        self._refresh_play_cell_states()
        if self._playback_poll_timer is not None and not self._playback_poll_timer.isActive():
            self._playback_poll_timer.start()

    def _on_playback_poll_timeout(self):
        if not self.playback_controller.is_audio_playing():
            self._clear_playing_state()
            return

        playing_file = self.playback_controller.get_current_playing_file()
        if not playing_file or not self._current_playing_file:
            self._clear_playing_state()
            return

        if os.path.abspath(playing_file) != os.path.abspath(self._current_playing_file):
            self._clear_playing_state()

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
            self._stop_playback_if_needed()

        is_delete_item_list = list()
        will_delete_in_db_list = list()
        for key, value in self.select_wave_data.items():
            raw_path = value[1]
            if os.path.isabs(raw_path):
                file_path = os.path.abspath(raw_path)
            else:
                file_path = os.path.abspath(os.path.join(DEFAULT_DIR, raw_path))
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
        self._rebuild_play_cells()

    def closeEvent(self, event):
        self._stop_playback_if_needed()
        super().closeEvent(event)

    def reject(self):
        self._stop_playback_if_needed()
        super().reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
    dialog.exec()
