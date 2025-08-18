import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QProgressDialog, QMessageBox, QFileDialog, QApplication

from base.file_ops import FileOps
from base.log_manager import LogManager
from consts import error_code, model_consts
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_data_manage_dialog import audioDataManageDialog


class ArchiveAudioDataDialog(audioDataManageDialog):

    def __init__(self, logger: LogManager):
        super(ArchiveAudioDataDialog, self).__init__(logger)
        self.package_btn = QPushButton(" 打  包 ")
        self.delete_btn = QPushButton(" 删  除 ")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("音频数据管理")

        self.set_bottom_layout()

    def set_bottom_layout(self):
        all_show_btn = QPushButton("全部显示")

        all_show_btn.clicked.connect(self.show_all_wave)
        self.package_btn.clicked.connect(self.on_clicked_package_btn)
        self.delete_btn.clicked.connect(self.on_clicked_delete_btn)

        self.bottom_layout.addWidget(all_show_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(self.package_btn)
        self.bottom_layout.addWidget(self.delete_btn)

    def on_clicked_package_btn(self):
        if not self.select_wave_data:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("您未选择任何音频进行导出，程序将仅导出数据库，是否确定？")
            confirm_btn = msg_box.addButton(" 确  认 ", QMessageBox.AcceptRole)
            cancel_btn = msg_box.addButton(" 取  消 ", QMessageBox.RejectRole)
            msg_box.exec_()
            if msg_box.clickedButton() != confirm_btn:
                return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存位置",
            os.path.join(model_consts.STORED_PACKAGE_PATH, "audio_data_export"),
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
        is_delete_item_list = list()
        will_delete_in_db_list = list()
        for key, value in self.select_wave_data.items():
            file_path = DEFAULT_DIR + value[1]
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
    dialog.exec()
