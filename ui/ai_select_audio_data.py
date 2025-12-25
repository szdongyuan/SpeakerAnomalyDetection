"""
========================================================================================================================
= Description: select audio data use to train/test ai model
= Author: DY Gao Qiang
= Date: 2025/08/11
= Version: 0.25.07.01
= Function Description:
    This module only needs to contain a table, which is used to display the audio recordings marked as "ok" or "ng" in
    the program. Users can check the desired audio data and click confirm to obtain the paths of these audio data.

= History:
    2025/08/11: Set up the project.
========================================================================================================================
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton

from base.log_manager import LogManager
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog


class SelectAudioDataView(AudioDataManageDialog):
    def __init__(self, logger: LogManager, select_data_list: list = list(), hide_select_not_label = True):
        super(SelectAudioDataView, self).__init__(logger, hide_select_not_label)

        self.logger = logger
        self.clicked_ok_flag = False
        self.init_select_data_list = select_data_list
        self.select_row_list = list()
        self.select_ng_num = int()
        self.select_ok_num = int()

        self.filter_signal.connect(self.refresh_data)

        self.init_ui()

    def init_ui(self):

        self.set_bottom_layout()
        self.get_select_row_by_init_list()
        self.set_checkacle_of_row(row_num=self.select_row_list)

    def set_checkacle_of_row(self, row_num: list[int]):
        for row in row_num:
            item = self.model().item(row, 0)
            item.setCheckState(Qt.Checked)

    def get_select_row_by_init_list(self):
        # 建立文件路径到行号的索引字典，时间复杂度 O(m)
        file_path_to_row = {}
        for row, item in enumerate(self.all_audio_data):
            file_path = item[1]
            # 如果同一个文件路径出现多次，保留第一个匹配的行号
            if file_path not in file_path_to_row:
                file_path_to_row[file_path] = row

        # 根据索引字典查找匹配的行号，时间复杂度 O(n)
        for key in self.init_select_data_list.keys():
            if key in file_path_to_row:
                self.select_row_list.append(file_path_to_row[key])

    def refresh_data(self):
        """刷新数据：在筛选后重新设置选中状态"""
        self.select_row_list.clear()
        self.select_ok_num = 0
        self.select_ng_num = 0

    def load_audio_data_to_view(self):
        self.all_audio_data = [i for i in self.all_audio_data if any(v in i for v in ["OK", "NG"])]
        super().load_audio_data_to_view()

    def on_all_selected_changed(self):
        if self.all_select_flag is True:
            self.set_all_checkboxes_checked([0], False)
            self.all_select_flag = False
        elif self.all_select_flag is False:
            self.set_all_checkboxes_checked([0], True)
            self.all_select_flag = True

    def set_bottom_layout(self):
        all_show_btn = QPushButton("全部显示")
        ok_btn = QPushButton(" 确  定 ")
        cancel_btn = QPushButton(" 取  消 ")

        all_show_btn.clicked.connect(self.show_all_wave)
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn.clicked.connect(self.on_click_cancel_btn)

        self.bottom_layout.addWidget(all_show_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(cancel_btn, alignment=Qt.AlignRight)
        self.bottom_layout.addWidget(ok_btn, alignment=Qt.AlignRight)

    def on_row_checkbox_toggled(self, item, is_checked):
        super().on_row_checkbox_toggled(item, is_checked)

        if self.is_filter_flag is True:
            audio_data = self.filter_audio_data
        elif self.is_filter_flag is False:
            audio_data = self.all_audio_data

        row = item.row()
        if is_checked is True:
            self.select_row_list.append(row)
            if "OK" == audio_data[row][5]:
                self.select_ok_num += 1
            elif "NG" == audio_data[row][5]:
                self.select_ng_num += 1
        elif is_checked is False:
            self.select_row_list.remove(row)
            if "OK" == audio_data[row][5]:
                self.select_ok_num -= 1
            elif "NG" == audio_data[row][5]:
                self.select_ng_num -= 1

    def on_click_ok_btn(self):
        self.clicked_ok_flag = True
        self.close()

    def on_click_cancel_btn(self):
        self.close()

    def exec(self):
        super().exec()
        data = dict()
        if self.clicked_ok_flag:
            for i in self.select_wave_data.values():
                data[i[1]] = [i[3], i[5]]
            return (data, (self.select_ok_num, self.select_ng_num))
        return None


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    audio_data_view = SelectAudioDataView(LogManager().set_log_handler("core"))
    audio_data_view.show()
    a = audio_data_view.exec()
