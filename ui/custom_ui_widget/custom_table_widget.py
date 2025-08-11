"""
========================================================================================================================
=  Description: DATA MANAGE DIALOG
=  Author: DY Gao Qiang
=  Date: 2025/08/06
=  Version: 0.25.07.01
=  FUnction List:
    1. set_select_data_num_text(text: str, list_num: tuple[int])
        In this function, you can custom displayed text in the status bar. Such as input list_num: text, (12, 23): "The
        number is %s, second number is %s", will display "The number is 12, second number is 23" in the status bar.

    2. set_h_header(h_header_list: list[str])
        In this function, you can set the header of the table. Such as h_header_list = ["ID", "Name", "Age"], will display
        +-------+-----------+---------+
        |ID     | Name      | Age     |
        +-------+-----------+---------+

    3. set_view_checked_changed(callback: callable)
        In this function, you can set the callback function when the checkbox of the table is checked or unchecked., this
        callback function

    4. set_checkable_of_column(column_num: list[int])
        In this function, you can set the checkable column of the table. Such as set_checkable_of_column([0, 1]), will
        display a checkbox in the first column and second column.

    5. set_all_checkboxes_checked(checked_column: list[int], checked=True)
        In this function, You can set all checkboxes in the table to checked or unchecked based on the number of columns.

    6. add_row_data(row_data)
        In this function, you can add a row of data to the table. Such as add_row_data(["DY001", "Jerry”, "18"]) will
        display the data in the table.
        +-------+-----------+---------+
        |ID     | Name      | Age     |
        +-------+-----------+---------+
        |DY001  | Jerry     | 18      |
        +-------+-----------+---------+

    7. resizeColumnsToContents()
        In this function, you can resize the column width to fit the content.

    8. resizeRowsToContents()
        In this function, you can resize the row height to fit the content.

    9. resizeColumnToContents(column_num: int)
        In this function, you can resize the column width to fit the content.

    10. horizontalHeader()
        In this function, you can get the horizontal header.

    11. setRowCount(row: int)
        In this function, you can set the row count.

    12. setColumnCount(column: int)
        In this function, you can set the column count.

    13. verticalHeader()
        In this function, you can get the vertical header.

    14. model()
        In this function, you can get the table model.

    15. set_column_alignment(column_num: int, alignment: Qt.AlignmentFlag)
        In this function, you can set the column alignment. Such as set_column_alignment(0, Qt.AlignCenter), as next:
        +-------+-----------+---------+
        |ID     | Name      | Age     |
        +-------+-----------+---------+
        | DY001 | Jerry     | 18      |
        +-------+-----------+---------+


=  History:
    None
========================================================================================================================
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QHeaderView, QTableView, QMessageBox, QLabel

from base.log_manager import LogManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class DataManageDialog(QDialog):

    def __init__(self):
        super(DataManageDialog, self).__init__()
        self.data_view = None

        self.select_data_num = QLabel()
        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.column_alignment_list = list()
        self.column_alignment = dict()

    def init_ui_layout(self, row: int, column: int, editable_column: list[int]):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.resize(900, 350)

        self.data_view = DataView(self.logger, row, column, editable_column)

        layout = QVBoxLayout()
        layout.addLayout(self.top_layout)
        layout.addWidget(self.data_view)
        layout.addWidget(self.select_data_num, alignment=Qt.AlignLeft)
        layout.addSpacing(4)
        layout.addLayout(self.bottom_layout)

        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qtableview_style
            + ui_style_const.qlineedit_style
        )

    def set_select_data_num_text(self, text: str, list_num: tuple[int]):
        status_text = text % list_num
        self.select_data_num.setText(status_text)

    def set_h_header(self, h_header_list: list[str]):
        self.data_view.set_h_header(h_header_list)

    def set_view_checked_changed(self, callback: callable):
        self.data_view.view_checked_changed.connect(callback)

    def set_checkable_of_column(self, column_num: list[int]):
        self.data_view.set_checkable_of_column(column_num)

    def set_all_checkboxes_checked(self, checked_column: list[int], checked=True):
        self.data_view.set_all_checkboxes_checked(checked_column, checked)

    def add_row_data(self, data):
        row_item_list = list()
        for column, item in enumerate(data):
            if item is None:
                view_item = QStandardItem()
            else:
                view_item = QStandardItem(str(item))
            if column in self.data_view.set_checkable_column:
                view_item.setCheckable(True)
            if column in self.column_alignment_list:
                view_item.setTextAlignment(self.column_alignment[str(column)])
            row_item_list.append(view_item)
        self.data_view.model().appendRow(row_item_list)

    def resizeColumnsToContents(self):
        self.data_view.resizeColumnsToContents()

    def resizeRowsToContents(self):
        self.data_view.resizeRowsToContents()

    def resizeColumnToContents(self, column_num: int):
        self.data_view.resizeColumnToContents(column_num)

    def horizontalHeader(self):
        return self.data_view.horizontalHeader()

    def setRowCount(self, row: int):
        self.data_view.model().setRowCount(row)

    def setColumnCount(self, column: int):
        self.data_view.model().setColumnCount(column)

    def verticalHeader(self):
        return self.data_view.verticalHeader()

    def model(self):
        return self.data_view.model()

    def set_column_alignment(self, column_num: int, alignment: Qt.AlignmentFlag):
        self.column_alignment[str(column_num)] = alignment
        self.column_alignment_list.append(column_num)

        model = self.model()
        for row in range(model.rowCount()):
            model.item(row, column_num).setTextAlignment(alignment)


class DataView(QTableView):
    view_checked_changed = pyqtSignal(QStandardItem, bool)

    def __init__(self, logger: LogManager, row: int, column: int, editable_column: list[int]):
        super(DataView, self).__init__()

        self.logger = logger
        self.audio_data_model = CustomStandardItemModel(row, column, editable_column)

        self.setModel(self.audio_data_model)
        self.model().itemChanged.connect(self.on_item_changed)
        self.is_send_signal = True
        self.set_checkable_flag = False
        self.set_checkable_column = list[int]
        self.setFocusPolicy(Qt.NoFocus)

    def set_h_header(self, h_header_list: list[str]):
        self.model().setHorizontalHeaderLabels(h_header_list)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def set_checkable_of_column(self, column_num: list[int]):
        self.set_checkable_column = column_num
        for row in range(self.model().rowCount()):
            item = self.model().item(row, column_num)
            item.setCheckable(True)

    def set_all_checkboxes_checked(self, checked_column: list[int], checked: bool):
        model = self.model()
        for column in checked_column:
            if column in self.set_checkable_column:
                for row in range(model.rowCount()):
                    index = model.index(row, column)
                    if model.data(index, Qt.CheckStateRole) is not None:
                        model.setData(index, Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
            else:
                QMessageBox.warning(self, "错误", "请选择可勾选列")
                return

    def on_item_changed(self, item):
        if self.is_send_signal is False:
            return
        if item.column() in self.set_checkable_column:
            checked = item.checkState() == Qt.Checked
            self.view_checked_changed.emit(item, checked)

    def del_model_row_with_list(self, row_list: list):
        new__row_list = sorted(row_list, reverse=True)
        for row in new__row_list:
            self.model().removeRow(row)


class CustomStandardItemModel(QStandardItemModel):

    def __init__(self, rows, columns, editable_column: list[int], parent=None):
        super().__init__(rows, columns, parent)
        self.editable_column = editable_column

    def flags(self, index):
        if index.isValid():
            if index.column() in self.editable_column:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        return super().flags(index)

    def setData(self, index, value, role=Qt.EditRole):
        return super().setData(index, value, role)
