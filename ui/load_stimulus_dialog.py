import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QApplication

from base.log_manager import LogManager
from base.stimulus_signal_management import StimulusSignalManagement
from base.stimulus_signal.methods import normalize_stimulus_method
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code
from ui.custom_ui_widget.custom_table_widget import DataManageDialog
from ui.custom_ui_widget.widgets import PushButton, MessageBox


class LoadStimulusDialog(DataManageDialog):
    # Define stimulus signal types
    STIMULUS_DICT = {
        "chirp": "啁啾",
        "step": "步进",
        "frequency_stepped": "step(sc)",
        "noise": "噪音",
    }
    # Mapping for stimulus signal chinese names
    STIMULUS_DICT_2 = {
        "log": "对数",
        "linear": "线性",
        "mirror_log": "对数镜像",
        "mirror_linear": "线性镜像",
        "white_noise": "白噪音",
        "pink_noise": "粉噪音",
        "octave": "倍频程",
        "custom_linear": "自定义线性",
        "custom_log": "自定义对数",
    }

    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.is_clicked_ok = False
        self.select_stimulus_row = None
        self.is_edit_item = True
        self._suppress_data_changed = False

        # Constraints aligned with StimulusWindow
        self.ALLOWED_SAMPLE_RATES = {44100, 48000}
        self.FREQ_MIN, self.FREQ_MAX = 10, 24000
        self.REPEAT_MIN, self.REPEAT_MAX = 1, 10
        self.STEPS_MIN, self.STEPS_MAX = 1, 100
        self.TOTAL_TIME_MIN, self.TOTAL_TIME_MAX = 0.5, 60.0
        self.VOLTAGE_MIN = 0.1

        self.loaded_stimulus = self.load_stimulus_config_from_db()

        # Enable editing for: Name(1), Repeat(4), Start Freq(5), Stop Freq(6), Sample Rate(7),
        # Total Time(8), Steps(9), Voltage(11). Keep mode(2), type(3) and voltage type(10) read-only.
        self.init_ui_layout(0, 12, [1, 4, 5, 6, 7, 8, 9, 11])

        self.set_view_checked_changed(self.on_row_checkbox_toggled)
        self.model().dataChanged.connect(self.is_edit_model_item)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("加载激励信号")
        self.set_checkable_of_column([0])
        self.set_h_header(
            [
                "",
                "配置名称",
                "信号模式",
                "信号类型",
                "重复次数",
                "起始频率",
                "截止频率",
                "采样频率",
                "信号时长",
                "步进数量",
                "电压类型",
                "电压(V)",
            ]
        )
        self.verticalHeader().setVisible(False)
        self.insert_stimulus_config_to_table(self.loaded_stimulus)
        self.set_column_alignment(4, Qt.AlignCenter)
        self.set_column_alignment(9, Qt.AlignCenter)
        self.set_column_alignment(10, Qt.AlignCenter)
        self.set_column_alignment(11, Qt.AlignCenter)
        self.select_data_num.hide()

        self.set_bottom_layout()
        self.layout().setContentsMargins(13, 20, 13, 13)

    def set_bottom_layout(self):
        del_config_btn = PushButton("删除配置")
        cancel_btn = PushButton(" 取  消 ")
        ok_btn = PushButton(" 确  定 ")

        del_config_btn.clicked.connect(self.del_config)
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.close)

        self.bottom_layout.addWidget(del_config_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(cancel_btn)
        self.bottom_layout.addWidget(ok_btn)

    def on_row_checkbox_toggled(self, item, is_checked):
        if is_checked:
            if self.select_stimulus_row is not None:
                self.model().item(self.select_stimulus_row, 0).setCheckState(Qt.Unchecked)
            self.select_stimulus_row = item.row()
        else:
            self.select_stimulus_row = None

    @staticmethod
    def load_stimulus_config_from_db():
        """
        Loads stimulus configuration data from the database and converts it into a specific data structure.

        This function calls the `query_all_stimulus_info` method of the `StimulusSignalManagement` class
        to retrieve all stimulus signal information from the database. If the query is successful, it
        transforms each stimulus signal into a dictionary and appends it to the `stimulus_list` for return.

        Returns:
            list: A list containing all stimulus configuration information. Each stimulus configuration is a dictionary
                with the following key-value pairs:
                    - 'name': The name of the stimulus signal, formatted as "stimulus_{stimulus_method}_{index}".
                    - 'stimulus_method': The method of the stimulus signal.
                    - 'stimulus_type': The type of the stimulus signal.
                    - 'start_freq': The starting frequency of the stimulus signal.
                    - 'stop_freq': The ending frequency of the stimulus signal.
                    - 'total_time': The total duration of the stimulus signal.
                    - 'repeat_times': The number of repetitions of the stimulus signal.
                    - 'sample_rate': The sampling rate of the stimulus signal.
                    - 'num_steps': The number of steps in the stimulus signal.
                    - 'stimulus_name': The name of the configuration for the stimulus signal.
                    - 'is_default': Indicates whether it is the default stimulus signal.
        """
        stimulus_list = []
        query_code, query_data = StimulusSignalManagement().query_all_stimulus_info()
        if query_code == error_code.OK:
            for idx, info in enumerate(query_data):
                query_data_idx = query_data[idx]
                if isinstance(query_data_idx, dict):
                    stimulus = dict(query_data_idx)
                    voltage = stimulus.get("voltage")
                    stimulus["voltage"] = float(voltage) if voltage is not None else 0.1
                else:
                    stimulus = {
                        "stimulus_id": query_data_idx[0],
                        "stimulus_method": query_data_idx[1],
                        "stimulus_type": query_data_idx[2],
                        "start_freq": query_data_idx[4],
                        "stop_freq": query_data_idx[5],
                        "total_time": query_data_idx[7],
                        "repeat_times": query_data_idx[3],
                        "sample_rate": query_data_idx[6],
                        "num_steps": query_data_idx[8],
                        "voltage_type": query_data_idx[9],
                        "voltage": float(query_data_idx[10]) if query_data_idx[10] is not None else 0.1,
                        "is_default": query_data_idx[11],
                        "stimulus_name": query_data_idx[12],
                    }
                stimulus_list.append(stimulus)
        return stimulus_list

    def insert_stimulus_config_to_table(self, stimulus_info: list[dict]):
        self.setRowCount(0)
        for stimulus in stimulus_info:
            config_list = [
                None,
                stimulus["stimulus_name"],
                stimulus["stimulus_method"],
                stimulus["stimulus_type"],
                stimulus["repeat_times"],
                stimulus["start_freq"],
                stimulus["stop_freq"],
                stimulus["sample_rate"],
                stimulus["total_time"],
                stimulus["num_steps"],
                stimulus["voltage_type"],
                f"{stimulus['voltage']:.2f}",
            ]
            self.add_row_data(config_list)

    def get_selected_stimulus_config(self):
        if self.select_stimulus_row is None:
            return None
        # Create a copy to avoid mutating the original data
        selected = self.loaded_stimulus[self.select_stimulus_row]
        if selected.get("step_sc_row_state") == "invalid_metadata":
            MessageBox.warning(self, "提示", "该 step(sc) 配置元数据无效，无法加载")
            return None
        if normalize_stimulus_method(selected.get("stimulus_method")) == "frequency_stepped":
            payload = selected.get("stimulus_payload")
            if not isinstance(payload, dict):
                MessageBox.warning(self, "提示", "该 step(sc) 配置元数据无效，无法加载")
                return None
            result = payload.copy()
        else:
            result = selected.copy()
        result.pop("stimulus_name")
        result.pop("is_default")
        result.pop("stimulus_id")
        result.pop("stimulus_metadata_json", None)
        result.pop("stimulus_payload", None)
        result.pop("step_sc_row_state", None)
        return result

    def is_edit_model_item(self, topLeft, bottomRight, roles):
        if self._suppress_data_changed:
            return
        if Qt.EditRole not in roles:
            return
        # Handle each edited cell within the changed index range
        for row in range(topLeft.row(), bottomRight.row() + 1):
            for col in range(topLeft.column(), bottomRight.column() + 1):
                index = self.model().index(row, col)
                self.on_data_changed(index)

    def on_data_changed(self, index: QStandardItem):
        """Validate, persist and reflect edits per cell.

        Columns mapping:
        1: stimulus_name (string, non-empty, unique)
        4: repeat_times (int 1-10)
        5: start_freq (int 10-24000)
        6: stop_freq (int 10-24000)
        7: sample_rate (int in {44100, 48000})
        8: total_time (float 0.5-60.0)
        9: num_steps (int 1-100)
        11: voltage (float 0.1 - max_voltage if available)
        """
        row = index.row()
        col = index.column()
        new_text = index.data()

        # Guard: ignore empty edits for name; revert to previous
        if new_text is None:
            return

        # Cols that are intentionally read-only are not editable via flags, but guard just in case
        if col in (0, 2, 3, 10):
            return

        # Helper to revert display to previous value without re-triggering handler
        def revert_to_previous():
            self._suppress_data_changed = True
            try:
                prev_val = self._get_display_value_for_cell(row, col)
                self.model().setData(index, prev_val)
            finally:
                self._suppress_data_changed = False

        # Name change
        if col == 1:
            new_name = str(new_text).strip()
            if new_name == "":
                MessageBox.warning(self, "提示", "配置名称不能为空")
                revert_to_previous()
                return
            stimulus_id = self.loaded_stimulus[row].get("stimulus_id")
            update_info = {"stimulus_id": stimulus_id, "new_name": new_name}
            code, msg = StimulusSignalManagement().update_stimulus_info_to_db(update_info)
            if code == error_code.OK:
                self.loaded_stimulus[row]["stimulus_name"] = new_name
                self.logger.info(msg)
            else:
                self.logger.error(msg)
                MessageBox.warning(self, "重命名失败", msg)
                revert_to_previous()
            return

        if normalize_stimulus_method(self.loaded_stimulus[row].get("stimulus_method")) == "frequency_stepped":
            MessageBox.warning(self, "提示", "step(sc) 配置不支持在列表中直接编辑参数")
            revert_to_previous()
            return

        # Map column to db field and validator
        field_by_col = {
            4: "repeat_times",
            5: "start_freq",
            6: "stop_freq",
            7: "sample_rate",
            8: "total_time",
            9: "num_steps",
            11: "voltage",
        }
        field = field_by_col.get(col)
        if not field:
            return

        # Parse and validate according to StimulusWindow constraints
        try:
            if field in ("repeat_times", "start_freq", "stop_freq", "sample_rate", "num_steps"):
                # Integer fields
                value = int(float(str(new_text).strip()))
                if field == "repeat_times" and not (self.REPEAT_MIN <= value <= self.REPEAT_MAX):
                    raise ValueError(f"重复次数需在 {self.REPEAT_MIN}-{self.REPEAT_MAX} 范围内")
                if field in ("start_freq", "stop_freq") and not (self.FREQ_MIN <= value <= self.FREQ_MAX):
                    raise ValueError(f"频率需在 {self.FREQ_MIN}-{self.FREQ_MAX} Hz 范围内")
                if field == "sample_rate" and value not in self.ALLOWED_SAMPLE_RATES:
                    raise ValueError("采样频率仅支持 44100 或 48000")
                if field == "num_steps" and not (self.STEPS_MIN <= value <= self.STEPS_MAX):
                    raise ValueError(f"步进数量需在 {self.STEPS_MIN}-{self.STEPS_MAX} 范围内")
            elif field == "total_time":
                value = float(str(new_text).strip())
                if not (self.TOTAL_TIME_MIN <= value <= self.TOTAL_TIME_MAX):
                    raise ValueError(f"信号时长需在 {self.TOTAL_TIME_MIN}-{self.TOTAL_TIME_MAX} s 范围内")
                # Normalize format to 1 decimal
                value = round(value, 1)
            elif field == "voltage":
                value = float(str(new_text).strip())
                max_v = self._get_max_input_voltage()
                if value < self.VOLTAGE_MIN:
                    raise ValueError(f"电压需不小于 {self.VOLTAGE_MIN} V")
                if max_v is not None and value > max_v:
                    raise ValueError(f"电压不能超过标定上限 {max_v} V")
                # Normalize to 2 decimals for display
                value = round(value, 2)
            else:
                return
        except ValueError as ve:
            MessageBox.warning(self, "输入无效", str(ve))
            revert_to_previous()
            return
        except (TypeError, AttributeError) as e:
            self.logger.error(f"Unexpected error in on_data_changed: {e}")
            MessageBox.warning(self, "输入无效", "请输入有效的数值")
            revert_to_previous()
            return

        # Check if value actually changed before updating DB
        current_value = self.loaded_stimulus[row].get(field)
        value_changed = False
        if field in ("total_time", "voltage"):
            # Use epsilon comparison for floats
            value_changed = abs(float(current_value) - float(value)) > 1e-6
        else:
            # Direct comparison for integers
            value_changed = int(current_value) != int(value)

        if not value_changed:
            # Value unchanged, just normalize display
            display_val = self._format_display_value(field, value)
            self._suppress_data_changed = True
            try:
                self.model().setData(index, display_val)
            finally:
                self._suppress_data_changed = False
            return

        # Persist to DB
        stimulus_id = self.loaded_stimulus[row].get("stimulus_id")
        code, msg = StimulusSignalManagement().update_stimulus_params_to_db(stimulus_id, {field: value})
        if code != error_code.OK:
            self.logger.error(msg)
            MessageBox.warning(self, "保存失败", msg)
            revert_to_previous()
            return

        # Success: update in-memory and normalize cell display
        self.loaded_stimulus[row][field] = value
        display_val = self._format_display_value(field, value)
        self._suppress_data_changed = True
        try:
            self.model().setData(index, display_val)
        finally:
            self._suppress_data_changed = False
        self.logger.info("Stimulus parameter updated: %s=%s" % (field, value))
        return

    def _get_max_input_voltage(self):
        """Fetch max voltage from calibration file; return None if unavailable."""
        try:
            code, data = SoundcardCalibrationManager().load_data_from_json("calibration_coefficients.json")
            if code == error_code.OK:
                return float(data.get("max_voltage", 0.0))
        except Exception:
            pass
        return None

    def _format_display_value(self, field: str, value):
        if field == "total_time":
            return f"{float(value):.1f}"
        if field == "voltage":
            return f"{float(value):.2f}"
        return (
            str(int(value))
            if isinstance(value, int)
            or field in {"repeat_times", "start_freq", "stop_freq", "sample_rate", "num_steps"}
            else str(value)
        )

    def _get_display_value_for_cell(self, row: int, col: int):
        field_by_col = {
            1: "stimulus_name",
            4: "repeat_times",
            5: "start_freq",
            6: "stop_freq",
            7: "sample_rate",
            8: "total_time",
            9: "num_steps",
            11: "voltage",
        }
        field = field_by_col.get(col)
        if not field:
            # Columns not tracked; fallback to existing model data
            return self.model().data(self.model().index(row, col))
        value = self.loaded_stimulus[row].get(field)
        # Normalize for display
        return self._format_display_value(field, value) if field != "stimulus_name" else value

    def ok_btn_clicked(self):
        if self.select_stimulus_row is None:
            MessageBox.warning(self, "提示", "请选择要加载的激励信号配置")
            return
        self.is_clicked_ok = True
        self.close()

    def del_config(self):
        if self.select_stimulus_row is not None:
            stimulus_name = self.loaded_stimulus[self.select_stimulus_row].get("stimulus_name")
            code, msg = StimulusSignalManagement().delete_stimulus_info_from_db(stimulus_name)

            # Check if deletion failed
            if code != error_code.OK:
                # Check for foreign key constraint (case-insensitive, supports Chinese and English)
                if "FOREIGN KEY" in msg.upper() or "外键" in msg:
                    MessageBox.warning(self, "提示", "请先删除该激励信号下的所有数据")
                else:
                    MessageBox.warning(self, "删除失败", msg)
                    self.logger.error("delete stimulus config %s failed: %s" % (stimulus_name, msg))
                return

            # Only update UI on successful deletion
            self.logger.info("delete stimulus config %s success" % stimulus_name)
            self.loaded_stimulus.pop(self.select_stimulus_row)
            self.model().removeRow(self.select_stimulus_row)
            self.select_stimulus_row = None

    def exec(self):
        super().exec_()
        if self.is_clicked_ok:
            return self.get_selected_stimulus_config()
        else:
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LoadStimulusDialog(LogManager.set_log_handler("core"))
    a = dialog.exec()
    print(a)
    # sys.exit(app.exec_())
