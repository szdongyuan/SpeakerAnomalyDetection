import json
from datetime import datetime

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QWidget,
    QStackedWidget,
    QMessageBox,
    QSizePolicy,
)

from base.save_data import ensure_test_result_file
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class SequenceCountBoard(QWidget):

    def __init__(self, analysis_config, parent=None):
        super(SequenceCountBoard, self).__init__(parent)
        self.setObjectName("sequenceCountBoard")
        self.setStyleSheet(ui_style_const.count_board_container_style)

        self.analysis_config = analysis_config
        self.mode = str()
        self._test_available = True
        self._test_unavailable_reason = ""
        self._mode_change_callbacks = []

        self.test_btn = QPushButton("测试")
        self.mark_btn = QPushButton("标记")
        self.total_line_edit = QLineEdit()
        self.ok_line_edit = QLineEdit()
        self.ng_line_edit = QLineEdit()
        self.not_labeled_line_edit = QLineEdit()
        self.yield_line_edit = QLineEdit()
        self.datatime_line_edit = QLineEdit()
        self.mark_total_edit = QLineEdit("0")
        self.mark_ok_edit = QLineEdit("0")
        self.mark_ng_edit = QLineEdit("0")
        self.mark_not_labeled_edit = QLineEdit("0")
        self.mark_yield_edit = QLineEdit("0%")
        self.mark_datatime_edit = QLineEdit()
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.reset_btn = QPushButton("重置统计")

        self.set_lineedit()
        self.set_btn()

        self.init_ui()

        self.on_mark_btn_clicked()
        # Do NOT force switch to test mode during init.

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.mode_switch_widget = self.create_mode_switch_widget()

        self.separator_line = QFrame()
        self.separator_line.setFrameShape(QFrame.HLine)
        self.separator_line.setFrameShadow(QFrame.Sunken)
        self.separator_line.setStyleSheet(ui_style_const.motor_result_divider_style)
        self.separator_line.setFixedHeight(2)

        test_widget = self.set_test_widget()
        mark_widget = self.set_mark_widget()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.stacked_widget.setFrameShape(QFrame.NoFrame)
        self.stacked_widget.setLineWidth(0)
        self.stacked_widget.setMidLineWidth(0)
        self.stacked_widget.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        self.stacked_widget.addWidget(test_widget)
        self.stacked_widget.addWidget(mark_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.mode_switch_widget)
        layout.addWidget(self.separator_line)
        layout.addWidget(self.stacked_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.setLayout(layout)

        self.set_test_text()
        self.set_mark_text()

    def create_horizontal_layout(self, label_str, item):
        label = QLabel(label_str)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(12)
        layout.addWidget(label)
        layout.addWidget(item)

        return layout

    def set_lineedit(self):
        self.total_line_edit.setAlignment(Qt.AlignCenter)
        self.ok_line_edit.setAlignment(Qt.AlignCenter)
        self.ng_line_edit.setAlignment(Qt.AlignCenter)
        self.not_labeled_line_edit.setAlignment(Qt.AlignCenter)
        self.yield_line_edit.setAlignment(Qt.AlignCenter)
        self.datatime_line_edit.setAlignment(Qt.AlignCenter)
        self.mark_total_edit.setAlignment(Qt.AlignCenter)
        self.mark_ok_edit.setAlignment(Qt.AlignCenter)
        self.mark_ng_edit.setAlignment(Qt.AlignCenter)
        self.mark_not_labeled_edit.setAlignment(Qt.AlignCenter)
        self.mark_yield_edit.setAlignment(Qt.AlignCenter)
        self.mark_datatime_edit.setAlignment(Qt.AlignCenter)

        self.total_line_edit.setDisabled(True)
        self.ok_line_edit.setDisabled(True)
        self.ng_line_edit.setDisabled(True)
        self.not_labeled_line_edit.setDisabled(True)
        self.yield_line_edit.setDisabled(True)
        self.datatime_line_edit.setDisabled(True)
        self.mark_total_edit.setDisabled(True)
        self.mark_ok_edit.setDisabled(True)
        self.mark_ng_edit.setDisabled(True)
        self.mark_not_labeled_edit.setDisabled(True)
        self.mark_yield_edit.setDisabled(True)
        self.mark_datatime_edit.setDisabled(True)

        # Keep height fixed (32 px) for visual consistency, but let width grow
        # with the parent panel so larger screens / wider splitter sections
        # don't leave a narrow 130 px column floating in white space.
        for edit in (
            self.total_line_edit,
            self.ok_line_edit,
            self.ng_line_edit,
            self.not_labeled_line_edit,
            self.yield_line_edit,
            self.datatime_line_edit,
            self.mark_total_edit,
            self.mark_ok_edit,
            self.mark_ng_edit,
            self.mark_not_labeled_edit,
            self.mark_yield_edit,
            self.mark_datatime_edit,
        ):
            edit.setMinimumSize(130, 32)
            edit.setMaximumSize(260, 32)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_btn(self):
        self.reset_btn.setStyleSheet(ui_style_const.qpushbutton_style)
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/green_circle.png"))
        self.ok_btn.setStyleSheet(ui_style_const.count_board_ok_button_style)
        self.ok_btn.setMinimumSize(148, 56)
        self.ok_btn.setMaximumSize(260, 56)
        self.ok_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ok_btn.setIconSize(QSize(20, 20))
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/red_circle.png"))
        self.ng_btn.setStyleSheet(ui_style_const.count_board_ng_button_style)
        self.ng_btn.setMinimumSize(148, 56)
        self.ng_btn.setMaximumSize(260, 56)
        self.ng_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ng_btn.setIconSize(QSize(20, 20))

    def create_mode_btn_layout(self):
        mode_label = QLabel("模式：")
        self.test_btn.setFixedSize(100, 35)
        self.mark_btn.setFixedSize(100, 35)
        self.test_btn.clicked.connect(self.on_test_btn_clicked)
        self.mark_btn.clicked.connect(self.on_mark_btn_clicked)

        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(mode_label)
        model_button_layout.addSpacing(70)
        model_button_layout.addWidget(self.test_btn)
        model_button_layout.addWidget(self.mark_btn)
        model_button_layout.setSpacing(0)

        return model_button_layout

    def create_mode_switch_widget(self):
        widget = QWidget()
        widget.setObjectName("sequenceCountBoardModeSwitch")
        widget.setLayout(self.create_mode_btn_layout())
        return widget

    def set_test_widget(self):
        total_layout = self.create_horizontal_layout("总    数：", self.total_line_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.ok_line_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.ng_line_edit)
        not_labeled_layout = self.create_horizontal_layout("未标记数：", self.not_labeled_line_edit)
        yield_layout = self.create_horizontal_layout("合 格 率：", self.yield_line_edit)
        datatime_layout = self.create_horizontal_layout("录制日期：", self.datatime_line_edit)
        test_layout = QVBoxLayout()
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.setSpacing(10)
        test_layout.addLayout(total_layout)
        test_layout.addLayout(ok_layout)
        test_layout.addLayout(ng_layout)
        test_layout.addLayout(not_labeled_layout)
        test_layout.addLayout(yield_layout)
        test_layout.addLayout(datatime_layout)

        test_widget = QWidget()
        test_widget.setObjectName("sequenceCountBoardPage")
        test_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        test_widget.setLayout(test_layout)

        return test_widget

    def set_mark_widget(self):
        total_layout = self.create_horizontal_layout("总    数：", self.mark_total_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.mark_ok_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.mark_ng_edit)
        not_labeled_layout = self.create_horizontal_layout("待标记数：", self.mark_not_labeled_edit)
        action_btn_layout = QHBoxLayout()
        action_btn_layout.setContentsMargins(0, 6, 0, 0)
        action_btn_layout.setSpacing(24)
        action_btn_layout.addStretch()
        action_btn_layout.addWidget(self.ok_btn)
        action_btn_layout.addWidget(self.ng_btn)
        action_btn_layout.addStretch()

        mark_layout = QVBoxLayout()
        mark_layout.setContentsMargins(0, 0, 0, 0)
        mark_layout.setSpacing(10)
        mark_layout.addLayout(total_layout)
        mark_layout.addLayout(ok_layout)
        mark_layout.addLayout(ng_layout)
        mark_layout.addLayout(not_labeled_layout)
        mark_layout.addLayout(action_btn_layout)

        mark_widget = QWidget()
        mark_widget.setObjectName("sequenceCountBoardPage")
        mark_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        mark_widget.setLayout(mark_layout)

        return mark_widget
    
    def on_test_btn_clicked(self):
        if not self._test_available:
            QMessageBox.information(self, "提示", self._test_unavailable_reason or "当前配置无法进入测试模式")
            self.on_mark_btn_clicked()
            return
        self.test_btn.setStyleSheet(ui_style_const.count_board_mode_active_style)
        self.mark_btn.setStyleSheet(ui_style_const.count_board_mode_inactive_style)
        self.test_btn.setEnabled(False)
        self.mark_btn.setEnabled(True)
        self.stacked_widget.setCurrentIndex(0)
        self.mode = "test"
        self.set_test_text()
        self._notify_mode_state_changed()

    def on_mark_btn_clicked(self):
        # Keep the operation summary visually stable; mark/test mode is now
        # selected from the top toolbar and mark actions live on waveform cards.
        self.stacked_widget.setCurrentIndex(0)
        self.mode = "mark"
        self.test_btn.setStyleSheet(ui_style_const.count_board_mode_inactive_style)
        self.mark_btn.setStyleSheet(ui_style_const.count_board_mode_active_style)
        self.mark_btn.setEnabled(False)
        self.test_btn.setEnabled(bool(self._test_available))
        self.set_mark_text()
        self._notify_mode_state_changed()

    def set_test_available(self, available: bool, reason: str = ""):
        """
        Control whether test mode can be entered.
        """
        self._test_available = bool(available)
        self._test_unavailable_reason = str(reason or "")
        try:
            self.test_btn.setEnabled(bool(self._test_available) and self.mode != "test")
            self.test_btn.setToolTip(self._test_unavailable_reason if not self._test_available else "")
        except Exception:
            pass
        if (not self._test_available) and self.mode == "test":
            self.on_mark_btn_clicked()
            return
        self._notify_mode_state_changed()

    def set_mode_switch_visible(self, visible: bool):
        if hasattr(self, "mode_switch_widget") and self.mode_switch_widget is not None:
            self.mode_switch_widget.setVisible(bool(visible))
        if hasattr(self, "separator_line") and self.separator_line is not None:
            self.separator_line.setVisible(bool(visible))

    def register_mode_change_callback(self, callback):
        if callable(callback):
            self._mode_change_callbacks.append(callback)

    def get_mode_state(self) -> dict:
        return {
            "mode": str(self.mode or ""),
            "test_available": bool(self._test_available),
            "test_unavailable_reason": str(self._test_unavailable_reason or ""),
        }

    def _notify_mode_state_changed(self):
        state = self.get_mode_state()
        for callback in list(self._mode_change_callbacks):
            try:
                callback(dict(state))
            except Exception:
                pass

    @staticmethod
    def _parse_test_log(lines):
        kv = {}
        for line in lines or []:
            if ":" not in str(line):
                continue
            k, v = str(line).split(":", 1)
            kv[k.strip().lower()] = v.strip()
        return kv

    def _set_visible_summary_text(self, total, ok, ng, not_labeled, yield_percent, datatime):
        self.total_line_edit.setText(str(total))
        self.ok_line_edit.setText(str(ok))
        self.ng_line_edit.setText(str(ng))
        self.not_labeled_line_edit.setText(str(not_labeled))
        self.yield_line_edit.setText(str(yield_percent))
        self.datatime_line_edit.setText(str(datatime))

    def set_test_text(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, "r") as f:
            lines = f.readlines()
            kv = self._parse_test_log(lines)
            total = kv.get("total", "0")
            ok = kv.get("ok", "0")
            ng = kv.get("ng", "0")
            not_labels = kv.get("not_labels", kv.get("not_labeled", "0"))
            ok_percent = kv.get("ok_percent", "0%")
            if ok_percent and (not ok_percent.endswith("%")):
                ok_percent = f"{ok_percent}%"
            datatime = kv.get("datatime", current_time)
            self._set_visible_summary_text(total, ok, ng, not_labels, ok_percent, datatime)

    def set_mark_text(self):
        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        with open(mark_result_path, "r") as f:
            data = json.load(f)
            total = int(data.get("total", 0) or 0)
            ok = int(data.get("ok", 0) or 0)
            ng = int(data.get("ng", 0) or 0)
            not_labels = int(data.get("not_labels", 0) or 0)
            self.mark_total_edit.setText(str(total))
            self.mark_ok_edit.setText(str(ok))
            self.mark_ng_edit.setText(str(ng))
            self.mark_not_labeled_edit.setText(str(not_labels))
            total = int(data.get("total", 0) or 0)
            ok = int(data.get("ok", 0) or 0)
            yield_percent = round(ok / total * 100, 2) if total > 0 else 0
            self.mark_yield_edit.setText(f"{yield_percent}%")
            datatime = str(data.get("datatime", datetime.now().strftime("%Y-%m-%d")))
            self.mark_datatime_edit.setText(datatime)
        self.set_test_text()

    def _read_shared_result_counts(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, "r") as f:
            kv = self._parse_test_log(f.readlines())
        try:
            total = int(kv.get("total", "0") or 0)
        except (TypeError, ValueError):
            total = 0
        try:
            ok = int(kv.get("ok", "0") or 0)
        except (TypeError, ValueError):
            ok = 0
        try:
            ng = int(kv.get("ng", "0") or 0)
        except (TypeError, ValueError):
            ng = 0
        try:
            not_labels = int(kv.get("not_labels", kv.get("not_labeled", "0")) or 0)
        except (TypeError, ValueError):
            not_labels = 0
        return max(0, total), max(0, ok), max(0, ng), max(0, not_labels)

    def _write_shared_result_counts(self, total: int, ok: int, ng: int, not_labels: int = 0):
        current_time = datetime.now().strftime("%Y-%m-%d")
        total = max(0, int(total or 0))
        ok = max(0, int(ok or 0))
        ng = max(0, int(ng or 0))
        not_labels = max(0, int(not_labels or 0))
        total = max(total, ok + ng + not_labels)
        ok_percent = round(ok / total * 100, 2) if total > 0 else 0
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, "w") as f:
            f.writelines(
                [
                    f"total: {total}\n",
                    f"ok: {ok}\n",
                    f"ng: {ng}\n",
                    f"not_labels: {not_labels}\n",
                    f"ok_percent: {ok_percent}%\n",
                    f"datatime: {current_time}\n",
                ]
            )

    def set_test_result_file(self, params):
        normalized = self._normalize_mark_label(params)
        if normalized not in ("OK", "NG", "not_labeled"):
            self.set_test_text()
            return

        total, ok, ng, not_labels = self._read_shared_result_counts()
        if normalized == "OK":
            ok += 1
        elif normalized == "NG":
            ng += 1
        elif normalized == "not_labeled":
            not_labels += 1
        total += 1
        self._write_shared_result_counts(total, ok, ng, not_labels)

    def set_mark_result_file(self, params):
        # Backward-compatible alias for historical callers.
        self.append_mark_result_file(params)

    def append_mark_result_file(self, params):
        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        datatime = datetime.now().strftime("%Y-%m-%d")
        with open(mark_result_path, "r") as f:
            data = json.load(f)

        total = int(data.get("total", 0) or 0)
        ok = int(data.get("ok", 0) or 0)
        ng = int(data.get("ng", 0) or 0)
        not_labels = int(data.get("not_labels", 0) or 0)

        normalized = self._normalize_mark_label(params)
        total += 1
        if normalized == "OK":
            ok += 1
        elif normalized == "NG":
            ng += 1
        else:
            not_labels += 1

        data["total"] = total
        data["ng"] = ng
        data["ok"] = ok
        data["not_labels"] = not_labels
        data["datatime"] = datatime
        with open(mark_result_path, "w") as f:
            json.dump(data, f, indent=4)
        if normalized in ("OK", "NG", "not_labeled"):
            self.set_test_result_file(normalized)
        else:
            self.set_test_text()

    @staticmethod
    def _normalize_mark_label(label: str) -> str:
        lowered = str(label or "").strip().lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
            return "not_labeled"
        return ""

    def update_mark_result_file_on_relabel(self, old_label: str, new_label: str):
        old_normalized = self._normalize_mark_label(old_label)
        new_normalized = self._normalize_mark_label(new_label)
        if old_normalized == new_normalized:
            return

        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        with open(mark_result_path, "r") as f:
            data = json.load(f)

        total = int(data.get("total", 0) or 0)
        ok = int(data.get("ok", 0) or 0)
        ng = int(data.get("ng", 0) or 0)
        not_labels = int(data.get("not_labels", 0) or 0)

        if old_normalized == "OK":
            ok = max(0, ok - 1)
        elif old_normalized == "NG":
            ng = max(0, ng - 1)
        elif old_normalized == "not_labeled":
            not_labels = max(0, not_labels - 1)

        if new_normalized == "OK":
            ok += 1
        elif new_normalized == "NG":
            ng += 1
        elif new_normalized == "not_labeled":
            not_labels += 1

        # Mark-mode total represents session count, so relabeling should not
        # increase/decrease total; only category buckets migrate.
        total = max(0, max(total, ok + ng + not_labels))
        data["total"] = total
        data["ok"] = ok
        data["ng"] = ng
        data["not_labels"] = not_labels
        data["datatime"] = datetime.now().strftime("%Y-%m-%d")
        with open(mark_result_path, "w") as f:
            json.dump(data, f, indent=4)
        self.update_shared_result_file_on_relabel(old_normalized, new_normalized)
        self.set_mark_text()

    def update_shared_result_file_on_relabel(self, old_label: str, new_label: str):
        old_normalized = self._normalize_mark_label(old_label)
        new_normalized = self._normalize_mark_label(new_label)
        if old_normalized == new_normalized:
            self.set_test_text()
            return

        total, ok, ng, not_labels = self._read_shared_result_counts()
        old_tracked = old_normalized in ("OK", "NG", "not_labeled")
        new_tracked = new_normalized in ("OK", "NG", "not_labeled")
        if old_normalized == "OK":
            ok = max(0, ok - 1)
        elif old_normalized == "NG":
            ng = max(0, ng - 1)
        elif old_normalized == "not_labeled":
            not_labels = max(0, not_labels - 1)

        if new_normalized == "OK":
            ok += 1
        elif new_normalized == "NG":
            ng += 1
        elif new_normalized == "not_labeled":
            not_labels += 1

        if not old_tracked and new_tracked:
            total += 1
        elif old_tracked and not new_tracked:
            total = max(0, total - 1)

        self._write_shared_result_counts(total, ok, ng, not_labels)
        self.set_test_text()

