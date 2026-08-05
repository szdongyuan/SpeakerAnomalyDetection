import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from base.excel_export_selection import (
    available_excel_outputs,
    normalize_save_item_outputs,
    serialize_save_item_outputs,
)
from consts.excel_export_consts import (
    EXCEL_OUTPUT_DEVIATION,
    EXCEL_OUTPUT_MARGIN,
    EXCEL_OUTPUT_ORDER,
    EXCEL_OUTPUT_TEST_CURVE,
    SAVE_ITEM_OUTPUTS_KEY,
)
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    GroupBox,
    Label,
    LineEdit,
    MessageBox,
    PushButton,
    SpinBox,
    TreeWidget,
)
from ui.ui_analysis_config.common_widgets import SemanticAnalysisConfigDialogBase


class ExcelConfigWindow(SemanticAnalysisConfigDialogBase):
    """
    Excel 结果导出配置（全局分析项）

    - 可感知当前序列中已有的分析项（排除自身与频谱 Spec）
    - 用户选择需要写入 Excel 的分析项
    - 支持配置保存目录、文件名、是否按日期拆分、单条曲线最大点数（避免 Excel 列数上限）
    """

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self._output_root_by_name: dict[str, QTreeWidgetItem] = {}
        self._output_child_by_name: dict[str, dict[str, QTreeWidgetItem]] = {}
        self._syncing_output_tree = False

        self.init_ui()

    def init_ui(self):
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        output_widget = QWidget(self)
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(12)

        output_layout.addWidget(self._create_excel_save_group())
        output_layout.addWidget(self._create_mes_save_group())
        self.add_semantic_section("output", widget=output_widget)

    def _create_excel_save_group(self):
        basic_box = GroupBox("Excel 保存设置")
        basic_layout = QVBoxLayout()

        # Save directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(Label("保存目录:"))
        self.save_dir_edit = LineEdit()
        self.save_dir_edit.setText(self.load_config.get("save_dir") or "")
        self.save_dir_edit.editingFinished.connect(self.on_save_dir_editing_finished)
        browse_btn = PushButton("浏览…")
        browse_btn.clicked.connect(self.on_browse_dir)
        dir_layout.addWidget(self.save_dir_edit)
        dir_layout.addWidget(browse_btn)
        basic_layout.addLayout(dir_layout)

        # Optional: add product model as a subdirectory under save_dir
        self.add_model_dir_chk = CheckBox("根据型号分类")
        self.add_model_dir_chk.setChecked(bool(self.load_config.get("add_model_dir", False)))
        basic_layout.addWidget(self.add_model_dir_chk)

        # File base name
        name_layout = QHBoxLayout()
        name_layout.addWidget(Label("文件名:"))
        self.file_base_edit = LineEdit()
        self.file_base_edit.setText(self.load_config.get("file_base") or "analysis_results")
        name_layout.addWidget(self.file_base_edit)
        basic_layout.addLayout(name_layout)

        # Append date
        self.append_date_chk = CheckBox("文件名添加日期(每日新建)")
        self.append_date_chk.setChecked(bool(self.load_config.get("add_date", True)))
        basic_layout.addWidget(self.append_date_chk)

        # Runtime file lock (CSV spool + daily XLSX)
        self.lock_files_chk = CheckBox("运行中锁定CSV/Excel文件")
        self.lock_files_chk.setChecked(bool(self.load_config.get("lock_files", True)))
        basic_layout.addWidget(self.lock_files_chk)

        # Max points
        max_points_layout = QHBoxLayout()
        max_points_layout.addWidget(Label("最大保存点数:"))
        self.max_points_spin = SpinBox()
        self.max_points_spin.setRange(10, 16382)  # Excel max columns 16384, reserve 2 cols for SN/日期
        self.max_points_spin.setValue(int(self.load_config.get("max_points", 2000) or 2000))
        max_points_layout.addWidget(self.max_points_spin)
        basic_layout.addLayout(max_points_layout)

        # Item selection
        select_box = GroupBox("选择需要保存的分析项")
        select_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        select_all_btn = PushButton("全选")
        clear_all_btn = PushButton("全不选")
        select_all_btn.clicked.connect(self.on_select_all)
        clear_all_btn.clicked.connect(self.on_clear_all)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_all_btn)
        btn_row.addStretch()
        select_layout.addLayout(btn_row)

        available_items = self._get_available_analysis_items()
        analysis_config = self.config_manager.load_config() or {}
        selected_outputs = normalize_save_item_outputs(
            self.load_config,
            analysis_config,
            available_items=available_items,
        )

        self.output_tree = TreeWidget(self)
        self.output_tree.setColumnCount(1)
        self.output_tree.setHeaderHidden(True)
        self.output_tree.setSelectionMode(TreeWidget.NoSelection)

        output_labels = {
            EXCEL_OUTPUT_TEST_CURVE: "测试曲线",
            EXCEL_OUTPUT_MARGIN: "Margin",
            EXCEL_OUTPUT_DEVIATION: "偏差曲线",
        }
        for name in available_items:
            item_config = analysis_config.get(name, {})
            available_outputs = set(available_excel_outputs(item_config))
            supported_outputs = [
                output
                for output in EXCEL_OUTPUT_ORDER
                if output in available_outputs
                and not (item_config.get("type") == "RSC" and output == EXCEL_OUTPUT_TEST_CURVE)
            ]
            if not supported_outputs:
                continue

            root = QTreeWidgetItem([name])
            root.setFlags(root.flags() | Qt.ItemIsUserCheckable)
            self.output_tree.addTopLevelItem(root)
            children: dict[str, QTreeWidgetItem] = {}
            for output in supported_outputs:
                child = QTreeWidgetItem(root, [output_labels[output]])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(
                    0,
                    Qt.Checked if output in selected_outputs.get(name, ()) else Qt.Unchecked,
                )
                children[output] = child

            self._output_root_by_name[name] = root
            self._output_child_by_name[name] = children
            self._refresh_output_root_state(root)
            root.setExpanded(False)

        self.output_tree.itemChanged.connect(self._on_output_tree_item_changed)
        select_layout.addWidget(self.output_tree)
        select_box.setLayout(select_layout)
        basic_layout.addWidget(select_box)

        basic_box.setLayout(basic_layout)
        return basic_box

    def _create_mes_save_group(self):
        mes_box = GroupBox("MES 保存设置")
        mes_layout = QVBoxLayout()

        self.mes_chk = CheckBox("保存MES结果")
        self.mes_chk.setChecked(bool(self.load_config.get("save_mes_enabled", False)))
        mes_layout.addWidget(self.mes_chk)

        mes_dir_layout = QHBoxLayout()
        mes_dir_layout.addWidget(Label("MES目录:"))
        self.mes_file_base_edit = LineEdit()
        if "mes_file_base" in self.load_config:
            self.mes_file_base_edit.setText(self.load_config.get("mes_file_base") or "")
        else:
            self.mes_file_base_edit.setText("D:/dataMES")
        mes_browse_btn = PushButton("浏览…")
        mes_browse_btn.clicked.connect(self.on_browse_mes_dir)
        mes_dir_layout.addWidget(self.mes_file_base_edit)
        mes_dir_layout.addWidget(mes_browse_btn)
        mes_layout.addLayout(mes_dir_layout)

        mes_name_layout = QHBoxLayout()
        mes_name_layout.addWidget(Label("MES文件名:"))
        self.mes_file_name_edit = LineEdit()
        if "mes_file_name" in self.load_config:
            self.mes_file_name_edit.setText(self.load_config.get("mes_file_name") or "")
        else:
            self.mes_file_name_edit.setText("Results")
        mes_name_layout.addWidget(self.mes_file_name_edit)
        mes_layout.addLayout(mes_name_layout)

        mes_box.setLayout(mes_layout)
        return mes_box

    def _get_available_analysis_items(self) -> list[str]:
        """
        Read from current sequence's analysis_list (via ConfigManager.config) to build dynamic options.
        """
        cfg = self.config_manager.load_config() or {}
        result: list[str] = []
        for k, v in cfg.items():
            if k == self.model_type:
                continue
            if not isinstance(v, dict):
                continue
            t = v.get("type")
            if t in ("Excel", "Spec"):
                continue
            result.append(k)
        result.sort()
        return result

    def on_browse_dir(self):
        init_dir = self.save_dir_edit.text().strip()
        if not init_dir:
            init_dir = os.path.join(DEFAULT_DIR, "audio_data")
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", init_dir)
        if path:
            self.save_dir_edit.setText(path)
            self.on_save_dir_editing_finished()

    def on_browse_mes_dir(self):
        init_dir = self.mes_file_base_edit.text().strip() or "D:/dataMES"
        path = QFileDialog.getExistingDirectory(self, "选择MES保存目录", init_dir)
        if path:
            self.mes_file_base_edit.setText(path)

    def _validate_save_dir_text(self, text: str, *, create: bool) -> tuple[bool, str]:
        raw = str(text or "").strip()
        if not raw:
            return True, ""

        # Expand variables and user home when possible.
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            path = raw

        # Fast reachability check for drive/UNC root.
        try:
            drive, _tail = os.path.splitdrive(path)
            if drive:
                root = drive + os.sep
                if not os.path.exists(root):
                    return False, f"保存目录不可达：{root} 不存在或未映射"
        except Exception:
            pass

        if create:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                return False, f"保存目录不可达或无权限：\n{path}\n{e}"
            try:
                if not os.path.isdir(path):
                    return False, f"保存目录不是有效文件夹：\n{path}"
            except Exception:
                return False, f"保存目录不可达：\n{path}"

        return True, ""

    def _validate_mes_file_name_text(self, text: str) -> tuple[bool, str]:
        raw = "" if text is None else str(text)
        trimmed = raw.strip()
        if not trimmed:
            return False, "MES 文件名不能为空"
        if raw[-1] in {" ", "."}:
            return False, "MES 文件名不能以空格或点结尾"
        if any(ord(ch) < 32 for ch in raw):
            return False, "MES 文件名不能包含控制字符"
        drive, _tail = os.path.splitdrive(raw)
        if drive:
            return False, "MES 文件名不能包含盘符或绝对路径"
        if trimmed in {".", ".."}:
            return False, "MES 文件名不能包含路径"
        if any(ch in raw for ch in '<>:"/\\|?*'):
            return False, "MES 文件名不能包含非法字符"
        normalized_basename = trimmed.split(".", 1)[0].rstrip(" .").upper()
        if os.path.basename(raw) != raw:
            return False, "MES 文件名不能包含路径"
        if any(sep in raw for sep in ("/", "\\")):
            return False, "MES 文件名不能包含路径"
        return True, ""

    def _validate_mes_config(self) -> tuple[bool, str]:
        if not self.mes_chk.isChecked():
            return True, ""

        mes_file_base = self.mes_file_base_edit.text()
        mes_file_name = self.mes_file_name_edit.text()
        if not mes_file_base:
            return False, "MES 保存目录不能为空"
        ok, msg = self._validate_save_dir_text(mes_file_base, create=True)
        if not ok:
            return False, msg
        return self._validate_mes_file_name_text(mes_file_name)

    def on_save_dir_editing_finished(self):
        ok, msg = self._validate_save_dir_text(self.save_dir_edit.text(), create=False)
        if ok:
            return
        MessageBox.warning(self, "保存目录不可用", msg)

    def on_select_all(self):
        self._set_all_output_children(Qt.Checked)

    def on_clear_all(self):
        self._set_all_output_children(Qt.Unchecked)

    def _root_state_from_children(self, root: QTreeWidgetItem) -> Qt.CheckState:
        states = [root.child(index).checkState(0) for index in range(root.childCount())]
        if all(state == Qt.Checked for state in states):
            return Qt.Checked
        if any(state != Qt.Unchecked for state in states):
            return Qt.PartiallyChecked
        return Qt.Unchecked

    def _refresh_output_root_state(self, root: QTreeWidgetItem) -> None:
        root.setCheckState(0, self._root_state_from_children(root))

    def _on_output_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing_output_tree or column != 0:
            return
        self._syncing_output_tree = True
        try:
            parent = item.parent()
            if parent is None:
                target = Qt.Unchecked if item.checkState(0) == Qt.Unchecked else Qt.Checked
                for index in range(item.childCount()):
                    item.child(index).setCheckState(0, target)
                item.setCheckState(0, target)
            else:
                self._refresh_output_root_state(parent)
        finally:
            self._syncing_output_tree = False

    def _set_all_output_children(self, state: Qt.CheckState) -> None:
        self._syncing_output_tree = True
        try:
            for name, children in self._output_child_by_name.items():
                for child in children.values():
                    child.setCheckState(0, state)
                self._output_root_by_name[name].setCheckState(0, state)
        finally:
            self._syncing_output_tree = False

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton("设为默认")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton("确定")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        save_dir = self.save_dir_edit.text().strip() or None
        file_base = self.file_base_edit.text().strip() or "analysis_results"
        add_date = self.append_date_chk.isChecked()
        lock_files = self.lock_files_chk.isChecked()
        add_model_dir = self.add_model_dir_chk.isChecked()
        max_points = int(self.max_points_spin.value())
        save_item_outputs = serialize_save_item_outputs(
            {
                name: [
                    output
                    for output in EXCEL_OUTPUT_ORDER
                    if output in children and children[output].checkState(0) == Qt.Checked
                ]
                for name, children in self._output_child_by_name.items()
            }
        )
        save_items = sorted(save_item_outputs)
        return {
            "enabled": True,
            "save_dir": save_dir,
            "file_base": file_base,
            "add_date": add_date,
            "add_model_dir": add_model_dir,
            "lock_files": lock_files,
            "date_format": "%Y%m%d",
            "max_points": max_points,
            "save_items": save_items,
            SAVE_ITEM_OUTPUTS_KEY: save_item_outputs,
            "save_mes_enabled": self.mes_chk.isChecked(),
            "mes_file_base": self.mes_file_base_edit.text() or None,
            "mes_file_name": self.mes_file_name_edit.text(),
        }

    def on_default_btn_clicked(self):
        ok, msg = self._validate_save_dir_text(self.save_dir_edit.text(), create=True)
        if not ok:
            MessageBox.warning(self, "保存目录不可用", msg)
            return
        ok, msg = self._validate_mes_config()
        if not ok:
            MessageBox.warning(self, "MES配置不可用", msg)
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("Excel", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.model_type, {})
        self._output_root_by_name = {}
        self._output_child_by_name = {}
        self._syncing_output_tree = False
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        ok, msg = self._validate_save_dir_text(self.save_dir_edit.text(), create=True)
        if not ok:
            MessageBox.warning(self, "保存目录不可用", msg)
            return None
        ok, msg = self._validate_mes_config()
        if not ok:
            MessageBox.warning(self, "MES配置不可用", msg)
            return None
        config_data = self.get_default_config()
        self.accept()
        return config_data
