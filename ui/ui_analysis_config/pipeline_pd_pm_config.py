from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from ui.ui_analysis_config.ai_config_dialog import AIConfigWindow
from ui.ui_analysis_config.fr_config_dialog import FrConfigWindow
from ui.ui_analysis_config.hd_config_dialog import HdConfigWindow
from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow
from ui.ui_analysis_config.pattern_match_config_dialog import PatternMatchConfigWindow
from ui.ui_analysis_config.pd_config_dialog import PDConfigWindow
from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.custom_ui_widget.widgets import GroupBox, Label, PushButton, CheckBox, SpinBox, MessageBox
from ui.ui_analysis_config.common_widgets import AnalysisConfigDialogBase


class PipelineConfigWindow(AnalysisConfigDialogBase):
    """
    pipeline configuration window (for inheritance)

    - select and jump to configure "前项分析" and "后项分析"
    - when saving, merge the configurations of the two analyses into the pipeline itself
    This class should be used for inheritance
    """

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        # full configuration dictionary (analysis_list)
        self.all_config = self.config_manager.load_config()
        # the saved configuration of this item
        self.load_config = self.all_config.get(model_type, {}) if isinstance(self.all_config, dict) else {}

        self.init_ui()
        self._hydrate_from_saved()

    def init_ui(self):
        self.setMinimumSize(720, 360)
        self.resize(760, 380)

        root_layout = QVBoxLayout()

        # two buttons (set by subclass)
        select_group = GroupBox("管道节点配置")
        col_btns = QVBoxLayout()
        self.btn_head_cfg = PushButton("配置前项…")
        self.btn_tail_cfg = PushButton("配置后项…")
        self.btn_head_cfg.setEnabled(False)
        self.btn_tail_cfg.setEnabled(False)
        arrow_label = Label("↓")
        arrow_label.setObjectName("arrowlabel")
        arrow_label.set_font_size(22)
        arrow_label.setAlignment(Qt.AlignCenter)
        col_btns.addStretch()
        col_btns.addWidget(self.btn_head_cfg, 0, Qt.AlignCenter)
        col_btns.addWidget(arrow_label, 0, Qt.AlignCenter)
        col_btns.addWidget(self.btn_tail_cfg, 0, Qt.AlignCenter)
        col_btns.addStretch()
        col_btns.setSpacing(8)
        select_group.setLayout(col_btns)
        # record the group reference,便于子类改标题
        self._group_box = select_group

        # bottom buttons
        btn_layout = QHBoxLayout()
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)

        root_layout.addWidget(select_group)
        root_layout.addStretch()
        root_layout.addLayout(btn_layout)

        self.setLayout(root_layout)

        # local storage (only used inside the pipeline)
        self.head_local_type = None
        self.tail_local_type = None
        self.head_local_config = {}
        self.tail_local_config = {}

        # subclass window title (can be set by subclass or overridden)
        self._head_window_title = None
        self._tail_window_title = None

        # enabled after subclass calls set_types / set_button_texts
        self.btn_head_cfg.clicked.connect(self.on_click_head_cfg)
        self.btn_tail_cfg.clicked.connect(self.on_click_tail_cfg)

    def _hydrate_from_saved(self):
        """if the item has saved head/tail configuration, initialize it to the local cache, avoid overlapping/clearing."""
        if isinstance(self.load_config, dict):
            head = self.load_config.get("head", {})
            tail = self.load_config.get("tail", {})
            if isinstance(head, dict):
                if head.get("type"):
                    self.head_local_type = head.get("type")
                    # 深拷贝以避免外部引用覆盖
                    try:
                        self.head_local_config = dict(head.get("config", {}))
                    except Exception:
                        self.head_local_config = head.get("config", {})
                    self.btn_head_cfg.setEnabled(True)
            if isinstance(tail, dict):
                if tail.get("type"):
                    self.tail_local_type = tail.get("type")
                    try:
                        self.tail_local_config = dict(tail.get("config", {}))
                    except Exception:
                        self.tail_local_config = tail.get("config", {})
                    self.btn_tail_cfg.setEnabled(True)

    def set_types(self, head_type: str, tail_type: str):
        """由子类调用，设置首/尾分析类型（如 "SPL"、"PD" 等）。"""
        self.head_local_type = head_type
        self.tail_local_type = tail_type
        self.btn_head_cfg.setEnabled(bool(head_type))
        self.btn_tail_cfg.setEnabled(bool(tail_type))

    def set_button_texts(self, head_text: str, tail_text: str):
        """由子类调用，设置按钮文案。"""
        if head_text:
            self.btn_head_cfg.setText(str(head_text))
        if tail_text:
            self.btn_tail_cfg.setText(str(tail_text))

    def set_group_title(self, title: str):
        """由子类或外部调用，设置分组标题（默认：管道节点配置）。"""
        if hasattr(self, "_group_box") and self._group_box and title:
            self._group_box.setTitle(str(title))

    def set_child_window_titles(self, head_title: str = None, tail_title: str = None):
        """由子类调用，设置打开的首/尾配置窗体标题。"""
        self._head_window_title = head_title
        self._tail_window_title = tail_title

    def _get_slot_model_name(self, slot: str) -> str:
        """生成子窗体使用的名称/标题，子类可重写以自定义。"""
        if slot == "head" and self._head_window_title:
            return str(self._head_window_title)
        if slot == "tail" and self._tail_window_title:
            return str(self._tail_window_title)
        return f"PIPE_TMP_{slot.upper()}"

    def _create_child_dialog_by_type(self, a_type: str, model_name: str) -> QDialog:
        # 这里复用各分析项配置窗口（与 OptionList.create_config_dialog 一致）
        if a_type == "SPL":
            return SplConfigWindow(self.config_manager, model_name)
        elif a_type == "FR":
            return FrConfigWindow(self.config_manager, model_name)
        elif a_type == "HD":
            return HdConfigWindow(self.config_manager, model_name)
        elif a_type == "RB":
            return RbConfigWindow(self.config_manager, model_name)
        elif a_type == "PRB":
            return PerceptualRbConfigWindow(self.config_manager, model_name)
        elif a_type == "AI":
            return AIConfigWindow(self.config_manager, model_name, 0)
        elif a_type == "Spec":
            return SpecConfigWindow(self.config_manager, model_name)
        elif a_type == "LP":
            return LPConfigWindow(self.config_manager, model_name)
        elif a_type == "PD":
            return PDConfigWindow(self.config_manager, model_name)
        elif a_type == "PM":
            return PatternMatchConfigWindow(self.config_manager, model_name)
        else:
            # 未知类型，返回空对话框
            return QDialog(self)

    def _open_and_capture_local(self, a_type: str, slot: str):
        # 使用临时名称承载配置，不污染 analysis_list
        temp_name = self._get_slot_model_name(slot)
        # 在打开子窗体前，用本地缓存预填充到 config_manager.config
        self._prefill_temp_config_to_manager(slot, temp_name)
        dialog = self._create_child_dialog_by_type(a_type, temp_name)
        dialog.setWindowTitle(temp_name)
        if dialog.exec_() == QDialog.Accepted:
            try:
                updated = dialog.on_click_ok_btn()
            except Exception:
                updated = None
            if isinstance(updated, dict):
                if slot == "head":
                    self.head_local_type = a_type
                    self.head_local_config = updated
                else:
                    self.tail_local_type = a_type
                    self.tail_local_config = updated
                # 同步更新到 config_manager.config，便于下次再次打开时保留填写
                self._write_back_temp_config(slot, temp_name, updated)

    def _open_and_update_child(self, slot: str):
        # 按类型打开，名称使用临时占位
        a_type = self.head_local_type if slot == "head" else self.tail_local_type
        if not a_type:
            MessageBox.information(self, "提示", "未设置该节点的分析类型。请在子类中调用 set_types 设置。")
            return
        self._open_and_capture_local(a_type, slot)

    def _prefill_temp_config_to_manager(self, slot: str, temp_name: str):
        """把本地缓存的 head/tail 配置写入到 config_manager.config[temp_name] 以便子窗体读取。
        不持久化到文件，仅在会话内保留，避免另一个节点被“看起来清空”。"""
        if not hasattr(self.config_manager, "config") or not isinstance(self.config_manager.config, dict):
            self.config_manager.config = {}
        local_cfg = self.head_local_config if slot == "head" else self.tail_local_config
        if isinstance(local_cfg, dict) and local_cfg:
            # Temporary child configs stay scoped to this pipeline and are not top-level analysis items.
            # 使用副本，避免子窗体原地修改带来意外引用问题
            try:
                self.config_manager.config[temp_name] = dict(local_cfg)
            except Exception:
                self.config_manager.config[temp_name] = local_cfg
        else:
            # 确保有键，哪怕是空 dict
            self.config_manager.config.setdefault(temp_name, {})

    def _write_back_temp_config(self, slot: str, temp_name: str, updated: dict):
        """子窗体关闭后，把最新配置回写到 config_manager.config[temp_name]，用于后续再次打开预填。"""
        if not hasattr(self.config_manager, "config") or not isinstance(self.config_manager.config, dict):
            self.config_manager.config = {}
        self.config_manager.config[temp_name] = dict(updated) if isinstance(updated, dict) else {}

    def on_click_head_cfg(self):
        self._open_and_update_child("head")

    def on_click_tail_cfg(self):
        self._open_and_update_child("tail")

    def get_default_config(self):
        # 如果未配置过，返回空配置但带类型（类型需由子类 set_types 提供）
        return {
            "type": "ED",
            "head": {"type": self.head_local_type, "config": self.head_local_config},
            "tail": {"type": self.tail_local_type, "config": self.tail_local_config},
        }

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data


class PipelinePdPmConfigWindow(PipelineConfigWindow):
    """PeakDetection -> PatternMatch pipeline configuration window"""

    def __init__(self, config_manager, model_type):
        super().__init__(config_manager, model_type)
        self.set_types("PD", "PM")
        self.set_button_texts("配置峰值检测参数", "配置模式匹配参数")
        self.set_child_window_titles("峰值检测参数", "模式匹配参数")
        self.set_group_title("峰值检测 -> 模式匹配")
        self._init_length_group()
        self._init_pass_condition_group()

    def _init_pass_condition_group(self):
        # pass condition: n1 ≤ matched points ≤ n2
        root_layout = self.layout()
        if not root_layout:
            return
        pass_group = GroupBox("通过条件")
        row = QHBoxLayout()
        label_prefix = Label("通过条件：")
        label_mid = Label("≤ 匹配点数 ≤")
        self._n1_spin = SpinBox()
        self._n2_spin = SpinBox()
        for sp in (self._n1_spin, self._n2_spin):
            sp.setRange(0, 1000000)
            sp.setSingleStep(1)
        # 默认值
        self._n1_spin.setValue(1)
        self._n2_spin.setValue(1)

        # 约束：n2 >= n1
        def on_n1_changed(val):
            if self._n2_spin.value() < val:
                self._n2_spin.setValue(val)

        def on_n2_changed(val):
            if val < self._n1_spin.value():
                self._n2_spin.setValue(self._n1_spin.value())

        self._n1_spin.valueChanged.connect(on_n1_changed)
        self._n2_spin.valueChanged.connect(on_n2_changed)

        row.addWidget(label_prefix)
        row.addWidget(self._n1_spin)
        row.addSpacing(6)
        row.addWidget(label_mid)
        row.addSpacing(6)
        row.addWidget(self._n2_spin)
        row.addStretch()
        pass_group.setLayout(row)

        try:
            # insert after length configuration
            root_layout.insertWidget(2, pass_group)
        except Exception:
            root_layout.addWidget(pass_group)

        try:
            if isinstance(self.load_config, dict):
                cond = self.load_config.get("pass_condition", {})
                if isinstance(cond, dict):
                    n1 = int(cond.get("n1", self._n1_spin.value()))
                    n2 = int(cond.get("n2", self._n2_spin.value()))
                    self._n1_spin.setValue(max(0, n1))
                    self._n2_spin.setValue(max(self._n1_spin.value(), n2))
        except Exception:
            pass

    def _init_length_group(self):
        root_layout = self.layout()
        if not root_layout:
            return
        length_group = GroupBox("长度控制")
        vbox = QVBoxLayout()

        # first row: left/right grid points (include peak point)
        row1 = QHBoxLayout()
        lbl_l = Label("左侧格点数")
        lbl_r = Label("右侧格点数")
        self._left_grid_spin = SpinBox()
        self._right_grid_spin = SpinBox()
        for sp in (self._left_grid_spin, self._right_grid_spin):
            sp.setRange(0, 9999999)
            sp.setSingleStep(1)
        row1.addWidget(lbl_l)
        row1.addWidget(self._left_grid_spin)
        row1.addSpacing(12)
        row1.addWidget(lbl_r)
        row1.addWidget(self._right_grid_spin)
        row1.addStretch()

        # second row: auto match template length
        row2 = QHBoxLayout()
        self._auto_equal_chk = CheckBox("自动匹配模板长度（对齐模板峰值）")
        row2.addWidget(self._auto_equal_chk)
        row2.addStretch()

        # when auto is checked, disable manual input
        def on_auto_changed(checked):
            self._left_grid_spin.setEnabled(not checked)
            self._right_grid_spin.setEnabled(not checked)

        self._auto_equal_chk.toggled.connect(on_auto_changed)

        # load existing configuration
        try:
            if isinstance(self.load_config, dict):
                auto_flag = bool(self.load_config.get("auto_equal_length", False))
                self._auto_equal_chk.setChecked(auto_flag)
                self._left_grid_spin.setEnabled(not auto_flag)
                self._right_grid_spin.setEnabled(not auto_flag)
                if not auto_flag:
                    lg = int(self.load_config.get("left_grid", 0) or 0)
                    rg = int(self.load_config.get("right_grid", 0) or 0)
                    self._left_grid_spin.setValue(max(0, lg))
                    self._right_grid_spin.setValue(max(0, rg))
        except Exception:
            pass

        vbox.addLayout(row1)
        vbox.addLayout(row2)
        length_group.setLayout(vbox)

        try:
            # insert after button group, before pass condition
            root_layout.insertWidget(1, length_group)
        except Exception:
            root_layout.addWidget(length_group)

    def get_default_config(self):
        cfg = super().get_default_config()
        # pipeline itself configuration
        cfg["auto_equal_length"] = bool(self._auto_equal_chk.isChecked()) if hasattr(self, "_auto_equal_chk") else False
        if not cfg["auto_equal_length"]:
            cfg["left_grid"] = int(self._left_grid_spin.value()) if hasattr(self, "_left_grid_spin") else 0
            cfg["right_grid"] = int(self._right_grid_spin.value()) if hasattr(self, "_right_grid_spin") else 0
        else:
            cfg.pop("left_grid", None)
            cfg.pop("right_grid", None)
        # pass condition
        cfg["pass_condition"] = {
            "n1": int(self._n1_spin.value()) if hasattr(self, "_n1_spin") else 1,
            "n2": int(self._n2_spin.value()) if hasattr(self, "_n2_spin") else 1,
        }
        return cfg

    def on_click_ok_btn(self):
        n1 = int(self._n1_spin.value()) if hasattr(self, "_n1_spin") else 1
        n2 = int(self._n2_spin.value()) if hasattr(self, "_n2_spin") else 1
        if n2 < n1:
            MessageBox.warning(self, "设置警告", "n2 应该大于等于 n1")
            return None
        config_data = self.get_default_config()
        self.accept()
        return config_data
