from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QLineEdit, QFormLayout, QVBoxLayout, QDialog, QPushButton, QHBoxLayout, QComboBox

from consts import ui_style_const



class GenericFeatureParamsDialog(QDialog):
    def __init__(self, param_definitions, current_values):
        super().__init__()
        self.setWindowTitle("参数配置")
        self.param_definitions = param_definitions
        self.current_values = current_values
        self.editors = {}

        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        for name, definition in self.param_definitions.items():
            default_value = self.current_values.get(name, definition.get('default'))
            label_text = definition.get('label', name)
            param_type = definition.get('type')
            if param_type == 'dropdown':
                editor = QComboBox()
                for option in definition.get('options', []):
                    editor.addItem(option['display_name'], userData=option['value'])
                index = editor.findData(default_value)
                if index >= 0:
                    editor.setCurrentIndex(index)
            else:
                editor = QLineEdit(str(default_value))

                if 'validation' in definition:
                    validator = self.create_validator_from_def(definition['validation'])
                    if validator:
                        editor.setValidator(validator)

            form_layout.addRow(label_text, editor)
            self.editors[name] = editor

        button_layout = self.create_button()

        self.main_layout.addLayout(form_layout)
        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)
        self.setStyleSheet(
            ui_style_const.qlabel_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qdialog_style
            + ui_style_const.qtextedit_style
        )

    def create_button(self):
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确认")
        cancel_button = QPushButton("取消")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(ok_button)
        return button_layout

    @staticmethod
    def create_validator_from_def(validation_def):
        validator_type = validation_def.get("type")
        if validator_type == "int":
            min_val = validation_def.get("min", -2147483648)
            max_val = validation_def.get("max", 2147483647)
            return QIntValidator(min_val, max_val)
        elif validator_type == "float":
            min_val = validation_def.get("min", -1.0e10)
            max_val = validation_def.get("max", 1.0e10)
            decimals = validation_def.get("decimals", 5)
            return QDoubleValidator(min_val, max_val, decimals)
        return None

    def get_params(self):
        params = {}
        for name, editor in self.editors.items():
            if isinstance(editor, QComboBox):
                params[name] = editor.currentData()
            elif isinstance(editor, QLineEdit):
                param_type_str = self.param_definitions[name].get('type', 'str')
                type_map = {'int': int, 'float': float, 'str': str}
                param_type = type_map.get(param_type_str, str)
                try:
                    params[name] = param_type(editor.text())
                except Exception as e:
                    params[name] = self.param_definitions[name].get('default')

        return params