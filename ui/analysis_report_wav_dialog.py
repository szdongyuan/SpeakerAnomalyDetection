"""WAV selection dialog used by the analysis report export flow."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from consts import ui_style_const
from ui.dialog_enter_policy import install_dialog_enter_policy


_NATURAL_PART_RE = re.compile(r"(\d+)")


def _natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART_RE.split(str(value or ""))
    )


class SelectAllCheckBox(QCheckBox):
    """Show partial state while toggling only between all and none."""

    def nextCheckState(self):
        next_state = Qt.Unchecked if self.checkState() == Qt.Checked else Qt.Checked
        self.setCheckState(next_state)


class FilterMultiSelect(QToolButton):
    """Checkable filter values with a tri-state select-all row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.setPopupMode(QToolButton.InstantPopup)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        menu = QMenu(self)
        self.setMenu(menu)
        panel = QWidget(menu)
        panel.setMinimumWidth(260)

        self.select_all_checkbox = SelectAllCheckBox("全选", panel)
        self.select_all_checkbox.setTristate(True)
        self.option_list = QListWidget(panel)
        self.option_list.setMinimumHeight(160)
        self.option_list.setMaximumHeight(280)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.select_all_checkbox)
        layout.addWidget(self.option_list)

        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)

        self.select_all_checkbox.stateChanged.connect(
            self._select_all_changed
        )
        self.option_list.itemChanged.connect(self._item_changed)
        self.set_values(())

    def set_values(self, values, *, selected_values=None):
        normalized = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in values
                    if str(value).strip()
                },
                key=_natural_key,
            )
        )
        selected = (
            set(normalized)
            if selected_values is None
            else {str(value).strip() for value in selected_values}
        )
        self._updating = True
        self.option_list.blockSignals(True)
        try:
            self.option_list.clear()
            for value in normalized:
                item = QListWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked if value in selected else Qt.Unchecked
                )
                self.option_list.addItem(item)
        finally:
            self.option_list.blockSignals(False)
            self._updating = False
        self._sync_state()

    def values(self):
        return tuple(
            self.option_list.item(row).text()
            for row in range(self.option_list.count())
        )

    def selected_values(self):
        return tuple(
            self.option_list.item(row).text()
            for row in range(self.option_list.count())
            if self.option_list.item(row).checkState() == Qt.Checked
        )

    def set_selected_values(self, values):
        selected = {str(value).strip() for value in values}
        self._updating = True
        self.option_list.blockSignals(True)
        try:
            for row in range(self.option_list.count()):
                item = self.option_list.item(row)
                item.setCheckState(
                    Qt.Checked if item.text() in selected else Qt.Unchecked
                )
        finally:
            self.option_list.blockSignals(False)
            self._updating = False
        self._sync_state()

    def set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        self._updating = True
        self.option_list.blockSignals(True)
        try:
            for row in range(self.option_list.count()):
                self.option_list.item(row).setCheckState(state)
        finally:
            self.option_list.blockSignals(False)
            self._updating = False
        self._sync_state()

    def _select_all_changed(self, state):
        if not self._updating:
            self.set_all_checked(state == Qt.Checked)

    def _item_changed(self, *_args):
        if not self._updating:
            self._sync_state()

    def _sync_state(self):
        total = self.option_list.count()
        selected = len(self.selected_values())
        if selected == 0:
            state = Qt.Unchecked
        elif selected == total:
            state = Qt.Checked
        else:
            state = Qt.PartiallyChecked

        self._updating = True
        self.select_all_checkbox.blockSignals(True)
        try:
            self.select_all_checkbox.setCheckState(state)
            self.select_all_checkbox.setEnabled(total > 0)
        finally:
            self.select_all_checkbox.blockSignals(False)
            self._updating = False

        if total == 0:
            text = "无可选项"
        elif selected == 0:
            text = "未选择"
        elif selected == total:
            text = "ALL"
        else:
            text = f"已选 {selected} / {total}"
        self.setText(text)
        self.setEnabled(total > 0)


class CandidateTableModel(QAbstractTableModel):
    """Checkable WAV rows for the final report scope."""

    selection_changed = pyqtSignal()

    SELECT_COLUMN = 0
    SAMPLE_COLUMN = 1
    PORT_COLUMN = 2
    CONDITION_COLUMN = 3
    ROUND_COLUMN = 4
    RESULT_COLUMN = 5
    RECORDED_AT_COLUMN = 6
    HEADERS = (
        "选择",
        "样本编号",
        "端口",
        "档位",
        "轮次",
        "总体判定",
        "录制时间",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidates = []
        self._checked_paths = set()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._candidates)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        candidate = self._candidates[index.row()]
        column = index.column()
        if role == Qt.CheckStateRole and column == self.SELECT_COLUMN:
            return (
                Qt.Checked
                if candidate.wav_path in self._checked_paths
                else Qt.Unchecked
            )
        if role == Qt.DisplayRole:
            values = (
                "",
                candidate.sample,
                candidate.port or "—",
                candidate.condition or "—",
                candidate.round_text,
                candidate.result_text,
                candidate.recorded_at_text,
            )
            return values[column]
        if role == Qt.ForegroundRole and column == self.RESULT_COLUMN:
            result_colors = {
                "OK": ui_style_const.COLOR_OK,
                "NG": ui_style_const.COLOR_NG,
            }
            return QColor(
                result_colors.get(
                    candidate.result_text,
                    ui_style_const.COLOR_TEXT_MUTED,
                )
            )
        if (
            role == Qt.FontRole
            and column == self.RESULT_COLUMN
            and candidate.result_text == "未产生判定"
        ):
            font = QFont(QApplication.font())
            font.setPointSizeF(font.pointSizeF() - 1.0)
            return font
        if role == Qt.ToolTipRole:
            details = [Path(candidate.wav_path).name, candidate.wav_path]
            details.extend(candidate.issues)
            return "\n".join(dict.fromkeys(details))
        if role == Qt.TextAlignmentRole and column in {
            self.SELECT_COLUMN,
            self.ROUND_COLUMN,
            self.RESULT_COLUMN,
            self.RECORDED_AT_COLUMN,
        }:
            return int(Qt.AlignCenter)
        return None

    def flags(self, index):
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.isValid() and index.column() == self.SELECT_COLUMN:
            flags |= Qt.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.EditRole):
        if (
            not index.isValid()
            or index.column() != self.SELECT_COLUMN
            or role != Qt.CheckStateRole
        ):
            return False
        path = self._candidates[index.row()].wav_path
        if value == Qt.Checked:
            self._checked_paths.add(path)
        else:
            self._checked_paths.discard(path)
        self.dataChanged.emit(index, index, [Qt.CheckStateRole])
        self.selection_changed.emit()
        return True

    def reset_index(self, candidates, *, select_all=True):
        self.beginResetModel()
        self._candidates = list(candidates)
        self._checked_paths = (
            {candidate.wav_path for candidate in self._candidates}
            if select_all
            else set()
        )
        self.endResetModel()
        self.selection_changed.emit()

    def candidates(self):
        return tuple(self._candidates)

    def checked_paths(self):
        return set(self._checked_paths)

    def checked_candidates(self):
        return [
            candidate
            for candidate in self._candidates
            if candidate.wav_path in self._checked_paths
        ]

    def candidate_at(self, row):
        return self._candidates[row]

    def set_checked_paths(self, paths):
        available = {candidate.wav_path for candidate in self._candidates}
        checked = available.intersection(paths)
        if checked == self._checked_paths:
            return
        self._checked_paths = checked
        self._emit_selection_changed()

    def set_paths_checked(self, paths, checked):
        available = {candidate.wav_path for candidate in self._candidates}
        paths = available.intersection(paths)
        updated = set(self._checked_paths)
        if checked:
            updated.update(paths)
        else:
            updated.difference_update(paths)
        if updated == self._checked_paths:
            return
        self._checked_paths = updated
        self._emit_selection_changed()

    def check_state_for_paths(self, paths):
        available = {candidate.wav_path for candidate in self._candidates}
        paths = available.intersection(paths)
        if not paths or self._checked_paths.isdisjoint(paths):
            return Qt.Unchecked
        if paths.issubset(self._checked_paths):
            return Qt.Checked
        return Qt.PartiallyChecked

    def _emit_selection_changed(self):
        if self._candidates:
            self.dataChanged.emit(
                self.index(0, self.SELECT_COLUMN),
                self.index(len(self._candidates) - 1, self.SELECT_COLUMN),
                [Qt.CheckStateRole],
            )
        self.selection_changed.emit()


class CandidateFilterProxyModel(QSortFilterProxyModel):
    """Apply per-column value sets without changing row selection."""

    filters_changed = pyqtSignal()
    FILTERABLE_COLUMNS = (
        CandidateTableModel.SAMPLE_COLUMN,
        CandidateTableModel.PORT_COLUMN,
        CandidateTableModel.CONDITION_COLUMN,
        CandidateTableModel.ROUND_COLUMN,
        CandidateTableModel.RESULT_COLUMN,
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filters = {}

    def filterAcceptsRow(self, source_row, source_parent):
        source_model = self.sourceModel()
        if source_model is None:
            return False
        for column, expected_values in self._filters.items():
            index = source_model.index(source_row, column, source_parent)
            value = str(source_model.data(index, Qt.DisplayRole))
            if value not in expected_values:
                return False
        return True

    def filter_values(self, column):
        if column not in self.FILTERABLE_COLUMNS or self.sourceModel() is None:
            return ()
        source_model = self.sourceModel()
        values = {
            str(source_model.data(source_model.index(row, column), Qt.DisplayRole))
            for row in range(source_model.rowCount())
        }
        return tuple(sorted(values, key=_natural_key))

    def selected_filter_values(self, column):
        values = self._filters.get(column)
        if values is None:
            return None
        return tuple(sorted(values, key=_natural_key))

    def active_filters(self):
        return dict(self._filters)

    def active_filter_count(self):
        return len(self._filters)

    def set_filter_value(self, column, value):
        self.set_filter_values(
            column,
            None if value in {None, ""} else (value,),
        )

    def set_filter_values(self, column, values):
        if column not in self.FILTERABLE_COLUMNS:
            return
        filters = self.active_filters()
        if values is None:
            filters.pop(column, None)
        else:
            filters[column] = values
        self.set_filters(filters)

    def set_filters(self, filters):
        normalized = {}
        for column, values in filters.items():
            if column not in self.FILTERABLE_COLUMNS or values is None:
                continue
            if isinstance(values, str):
                values = (values,)
            selected = frozenset(
                str(value).strip()
                for value in values
                if str(value).strip()
            )
            available = frozenset(self.filter_values(column))
            if selected != available:
                normalized[column] = selected
        if normalized == self._filters:
            return
        self._filters = normalized
        self.invalidateFilter()
        self.filters_changed.emit()

    def clear_filters(self):
        self.set_filters({})

    def has_active_filters(self):
        return bool(self._filters)


class CandidateFilterDialog(QDialog):
    """Collect WAV filters in a compact modal dialog."""

    FILTER_FIELDS = (
        ("样本编号", CandidateTableModel.SAMPLE_COLUMN),
        ("端口", CandidateTableModel.PORT_COLUMN),
        ("档位", CandidateTableModel.CONDITION_COLUMN),
        ("轮次", CandidateTableModel.ROUND_COLUMN),
        ("总体判定", CandidateTableModel.RESULT_COLUMN),
    )

    def __init__(self, filter_model, parent=None):
        super().__init__(parent)
        self.filter_model = filter_model
        self.selector_by_column = {}
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("analysisReportWavFilterDialog")
        self.setWindowTitle("WAV 筛选")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.setStyleSheet(ui_style_const.analysis_report_dialog_style)
        self.resize(500, 360)
        self.setMinimumSize(480, 330)

        fields_layout = QGridLayout()
        fields_layout.setHorizontalSpacing(12)
        fields_layout.setVerticalSpacing(14)
        for row, (label_text, column) in enumerate(self.FILTER_FIELDS):
            label = QLabel(f"{label_text}：", self)
            selector = FilterMultiSelect(self)
            selector.setFixedWidth(220)
            fields_layout.addWidget(label, row, 0)
            fields_layout.addWidget(selector, row, 1, alignment=Qt.AlignLeft)
            self.selector_by_column[column] = selector
        fields_layout.setColumnStretch(1, 1)

        self.cancel_button = QPushButton("取消", self)
        self.apply_button = QPushButton("应用", self)
        self.apply_button.setObjectName("reportWavFilterApplyButton")
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(16)
        root.addLayout(fields_layout)
        root.addStretch(1)
        root.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self._apply_filters)
        install_dialog_enter_policy(self, self.apply_button)

    def reload(self):
        for _label, column in self.FILTER_FIELDS:
            selector = self.selector_by_column[column]
            available = self.filter_model.filter_values(column)
            selected = self.filter_model.selected_filter_values(column)
            selector.set_values(
                available,
                selected_values=available if selected is None else selected,
            )

    def _apply_filters(self):
        self.filter_model.set_filters(
            {
                column: selector.selected_values()
                for column, selector in self.selector_by_column.items()
            }
        )
        self.accept()

class AnalysisReportWavDialog(QDialog):
    """Adjust exact WAV rows without expanding the main export dialog."""

    def __init__(self, candidate_model, parent=None):
        super().__init__(parent)
        self.candidate_model = candidate_model
        self._initial_checked_paths = candidate_model.checked_paths()
        self._committed = False
        self._build_ui()
        self._sync_summary()

    def _build_ui(self):
        self.setObjectName("analysisReportWavDialog")
        self.setWindowTitle("WAV 明细")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(ui_style_const.analysis_report_dialog_style)
        self.resize(840, 460)
        self.setMinimumSize(720, 360)

        self.select_all_checkbox = SelectAllCheckBox("全选", self)
        self.select_all_checkbox.setObjectName("reportWavSelectAll")
        self.select_all_checkbox.setTristate(True)
        self.summary_label = QLabel(self)
        self.summary_label.setObjectName("reportWavSelectionSummary")
        self.filter_button = QPushButton("筛选", self)
        self.filter_button.setObjectName("reportWavFilterButton")

        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.select_all_checkbox)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.filter_button)

        self.candidate_table = QTableView(self)
        self.candidate_table.setObjectName("reportWavTable")
        self.filter_model = CandidateFilterProxyModel(self)
        self.filter_model.setSourceModel(self.candidate_model)
        self.filter_dialog = CandidateFilterDialog(self.filter_model, self)
        self.candidate_table.setModel(self.filter_model)
        self.candidate_table.setAlternatingRowColors(False)
        self.candidate_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.candidate_table.setEditTriggers(QTableView.AllEditTriggers)
        self.candidate_table.verticalHeader().hide()
        header = self.candidate_table.horizontalHeader()
        header.setSectionsClickable(False)
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(
            CandidateTableModel.SELECT_COLUMN,
            QHeaderView.Fixed,
        )
        header.resizeSection(CandidateTableModel.SELECT_COLUMN, 54)
        header.setSectionResizeMode(
            CandidateTableModel.RECORDED_AT_COLUMN,
            QHeaderView.Fixed,
        )
        header.resizeSection(CandidateTableModel.RECORDED_AT_COLUMN, 190)

        self.cancel_button = QPushButton("取消", self)
        self.confirm_button = QPushButton("确定", self)
        self.confirm_button.setObjectName("reportWavConfirmButton")
        self.confirm_button.setDefault(True)
        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.summary_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.confirm_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        root.addLayout(toolbar_layout)
        root.addWidget(self.candidate_table, 1)
        root.addLayout(footer_layout)

        self.select_all_checkbox.stateChanged.connect(self._select_all_changed)
        self.candidate_model.selection_changed.connect(self._sync_summary)
        self.filter_model.filters_changed.connect(self._filters_changed)
        self.filter_button.clicked.connect(self._show_filter_dialog)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self.accept)
        install_dialog_enter_policy(self, self.confirm_button)

    def _select_all_changed(self, state):
        self.candidate_model.set_paths_checked(
            self._visible_paths(),
            state == Qt.Checked,
        )

    def _sync_summary(self):
        total = self.candidate_model.rowCount()
        visible = self.filter_model.rowCount()
        selected = len(self.candidate_model.checked_candidates())
        sample_count = len(
            {
                candidate.sample
                for candidate in self.candidate_model.candidates()
                if candidate.sample
            }
        )
        visible_text = f"当前显示 {visible} 个，" if self.filter_model.has_active_filters() else ""
        self.summary_label.setText(
            f"已选择 {selected} / 共 {total} 个 WAV（{visible_text}"
            f"来自 {sample_count} 个样本）"
        )
        self.select_all_checkbox.blockSignals(True)
        try:
            self.select_all_checkbox.setCheckState(
                self.candidate_model.check_state_for_paths(self._visible_paths())
            )
            self.select_all_checkbox.setEnabled(bool(visible))
        finally:
            self.select_all_checkbox.blockSignals(False)

    def _filters_changed(self):
        active = self.filter_model.has_active_filters()
        self.select_all_checkbox.setText(
            "全选当前筛选结果" if active else "全选"
        )
        filter_count = self.filter_model.active_filter_count()
        self.filter_button.setText(
            f"筛选（{filter_count}）" if filter_count else "筛选"
        )
        self.filter_button.setProperty("filtersActive", active)
        self.filter_button.style().unpolish(self.filter_button)
        self.filter_button.style().polish(self.filter_button)
        self._sync_summary()

    def _visible_paths(self):
        paths = []
        for proxy_row in range(self.filter_model.rowCount()):
            source_index = self.filter_model.mapToSource(
                self.filter_model.index(
                    proxy_row,
                    CandidateTableModel.SELECT_COLUMN,
                )
            )
            paths.append(
                self.candidate_model.candidate_at(source_index.row()).wav_path
            )
        return paths

    def _show_filter_dialog(self):
        self.filter_dialog.reload()
        self.filter_dialog.exec()

    def accept(self):
        self._committed = True
        super().accept()

    def reject(self):
        if not self._committed:
            self.candidate_model.set_checked_paths(self._initial_checked_paths)
        super().reject()
