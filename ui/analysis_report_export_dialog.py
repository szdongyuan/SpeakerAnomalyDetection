"""Manual project and exact-WAV selection UI for one analysis-report PDF."""

from __future__ import annotations

import os
import re
from typing import Iterable

from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QRegion
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from base.analysis_report import (
    export_analysis_report_pdf,
    prepare_analysis_report_runtime,
)
from base.analysis_report_source import (
    ProjectReportIndex,
    catalog_analysis_items,
    default_report_path,
    filter_candidates,
    scan_project,
)
from consts import model_consts, ui_style_const
from ui.analysis_report_wav_dialog import (
    AnalysisReportWavDialog,
    CandidateTableModel,
    SelectAllCheckBox,
)


_NATURAL_PART_RE = re.compile(r"(\d+)")


def _natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART_RE.split(str(value or ""))
    )


class SampleMultiSelect(QToolButton):
    """Scrollable sample selector with one tri-state select-all checkbox."""

    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.setObjectName("reportSampleSelector")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        menu = QMenu(self)
        self.setMenu(menu)
        panel = QWidget(menu)
        panel.setMinimumWidth(300)

        self.select_all_checkbox = SelectAllCheckBox("全选", panel)
        self.select_all_checkbox.setObjectName("reportSampleSelectAll")
        self.select_all_checkbox.setTristate(True)
        self.option_list = QListWidget(panel)
        self.option_list.setObjectName("reportSampleList")
        self.option_list.setMinimumHeight(220)
        self.option_list.setMaximumHeight(340)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.select_all_checkbox)
        layout.addWidget(self.option_list)

        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)

        self.select_all_checkbox.stateChanged.connect(self._select_all_changed)
        self.option_list.itemChanged.connect(self._item_changed)
        self.set_values(())

    def set_values(
        self,
        values: Iterable[str],
        *,
        preserve=False,
        select_all=False,
    ):
        previous = set(self.selected_values()) if preserve else set()
        normalized = sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            },
            key=_natural_key,
        )
        self._updating = True
        self.option_list.blockSignals(True)
        try:
            self.option_list.clear()
            for value in normalized:
                item = QListWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked
                    if select_all or value in previous
                    else Qt.Unchecked
                )
                self.option_list.addItem(item)
        finally:
            self.option_list.blockSignals(False)
            self._updating = False
        self._sync_select_all_state()
        self._update_text()
        self.setEnabled(bool(normalized))

    def selected_values(self):
        return [
            self.option_list.item(row).text()
            for row in range(self.option_list.count())
            if self.option_list.item(row).checkState() == Qt.Checked
        ]

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
        self._selection_updated()

    def clear_selection(self):
        self.set_all_checked(False)

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
        self._selection_updated()

    def option_count(self):
        return self.option_list.count()

    def _select_all_changed(self, state):
        if self._updating:
            return
        self.set_all_checked(state == Qt.Checked)

    def _item_changed(self, *_args):
        if not self._updating:
            self._selection_updated()

    def _selection_updated(self):
        self._sync_select_all_state()
        self._update_text()
        self.selection_changed.emit()

    def _sync_select_all_state(self):
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
            self.select_all_checkbox.setEnabled(bool(total))
        finally:
            self.select_all_checkbox.blockSignals(False)
            self._updating = False

    def _update_text(self):
        total = self.option_list.count()
        selected = len(self.selected_values())
        if total == 0:
            text = "暂无样本"
        elif selected == 0:
            text = "未选择"
        elif selected == total:
            text = f"全部已选（{total}）"
        else:
            text = f"已选 {selected} / {total}"
        self.setText(text)


class AnalysisItemCheckBoxGroup(QWidget):
    """Direct checkboxes that keep exact analysis-item identities internally."""

    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._options = []
        self._checkboxes = {}
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(18)
        self._layout.setVerticalSpacing(6)
        self._empty_label = None
        self.set_options(())

    def set_options(self, options, *, preserve=True):
        previously_selected = set(self.selected_items()) if preserve else set()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self._options = list(options)
        self._checkboxes = {}
        self._empty_label = None
        if not self._options:
            self._empty_label = QLabel("请先选择样本", self)
            self._empty_label.setObjectName("reportAnalysisItemEmptyLabel")
            self._empty_label.setAlignment(Qt.AlignCenter)
            self._empty_label.setMinimumHeight(72)
            self._layout.addWidget(self._empty_label, 0, 0)
            return

        for index, option in enumerate(self._options):
            checkbox = QCheckBox(self._display_name(option), self)
            checkbox.setObjectName("reportAnalysisItemCheckBox")
            if option.available_count < option.selected_wav_count:
                checkbox.setToolTip("部分 WAV 缺少该分析项结果。")
            checkbox.setChecked(option.identity in previously_selected)
            checkbox.stateChanged.connect(self.selection_changed.emit)
            self._checkboxes[option.identity] = checkbox
            self._layout.addWidget(checkbox, index // 3, index % 3)

    def selected_items(self):
        return [
            option.identity
            for option in self._options
            if self._checkboxes[option.identity].isChecked()
        ]

    def set_empty_text(self, text):
        if self._empty_label is not None:
            self._empty_label.setText(text)

    def selected_options(self):
        return [
            option
            for option in self._options
            if self._checkboxes[option.identity].isChecked()
        ]

    def set_checked(self, identity, checked):
        checkbox = self._checkboxes.get(identity)
        if checkbox is None:
            return False
        checkbox.setChecked(bool(checked))
        return True

    def option_count(self):
        return len(self._options)

    @staticmethod
    def _display_name(option):
        name = str(option.display_name or option.identity.key)
        analysis_type = str(option.identity.analysis_type or "")
        if analysis_type and analysis_type.casefold() not in name.casefold():
            return f"{name}（{analysis_type}）"
        return name


class _ProjectScanThread(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, project_directory, database_path, parent=None):
        super().__init__(parent)
        self.project_directory = project_directory
        self.database_path = database_path

    def run(self):
        try:
            result = scan_project(
                self.project_directory,
                database_path=self.database_path,
                cancel_requested=self.isInterruptionRequested,
            )
        except InterruptedError:
            return
        except Exception as error:
            self.failed.emit(str(error) or type(error).__name__)
        else:
            self.completed.emit(result)


class _ReportExportThread(QThread):
    completed = pyqtSignal(object)

    def __init__(self, file_path, candidates, analysis_items, report_content, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.candidates = tuple(candidates)
        self.analysis_items = tuple(analysis_items)
        self.report_content = report_content

    def run(self):
        self.completed.emit(
            export_analysis_report_pdf(
                self.file_path,
                self.candidates,
                self.analysis_items,
                report_content=self.report_content,
                cancel_requested=self.isInterruptionRequested,
            )
        )


class AnalysisReportExportDialog(QDialog):
    """Select one project, one model, exact WAVs, and exact items for one PDF."""

    def __init__(
        self,
        parent=None,
        *,
        project_directory="",
        database_path=None,
        auto_scan=True,
    ):
        super().__init__(parent)
        self._database_path = str(database_path or model_consts.DATABASE_PATH)
        self._index = None
        self._scan_thread = None
        self._export_thread = None
        self._pending_export_result = None
        self._busy = False
        self._updating_scope = False
        self._build_ui()
        self.project_path_edit.setText(str(project_directory or ""))
        self._refresh_summary()
        if project_directory and auto_scan:
            QTimer.singleShot(0, self._start_scan)

    def _build_ui(self):
        self.setObjectName("analysisReportExportDialog")
        self.setWindowTitle("报告导出")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(ui_style_const.analysis_report_dialog_style)
        self.resize(800, 500)
        self.setMinimumSize(760, 500)

        self.project_path_edit = QLineEdit(self)
        self.project_path_edit.setReadOnly(True)
        self.project_path_edit.setPlaceholderText("请选择项目目录")
        self.browse_button = QPushButton("选择项目", self)
        self.browse_button.setFixedWidth(100)
        project_label = QLabel("项目目录：", self)
        project_row = QHBoxLayout()
        project_row.setContentsMargins(0, 0, 0, 0)
        project_row.setSpacing(10)
        project_row.addWidget(project_label)
        project_row.addWidget(self.project_path_edit, 1)
        project_row.addWidget(self.browse_button)

        self.model_combo = QComboBox(self)
        self.model_combo.setObjectName("reportModelCombo")
        self.model_combo.addItem("请选择型号", "")
        self.sample_selector = SampleMultiSelect(self)
        self.candidate_model = CandidateTableModel(self)
        self.details_button = QPushButton("查看 WAV", self)
        self.details_button.setFixedWidth(100)
        self.details_button.setObjectName("reportWavDetailsButton")
        self.details_button.setToolTip("查看并调整已匹配的 WAV 明细")
        self.details_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        model_label = QLabel("型号：", self)
        sample_label = QLabel("样本编号：", self)
        scope_label_width = max(
            model_label.sizeHint().width(),
            sample_label.sizeHint().width(),
        )
        project_label.setFixedWidth(scope_label_width)
        model_label.setFixedWidth(scope_label_width)
        sample_label.setFixedWidth(scope_label_width)

        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo, 1)
        model_row.addSpacing(110)

        sample_row = QHBoxLayout()
        sample_row.setSpacing(10)
        sample_row.addWidget(sample_label)
        sample_row.addWidget(self.sample_selector, 1)
        sample_row.addWidget(self.details_button)

        scope_layout = QVBoxLayout()
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setSpacing(14)
        scope_layout.addLayout(project_row)
        scope_layout.addLayout(model_row)
        scope_layout.addLayout(sample_row)
        self.scope_group = QGroupBox("项目与导出范围", self)
        self.scope_group.setObjectName("reportScopeGroup")
        self.scope_group.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum,
        )
        self.scope_group.setLayout(scope_layout)

        self.analysis_item_panel = AnalysisItemCheckBoxGroup(self)
        self.analysis_item_panel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum,
        )
        self.analysis_item_label = QLabel("分析项：", self)
        self.analysis_item_label.setFixedWidth(scope_label_width)
        self.include_charts_checkbox = QCheckBox("包含分析图", self)
        self.include_charts_checkbox.setObjectName("reportIncludeCharts")
        self.include_charts_checkbox.setChecked(False)
        content_layout = QGridLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(10)
        content_layout.setVerticalSpacing(16)
        content_layout.addWidget(self.analysis_item_label, 0, 0, Qt.AlignTop)
        content_layout.addWidget(self.analysis_item_panel, 0, 1, Qt.AlignTop)
        content_layout.addWidget(
            self.include_charts_checkbox,
            1,
            1,
            Qt.AlignTop,
        )
        content_layout.setColumnStretch(1, 1)
        content_layout.setRowStretch(2, 1)
        self.analysis_item_group = QGroupBox("导出内容", self)
        self.analysis_item_group.setObjectName("reportContentGroup")
        self.analysis_item_group.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )
        self.analysis_item_group.setLayout(content_layout)

        self.cancel_button = QPushButton("关闭", self)
        self.export_button = QPushButton("导出 PDF…", self)
        self.export_button.setObjectName("reportExportButton")
        self.export_button.setEnabled(True)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("reportBusyProgress")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedSize(180, 12)
        # Exclude the native Windows marquee's dark bottom strip while retaining its animation.
        self.progress_bar.setMask(QRegion(self.progress_bar.rect().adjusted(0, 0, 0, -2)))
        self.progress_bar.hide()
        self.status_label = QLabel(self)
        self.status_label.hide()
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.progress_bar)
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.export_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)
        root.addWidget(self.scope_group)
        root.addWidget(self.analysis_item_group, 1)
        root.addLayout(footer)

        self.browse_button.clicked.connect(self._choose_project)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.sample_selector.selection_changed.connect(self._on_samples_changed)
        self.candidate_model.selection_changed.connect(self._candidate_selection_changed)
        self.analysis_item_panel.selection_changed.connect(self._analysis_items_changed)
        self.details_button.clicked.connect(self._open_wav_details)
        self.include_charts_checkbox.stateChanged.connect(self._refresh_summary)
        self.cancel_button.clicked.connect(self._cancel_or_close)
        self.export_button.clicked.connect(self._start_export)

    def _choose_project(self):
        initial = self.project_path_edit.text().strip() or os.getcwd()
        selected = QFileDialog.getExistingDirectory(self, "选择测试项目目录", initial)
        if not selected:
            return
        self.project_path_edit.setText(os.path.normpath(selected))
        self._start_scan()

    def _start_scan(self):
        project_directory = self.project_path_edit.text().strip()
        if not project_directory:
            QMessageBox.warning(self, "报告导出", "请先选择项目目录。")
            return
        if self._busy:
            return
        self._clear_loaded_index()
        self._scan_cancelled = False
        self.analysis_item_panel.set_empty_text("")
        self._set_busy(True, "正在扫描项目目录……")
        thread = _ProjectScanThread(project_directory, self._database_path, self)
        self._scan_thread = thread
        thread.completed.connect(self._scan_completed)
        thread.failed.connect(self._scan_failed)
        thread.finished.connect(self._scan_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _scan_completed(self, index):
        if self._scan_cancelled:
            return
        self.load_index(index)

    def _scan_failed(self, message):
        self.analysis_item_panel.set_empty_text("项目扫描失败，请重新选择项目目录。")
        QMessageBox.critical(self, "项目扫描失败", message)

    def _scan_finished(self):
        if self._scan_cancelled:
            self.analysis_item_panel.set_empty_text("已取消扫描，请重新选择项目目录。")
        self._scan_thread = None
        self._set_busy(False)

    def load_index(self, index: ProjectReportIndex):
        """Load an immutable project index; exposed for focused UI tests."""

        self._updating_scope = True
        try:
            self._index = index
            self.project_path_edit.setText(index.project_directory)
            models = sorted(
                {candidate.model for candidate in index.candidates if candidate.model},
                key=_natural_key,
            )
            self.model_combo.blockSignals(True)
            try:
                self.model_combo.clear()
                if models:
                    for model in models:
                        self.model_combo.addItem(model, model)
                    self.model_combo.setCurrentIndex(0)
                else:
                    self.model_combo.addItem("未发现型号", "")
            finally:
                self.model_combo.blockSignals(False)
        finally:
            self._updating_scope = False
        self._on_model_changed()

        if not index.candidates:
            self.analysis_item_panel.set_empty_text("该项目目录下未发现可导出的 WAV。")

    def _clear_loaded_index(self):
        self._updating_scope = True
        try:
            self._index = None
            self.model_combo.blockSignals(True)
            try:
                self.model_combo.clear()
                self.model_combo.addItem("请选择型号", "")
            finally:
                self.model_combo.blockSignals(False)
            self.sample_selector.set_values(())
            self.candidate_model.reset_index((), select_all=False)
            self.analysis_item_panel.set_options((), preserve=False)
        finally:
            self._updating_scope = False
        self._refresh_summary()

    def _on_model_changed(self, *_args):
        if self._updating_scope:
            return
        model = self._selected_model()
        samples = []
        if self._index is not None and model:
            samples = sorted(
                {
                    candidate.sample
                    for candidate in self._index.candidates
                    if candidate.model == model and candidate.sample
                },
                key=_natural_key,
            )
        self._updating_scope = True
        try:
            self.sample_selector.set_values(samples, select_all=True)
            self.candidate_model.reset_index((), select_all=False)
            self.analysis_item_panel.set_options((), preserve=False)
        finally:
            self._updating_scope = False
        self._on_samples_changed()

    def _on_samples_changed(self):
        if self._updating_scope:
            return
        model = self._selected_model()
        samples = self.sample_selector.selected_values()
        candidates = []
        if self._index is not None and model and samples:
            candidates = filter_candidates(
                self._index.candidates,
                {
                    "models": [model],
                    "sample_numbers": samples,
                },
            )
        self._updating_scope = True
        try:
            self.candidate_model.reset_index(candidates, select_all=True)
            options = catalog_analysis_items(candidates)
            self.analysis_item_panel.set_options(options, preserve=True)
        finally:
            self._updating_scope = False
        self._refresh_summary()

    def _candidate_selection_changed(self):
        if self._updating_scope:
            return
        candidates = self.candidate_model.checked_candidates()
        self._updating_scope = True
        try:
            self.analysis_item_panel.set_options(
                catalog_analysis_items(candidates),
                preserve=True,
            )
        finally:
            self._updating_scope = False
        self._refresh_summary()

    def _analysis_items_changed(self):
        if self._updating_scope:
            return
        self._refresh_summary()

    def _open_wav_details(self):
        if self._busy or self.candidate_model.rowCount() == 0:
            return
        dialog = AnalysisReportWavDialog(self.candidate_model, self)
        self.candidate_model.selection_changed.disconnect(
            self._candidate_selection_changed
        )
        try:
            dialog.exec()
        finally:
            self.candidate_model.selection_changed.connect(
                self._candidate_selection_changed
            )
            dialog.deleteLater()
        self._candidate_selection_changed()

    def _refresh_summary(self, *_args):
        self._refresh_include_charts_control()
        self._refresh_control_states()

    def _refresh_include_charts_control(self):
        self.analysis_item_label.setVisible(
            self.analysis_item_panel.option_count() > 0
        )
        has_selected_charts = any(
            option.has_charts
            for option in self.analysis_item_panel.selected_options()
        )
        if not has_selected_charts and self.include_charts_checkbox.isChecked():
            self.include_charts_checkbox.blockSignals(True)
            try:
                self.include_charts_checkbox.setChecked(False)
            finally:
                self.include_charts_checkbox.blockSignals(False)
        self.include_charts_checkbox.setVisible(has_selected_charts)

    def _refresh_control_states(self):
        has_index = self._index is not None
        has_model_options = any(
            self.model_combo.itemData(row)
            for row in range(self.model_combo.count())
        )
        has_model = bool(self._selected_model())
        has_candidates = self.candidate_model.rowCount() > 0

        self.browse_button.setEnabled(not self._busy)
        self.model_combo.setEnabled(
            not self._busy and has_index and has_model_options
        )
        self.sample_selector.setEnabled(
            not self._busy and has_model and self.sample_selector.option_count() > 0
        )
        self.analysis_item_group.setEnabled(not self._busy and has_candidates)
        self.details_button.setEnabled(not self._busy and has_candidates)
        self.include_charts_checkbox.setEnabled(not self._busy)
        self.export_button.setEnabled(not self._busy)

    def _selected_model(self):
        return str(self.model_combo.currentData() or "")

    def _report_content(self):
        return (
            "values_and_charts"
            if self.include_charts_checkbox.isChecked()
            else "values_only"
        )

    def _start_export(self):
        candidates = self.candidate_model.checked_candidates()
        items = self.analysis_item_panel.selected_items()
        if self._index is None:
            QMessageBox.warning(self, "报告导出", "请先选择项目目录。")
            return
        if not self._selected_model():
            QMessageBox.warning(self, "报告导出", "请选择型号。")
            return
        if not self.sample_selector.selected_values():
            QMessageBox.warning(self, "报告导出", "请至少选择一个样本编号。")
            return
        if not candidates:
            QMessageBox.warning(self, "报告导出", "请至少勾选一个 WAV。")
            return
        if not items:
            QMessageBox.warning(self, "报告导出", "请至少勾选一个具体分析项。")
            return

        report_content = self._report_content()
        if report_content == "values_only" and not any(
            option.has_scalar_values
            for option in self.analysis_item_panel.selected_options()
        ):
            QMessageBox.warning(
                self,
                "报告导出",
                "当前所选分析项没有可导出的标量数值，请勾选“包含分析图”或选择其他分析项。",
            )
            return

        suggested = default_report_path(self._index.project_directory)
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存分析报告",
            suggested,
            "PDF 文件 (*.pdf)",
        )
        if not file_path:
            return
        try:
            prepare_analysis_report_runtime()
        except Exception as error:
            QMessageBox.critical(self, "报告导出失败", str(error))
            return
        self._set_busy(True, "正在生成 PDF 报告……")
        thread = _ReportExportThread(
            file_path,
            candidates,
            items,
            report_content,
            self,
        )
        self._export_thread = thread
        thread.completed.connect(self._export_completed)
        thread.finished.connect(self._export_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _export_completed(self, result):
        self._pending_export_result = result

    def _export_finished(self):
        self._export_thread = None
        self._set_busy(False)
        result = self._pending_export_result
        self._pending_export_result = None
        if result is None:
            message = "PDF 导出线程未返回结果"
            QMessageBox.critical(self, "报告导出失败", message)
            return
        if result.cancelled:
            return
        if result.ok:
            QMessageBox.information(self, "报告导出", result.message)
        else:
            QMessageBox.critical(self, "报告导出失败", result.message)

    def _set_busy(self, busy, message=""):
        self._busy = bool(busy)
        self.status_label.setText(message if self._busy else "")
        self.status_label.setVisible(self._busy)
        self.progress_bar.setVisible(self._busy)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("取消任务" if self._busy else "关闭")
        self._refresh_summary()

    def _cancel_or_close(self):
        if not self._busy:
            self.reject()
            return
        thread = self._export_thread or self._scan_thread
        if thread is not None:
            if thread is self._scan_thread:
                self._scan_cancelled = True
            thread.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("正在取消……")
            self.status_label.setText("正在取消，请稍候……")

    def reject(self):
        if self._busy:
            self._cancel_or_close()
            return
        super().reject()

    def closeEvent(self, event):
        if self._busy:
            self._cancel_or_close()
            event.ignore()
            return
        super().closeEvent(event)
