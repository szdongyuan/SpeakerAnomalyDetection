from dataclasses import replace
from unittest.mock import Mock

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLabel

from base.ai_model_catalog import AiModelCatalog, ModelCatalogError, ModelRecord
from ui.ai_model_evaluation_dialog import AiModelEvaluationDialog
from ui.ai_model_manager_dialog import AiModelDetailsDialog, AiModelManagerDialog


def model_record(**changes):
    return replace(ModelRecord(
        model_id="a", model_name="测试模型 A", model_path="models/a.keras",
        config_path="models/a.yml", input_dim="132300 x 1", output_dim=1,
        accuracy=None, update_date="2026-01-01", model_description=None,
        model_file_exists=False, config_file_exists=False,
    ), **changes)


def test_empty_registry_has_no_sample_rows_and_no_model_actions(ui_qapp, tmp_path):
    database = tmp_path / "absent.db"
    dialog = AiModelManagerDialog(AiModelCatalog(database, tmp_path))
    try:
        assert dialog.table.rowCount() == 0
        assert dialog.count_label.text() == "0 个模型"
        assert dialog.error_label.isHidden()
        assert not dialog.evaluate_button.isEnabled()
        assert dialog.import_button.isEnabled()
        assert dialog.delete_button.isEnabled()
        assert not database.exists()
        dialog.show_details(-1)
        dialog.show_evaluation()
    finally:
        dialog.close()


def test_registered_model_selection_drives_details_and_evaluation(ui_qapp, monkeypatch):
    records = (model_record(), model_record(model_id="b", model_name="测试模型 B"))
    source = Mock(list_records=Mock(return_value=records))
    dialog = AiModelManagerDialog(source)
    details, evaluations = [], []
    monkeypatch.setattr(AiModelDetailsDialog, "exec", lambda self: details.append(self.record))
    monkeypatch.setattr(AiModelEvaluationDialog, "exec", lambda self: evaluations.append(self.record))
    try:
        assert dialog.count_label.text() == "2 个模型"
        assert dialog.table.item(0, 3).text() == "文件缺失"
        dialog.table.cellWidget(1, 4).click()
        assert details == [records[1]]
        assert dialog.selection_label.text() == "已选：测试模型 B"
        dialog.evaluate_button.click()
        assert evaluations == [records[1]]
        dialog.table.clearSelection()
        assert not dialog.evaluate_button.isEnabled()
        dialog.show_evaluation()
        assert evaluations == [records[1]]
        source.list_records.assert_called_once_with()
    finally:
        dialog.close()


def test_manager_typography_and_column_widths_remain_balanced(ui_qapp):
    dialog = AiModelManagerDialog(Mock(list_records=lambda: (model_record(),)))
    try:
        dialog.show()
        ui_qapp.processEvents()
        title = dialog.findChild(QLabel, "aiModelTitle")
        assert title.font().pixelSize() == 23
        assert dialog.table.rowHeight(0) == 68
        assert dialog.table.columnWidth(0) < dialog.table.viewport().width() * 0.5
        assert dialog.table.columnWidth(1) >= 150
        assert dialog.table.item(0, 0).data(Qt.UserRole) == "a.keras"
        dialog.resize(680, 440)
        ui_qapp.processEvents()
        assert dialog.table.columnWidth(0) >= 100
        last_column_right = dialog.table.columnViewportPosition(4) + dialog.table.columnWidth(4)
        assert last_column_right <= dialog.table.viewport().width()
    finally:
        dialog.close()


def test_query_failure_clears_old_records_and_can_be_reloaded(ui_qapp):
    source = Mock()
    source.list_records.side_effect = [
        (model_record(),), ModelCatalogError("模型目录读取失败"), (),
    ]
    dialog = AiModelManagerDialog(source)
    try:
        dialog.load_records()
        assert dialog.records == ()
        assert dialog.table.rowCount() == 0
        assert dialog.count_label.text() == "读取失败"
        assert not dialog.error_label.isHidden()
        assert not dialog.evaluate_button.isEnabled()
        dialog.load_records()
        assert dialog.error_label.isHidden()
        assert dialog.count_label.text() == "0 个模型"
    finally:
        dialog.close()


def test_details_preserve_real_fields_and_distinguish_zero_from_missing(ui_qapp):
    record = model_record(model_name="<b>原始名称</b>", accuracy=0.0,
                          model_description="多行备注\n第二行", config_file_exists=True)
    dialog = AiModelDetailsDialog(record)
    try:
        assert dialog.field_labels["模型名称"].text() == record.model_name
        assert dialog.field_labels["模型名称"].textFormat() == Qt.PlainText
        assert dialog.field_labels["模型文件"].text() == "models/a.keras"
        assert dialog.field_labels["配置文件情况"].text() == "文件存在"
        assert dialog.field_labels["历史评估准确率"].text() == "0.00%"
        assert dialog.field_labels["备注"].text() == "多行备注\n第二行"
    finally:
        dialog.close()
    missing = AiModelDetailsDialog(model_record())
    try:
        assert missing.field_labels["历史评估准确率"].text() == "—"
        assert missing.field_labels["备注"].text() == "—"
    finally:
        missing.close()


def test_evaluation_never_uses_historical_accuracy_as_a_current_result(ui_qapp):
    dialog = AiModelEvaluationDialog(model_record(accuracy=0.97))
    try:
        assert dialog.sample_count_label.text() == "已选 0 条音频"
        assert dialog.result_table.rowCount() == 0
        assert {label.text() for label in dialog.metric_labels.values()} == {"—"}
        assert not dialog.select_audio_button.isEnabled()
        assert not dialog.start_button.isEnabled()
        dialog.start_button.click()
        assert dialog.result_table.rowCount() == 0
        dialog.show()
        ui_qapp.processEvents()
        QTest.keyClick(dialog, Qt.Key_Escape)
        assert dialog.result() == QDialog.Rejected
        assert not dialog.isVisible()
    finally:
        dialog.close()
