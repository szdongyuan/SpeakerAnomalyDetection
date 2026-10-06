from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from base.load_config import LoadUiConfig
from base.product_model_validation import validate_product_model
from base.save_data import save_recorded_data_to_json


class SequenceWidgetTestMetadataOpsMixin:
    """Own operator metadata without changing the product-condition state machine."""

    def _init_test_round_metadata(self):
        self._test_round_metadata = None
        self._test_round_product_model = None
        self._test_metadata_validation_open = False
        self._test_metadata_last_error = None
        self._restore_test_metadata_ui_state()
        self.toolsbar.current_round_spinbox.valueChanged.connect(
            self._sync_test_round_label
        )
        self.toolsbar.sample_number_lineedit.editingFinished.connect(
            self._persist_test_metadata_ui_state
        )
        self.toolsbar.current_round_spinbox.editingFinished.connect(
            self._persist_test_metadata_ui_state
        )
        self._sync_test_round_label()

    def _restore_test_metadata_ui_state(self):
        saved_state = LoadUiConfig.load_last_recorded_info(self.default_logger)
        if not isinstance(saved_state, dict):
            return

        sample_number = saved_state.get("sample_number")
        if isinstance(sample_number, str):
            self.toolsbar.sample_number_lineedit.setText(sample_number.strip())

        current_test_round = saved_state.get("current_test_round")
        if type(current_test_round) is int and 1 <= current_test_round <= 9999:
            self.toolsbar.current_round_spinbox.setValue(current_test_round)

    def _persist_test_metadata_ui_state(self):
        round_edit = self.toolsbar.current_round_spinbox
        round_edit.interpretText()
        save_recorded_data_to_json(
            sample_number=self.toolsbar.sample_number_lineedit.text().strip(),
            current_test_round=round_edit.value(),
        )

    def _sync_test_round_label(self, *_):
        metadata = self._test_round_metadata
        round_number = (
            metadata["test_round"]
            if metadata is not None
            else self.toolsbar.current_round_spinbox.value()
        )
        self.left_panel.set_current_round(round_number)

    def _validate_product_model(self):
        if self._test_metadata_validation_open:
            return False
        editor = self.lineedit_type
        locked_model = self._test_round_product_model
        if locked_model is not None:
            if editor.text() != locked_model:
                return self._show_test_metadata_error(editor, "本轮型号已锁定，不能在测试中途修改。")
            return True
        try:
            model = validate_product_model(editor.text())
        except ValueError as error:
            return self._show_test_metadata_error(editor, str(error))
        editor.setText(model)
        self._clear_product_model_error()
        return True

    def _clear_product_model_error(self):
        # Finishing an edit retries the warning; automatic triggers still deduplicate.
        if (
            not self._test_metadata_validation_open
            and self._test_metadata_last_error is not None
            and self._test_metadata_last_error[0] == self.lineedit_type.objectName()
        ):
            self._test_metadata_last_error = None

    def _validate_test_round_metadata(self):
        if not self._validate_product_model():
            return False
        if self._test_round_metadata is not None:
            return True

        sample_edit = self.toolsbar.sample_number_lineedit
        round_edit = self.toolsbar.current_round_spinbox
        if not sample_edit.text().strip():
            invalid_field = sample_edit
            message = "请先填写样本编号，再开始测试。"
        elif not round_edit.hasAcceptableInput():
            invalid_field = round_edit
            message = "当前测试轮次必须为 1–9999 的整数。"
        else:
            self._test_metadata_last_error = None
            return True

        return self._show_test_metadata_error(invalid_field, message)

    def _show_test_metadata_error(self, invalid_field, message):
        # Serial frames can arrive while QMessageBox runs its nested event loop.
        error_key = (invalid_field.objectName(), invalid_field.text())
        if self._test_metadata_last_error == error_key:
            invalid_field.setFocus(Qt.OtherFocusReason)
            return False
        self._test_metadata_last_error = error_key
        self._test_metadata_validation_open = True
        try:
            QMessageBox.warning(self, "测试信息未填写完整", message)
        finally:
            self._test_metadata_validation_open = False
        invalid_field.setFocus(Qt.OtherFocusReason)
        invalid_field.selectAll()
        return False

    def _begin_test_round_metadata(self):
        if not self._validate_test_round_metadata():
            return False
        if self._test_round_metadata is not None:
            return True

        sample_edit = self.toolsbar.sample_number_lineedit
        round_edit = self.toolsbar.current_round_spinbox
        round_edit.interpretText()
        self._test_round_metadata = {
            "sample_number": sample_edit.text().strip(),
            "test_round": round_edit.value(),
        }
        self._test_round_product_model = self.lineedit_type.text()
        self.lineedit_type.setReadOnly(True)
        sample_edit.setText(self._test_round_metadata["sample_number"])
        self._persist_test_metadata_ui_state()
        sample_edit.setReadOnly(True)
        round_edit.setReadOnly(True)
        self._sync_test_round_label()
        return True

    def _end_test_round_metadata(self):
        self._test_round_metadata = None
        self._test_round_product_model = None
        self.toolsbar.sample_number_lineedit.setReadOnly(False)
        self.toolsbar.current_round_spinbox.setReadOnly(False)
        self._sync_test_round_label()

    def _attach_test_round_metadata(self, recorded_signal_info):
        if self._test_round_metadata is not None:
            recorded_signal_info.update(self._test_round_metadata)
        register = getattr(self, "_register_round_recording", None)
        if callable(register) and recorded_signal_info.get("file_path"):
            register(recorded_signal_info)

    def _cancel_test_metadata_preflight(self):
        # No recording has started. Preserve a partially completed round for retry.
        if not self._manual_product_condition_completed_keys:
            self._reset_manual_product_condition_cycle(clear_waveforms=False)

    def _abort_test_round_metadata(self, reason):
        if self._test_round_metadata is None:
            return False
        self._reset_manual_product_condition_cycle(clear_waveforms=False)
        self._record_workflow_busy = False
        self.player_status_flag = False
        self._awaiting_ok_ng = False
        self._pending_recent_session_append = False
        self.replayer_btn.setDisabled(True)
        self.update_player_btn_is_paused()
        QMessageBox.warning(self, "本轮测试已中止", str(reason))
        return True
