"""Non-blocking, window-modal presentation for sequence exports."""

from __future__ import annotations

from typing import Any, Callable

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog

from ui.custom_ui_widget.widgets import MessageBox
from ui.sequence.sequence_messages import (
    AbortShutdownRequested,
    ConfirmShutdownCancellationRequested,
    IgnoreExportFailureRequested,
    IgnoreShutdownFlushFailureRequested,
    RetryExportRequested,
    RetryShutdownFlushRequested,
)


class SequenceExportView(QObject):
    """Present progress and decisions without a nested Qt event loop."""

    retry_requested = pyqtSignal(object)
    ignore_requested = pyqtSignal(object)
    confirm_shutdown_requested = pyqtSignal(object)
    abort_shutdown_requested = pyqtSignal(object)
    shutdown_retry_requested = pyqtSignal(object)
    shutdown_ignore_requested = pyqtSignal(object)

    def __init__(
        self,
        *,
        parent: QObject | None = None,
        progress_dialog_factory: Callable[..., Any] = QProgressDialog,
        failure_dialog_factory: Callable[..., Any] = MessageBox,
        fallback_failure_dialog_factory: Callable[..., Any] = MessageBox,
        logger: Any = None,
    ) -> None:
        super().__init__(parent)
        self._dialog_parent = parent
        self._progress_dialog_factory = progress_dialog_factory
        self._failure_dialog_factory = failure_dialog_factory
        self._fallback_failure_dialog_factory = fallback_failure_dialog_factory
        self._logger = logger
        self._progress_dialog: Any = None
        self._failure_dialog: Any = None
        self._shutdown_confirmation_dialog: Any = None
        self._shutdown_waiting_dialog: Any = None
        self._shutdown_failure_dialog: Any = None
        self._shutdown_confirmation_generation: int | None = None
        self._shutdown_waiting_generation: int | None = None
        self._shutdown_failure_identity: tuple[int, str, str] | None = None
        self._active_identity: tuple[str, str] | None = None
        self._recovery_pending_identity: tuple[str, str] | None = None
        self._generation = 0
        self._destroyed = False
        self.destroyed.connect(self._mark_destroyed)

    def _mark_destroyed(self, *_args: Any) -> None:
        self._destroyed = True
        self._generation += 1
        if self._active_identity is not None:
            self._recovery_pending_identity = self._active_identity

    @property
    def recovery_pending_identity(self) -> tuple[str, str] | None:
        return self._recovery_pending_identity

    def _log(self, message: str) -> None:
        try:
            callback = getattr(self._logger, "warning", None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    @staticmethod
    def _close_dialog(dialog: Any) -> None:
        if dialog is None:
            return
        try:
            dialog.close()
        except BaseException:
            pass
        try:
            dialog.deleteLater()
        except BaseException:
            pass

    def prepare_failure_identity(self, job_id: str, attempt_id: str) -> bool:
        """Activate an exact recovery identity without pretending I/O started."""
        identity = (job_id, attempt_id)
        self._generation += 1
        self._active_identity = identity
        self._close_dialog(self._progress_dialog)
        self._close_dialog(self._failure_dialog)
        self._progress_dialog = None
        self._failure_dialog = None
        if self._destroyed:
            self._recovery_pending_identity = identity
            return False
        self._recovery_pending_identity = None
        return True

    def show_progress(self, job_id: str, attempt_id: str) -> bool:
        if self._destroyed:
            self._active_identity = (job_id, attempt_id)
            self._recovery_pending_identity = (job_id, attempt_id)
            return False
        self._generation += 1
        self._active_identity = (job_id, attempt_id)
        self._recovery_pending_identity = None
        self._close_dialog(self._failure_dialog)
        self._failure_dialog = None
        self._close_dialog(self._progress_dialog)
        try:
            dialog = self._progress_dialog_factory(self._dialog_parent)
            dialog.setWindowTitle("正在保存")
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setLabelText("正在保存数据，请稍候...")
            dialog.setCancelButton(None)
            dialog.setRange(0, 0)
            dialog.open()
        except BaseException as error:
            self._close_dialog(locals().get("dialog"))
            self._log("export progress presentation failed")
            return False
        self._progress_dialog = dialog
        return True

    def _show_decision(
        self,
        job_id: str,
        attempt_id: str,
        failures: Any,
        *,
        allow_ignore: bool,
    ) -> bool:
        identity = (job_id, attempt_id)
        if self._active_identity != identity:
            return False
        if self._destroyed:
            self._recovery_pending_identity = identity
            return False
        self._close_dialog(self._progress_dialog)
        self._progress_dialog = None
        self._close_dialog(self._failure_dialog)
        self._failure_dialog = None
        self._generation += 1
        generation = self._generation
        failure_values = tuple(failures or ())

        def create_dialog(factory: Callable[..., Any]) -> Any:
            candidate = None
            try:
                candidate = factory(self._dialog_parent)
                candidate.setWindowModality(Qt.WindowModal)
                candidate.setIcon(MessageBox.Warning)
                candidate.setWindowTitle("数据保存失败")
                candidate.setText(
                    "无法保存数据到文件，可能是文件被占用、保存目录不可达或权限不足。\n"
                    "请检查保存目录或关闭相关文件后重试。"
                )
                details = []
                for failure in failure_values[:5]:
                    name = getattr(failure, "config_name", None)
                    message = getattr(failure, "message", None)
                    if name is None and isinstance(failure, (tuple, list)):
                        name = failure[0] if failure else "unknown"
                        message = failure[1] if len(failure) > 1 else "failed"
                    details.append(f"{name or 'unknown'}: {message or 'failed'}")
                if len(failure_values) > 5:
                    details.append("...")
                if details:
                    candidate.setInformativeText("\n".join(details))
                retry_button = candidate.addButton(
                    "重试", MessageBox.AcceptRole
                )
                ignore_button = (
                    candidate.addButton("忽略", MessageBox.RejectRole)
                    if allow_ignore
                    else None
                )
                candidate.setDefaultButton(retry_button)

                def decide(button: Any) -> None:
                    if (
                        self._destroyed
                        or generation != self._generation
                        or self._active_identity != (job_id, attempt_id)
                    ):
                        return
                    if button is retry_button:
                        self.retry_requested.emit(
                            RetryExportRequested(job_id, attempt_id)
                        )
                    elif ignore_button is not None and button is ignore_button:
                        self.ignore_requested.emit(
                            IgnoreExportFailureRequested(job_id, attempt_id)
                        )

                candidate.buttonClicked.connect(decide)
                candidate.open()
                return candidate
            except BaseException:
                self._close_dialog(candidate)
                raise

        dialog = None
        for factory in (
            self._failure_dialog_factory,
            self._fallback_failure_dialog_factory,
        ):
            try:
                dialog = create_dialog(factory)
                break
            except BaseException:
                dialog = None
        if dialog is None:
            self._recovery_pending_identity = identity
            self._log("export failure presentation failed")
            return False
        self._failure_dialog = dialog
        self._recovery_pending_identity = None
        return True

    def show_failure(
        self, job_id: str, attempt_id: str, failures: Any
    ) -> bool:
        return self._show_decision(
            job_id, attempt_id, failures, allow_ignore=True
        )

    def show_publication_failure(
        self, job_id: str, attempt_id: str, failures: Any
    ) -> bool:
        """Terminal publication can only be retried, never ignored."""
        return self._show_decision(
            job_id, attempt_id, failures, allow_ignore=False
        )

    def finish(self, job_id: str, attempt_id: str) -> bool:
        if self._active_identity != (job_id, attempt_id):
            return False
        self._generation += 1
        self._active_identity = None
        self._recovery_pending_identity = None
        self._close_dialog(self._progress_dialog)
        self._close_dialog(self._failure_dialog)
        self._progress_dialog = None
        self._failure_dialog = None
        return True

    def raise_active(self) -> bool:
        dialog = self._failure_dialog or self._progress_dialog
        if dialog is None:
            return False
        try:
            dialog.raise_()
            dialog.activateWindow()
        except BaseException:
            return False
        return True

    def show_shutdown_confirmation(self, shutdown_generation: int) -> bool:
        if self._destroyed:
            return False
        if self._shutdown_confirmation_generation == shutdown_generation:
            return self.raise_shutdown(shutdown_generation)
        self.finish_shutdown_confirmation(
            self._shutdown_confirmation_generation
            if self._shutdown_confirmation_generation is not None
            else shutdown_generation
        )
        dialog = None
        try:
            dialog = self._failure_dialog_factory(self._dialog_parent)
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setIcon(MessageBox.Warning)
            dialog.setWindowTitle("确认退出")
            dialog.setText("当前测试仍在进行，是否取消当前任务并退出？")
            confirm_button = dialog.addButton("退出", MessageBox.AcceptRole)
            cancel_button = dialog.addButton("取消", MessageBox.RejectRole)
            dialog.setDefaultButton(cancel_button)

            def decide(button: Any) -> None:
                if (
                    self._destroyed
                    or self._shutdown_confirmation_generation
                    != shutdown_generation
                ):
                    return
                if button is confirm_button:
                    self.confirm_shutdown_requested.emit(
                        ConfirmShutdownCancellationRequested(shutdown_generation)
                    )
                elif button is cancel_button:
                    self.abort_shutdown_requested.emit(
                        AbortShutdownRequested(shutdown_generation)
                    )

            dialog.buttonClicked.connect(decide)
            dialog.open()
            self._shutdown_confirmation_generation = shutdown_generation
            self._shutdown_confirmation_dialog = dialog
            return True
        except BaseException:
            self._close_dialog(dialog)
            self._shutdown_confirmation_dialog = None
            self._shutdown_confirmation_generation = None
            self._log("shutdown confirmation presentation failed")
            return False

    def finish_shutdown_confirmation(self, shutdown_generation: int) -> bool:
        if self._shutdown_confirmation_generation != shutdown_generation:
            return False
        self._close_dialog(self._shutdown_confirmation_dialog)
        self._shutdown_confirmation_dialog = None
        self._shutdown_confirmation_generation = None
        return True

    def show_shutdown_waiting(self, shutdown_generation: int) -> bool:
        if self._destroyed:
            return False
        if self._shutdown_waiting_generation == shutdown_generation:
            return self.raise_shutdown(shutdown_generation)
        self.finish_shutdown_waiting(
            self._shutdown_waiting_generation
            if self._shutdown_waiting_generation is not None
            else shutdown_generation
        )
        dialog = None
        try:
            dialog = self._progress_dialog_factory(self._dialog_parent)
            dialog.setWindowTitle("正在退出")
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setLabelText("正在等待当前任务安全结束，请稍候...")
            dialog.setCancelButton(None)
            dialog.setRange(0, 0)
            dialog.open()
            self._shutdown_waiting_generation = shutdown_generation
            self._shutdown_waiting_dialog = dialog
            return True
        except BaseException:
            self._close_dialog(dialog)
            self._shutdown_waiting_dialog = None
            self._shutdown_waiting_generation = None
            self._log("shutdown cancellation progress presentation failed")
            return False

    def finish_shutdown_waiting(self, shutdown_generation: int) -> bool:
        if self._shutdown_waiting_generation != shutdown_generation:
            return False
        self._close_dialog(self._shutdown_waiting_dialog)
        self._shutdown_waiting_dialog = None
        self._shutdown_waiting_generation = None
        return True

    def show_shutdown_failure(
        self,
        shutdown_generation: int,
        job_id: str,
        attempt_id: str,
        failures: Any,
    ) -> bool:
        identity = (shutdown_generation, job_id, attempt_id)
        if self._destroyed:
            return False
        if self._shutdown_failure_identity == identity:
            return self.raise_shutdown(shutdown_generation)
        self._close_dialog(self._progress_dialog)
        self._close_dialog(self._failure_dialog)
        self._progress_dialog = None
        self._failure_dialog = None
        self._close_dialog(self._shutdown_failure_dialog)
        self._shutdown_failure_dialog = None
        self._shutdown_failure_identity = None
        dialog = None
        try:
            dialog = self._failure_dialog_factory(self._dialog_parent)
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setIcon(MessageBox.Warning)
            dialog.setWindowTitle("Excel同步失败")
            dialog.setText(
                "无法将数据同步到Excel文件，可能是文件被占用或权限不足。\n"
                "请关闭相关Excel文件后重试。"
            )
            retry_button = dialog.addButton("重试", MessageBox.AcceptRole)
            ignore_button = dialog.addButton("忽略", MessageBox.RejectRole)
            dialog.setDefaultButton(retry_button)

            def decide(button: Any) -> None:
                if self._destroyed or self._shutdown_failure_identity != identity:
                    return
                if button is retry_button:
                    self.shutdown_retry_requested.emit(
                        RetryShutdownFlushRequested(
                            shutdown_generation, job_id, attempt_id
                        )
                    )
                elif button is ignore_button:
                    self.shutdown_ignore_requested.emit(
                        IgnoreShutdownFlushFailureRequested(
                            shutdown_generation, job_id, attempt_id
                        )
                    )

            dialog.buttonClicked.connect(decide)
            dialog.open()
            self._shutdown_failure_dialog = dialog
            self._shutdown_failure_identity = identity
            return True
        except BaseException:
            self._close_dialog(dialog)
            self._shutdown_failure_dialog = None
            self._shutdown_failure_identity = None
            self._log("shutdown flush failure presentation failed")
            return False

    def finish_shutdown_failure(
        self, shutdown_generation: int, job_id: str, attempt_id: str
    ) -> bool:
        if self._shutdown_failure_identity != (
            shutdown_generation,
            job_id,
            attempt_id,
        ):
            return False
        self._close_dialog(self._shutdown_failure_dialog)
        self._shutdown_failure_dialog = None
        self._shutdown_failure_identity = None
        return True

    def raise_shutdown(self, shutdown_generation: int) -> bool:
        dialog = None
        if self._shutdown_confirmation_generation == shutdown_generation:
            dialog = self._shutdown_confirmation_dialog
        elif self._shutdown_waiting_generation == shutdown_generation:
            dialog = self._shutdown_waiting_dialog
        elif (
            self._shutdown_failure_identity is not None
            and self._shutdown_failure_identity[0] == shutdown_generation
        ):
            dialog = self._shutdown_failure_dialog
        if dialog is None:
            return self.raise_active()
        try:
            dialog.raise_()
            dialog.activateWindow()
            return True
        except BaseException:
            return False

    def disconnect(self) -> None:
        self._generation += 1
        self._destroyed = True
        if self._active_identity is not None:
            self._recovery_pending_identity = self._active_identity
        self._close_dialog(self._progress_dialog)
        self._close_dialog(self._failure_dialog)
        self._close_dialog(self._shutdown_confirmation_dialog)
        self._close_dialog(self._shutdown_waiting_dialog)
        self._close_dialog(self._shutdown_failure_dialog)
        self._progress_dialog = None
        self._failure_dialog = None
        self._shutdown_confirmation_dialog = None
        self._shutdown_waiting_dialog = None
        self._shutdown_failure_dialog = None
        self._shutdown_confirmation_generation = None
        self._shutdown_waiting_generation = None
        self._shutdown_failure_identity = None


__all__ = ["SequenceExportView"]
