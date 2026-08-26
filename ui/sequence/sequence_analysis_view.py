"""Qt presentation boundary for sequence-analysis results and feedback."""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import QApplication

from consts import ui_style_const
from base.analysis_warning_preferences import (
    save_uncalibrated_microphone_warning_suppressed,
)
from ui.custom_ui_widget.widgets import CheckBox, MessageBox
from ui.signal_analysis_window import AnalysisResultSummaryWindow

from ui.sequence.sequence_analysis_model import (
    SequenceAnalysisModel,
    normalize_analysis_geometry,
)


class SequenceAnalysisView:
    """Own analysis windows, feedback, and persisted window geometry."""

    SUMMARY_KEY = "__analysis_result_summary__"

    def __init__(
        self,
        model: SequenceAnalysisModel,
        *,
        parent: Any = None,
        logger: Any = None,
        geometry_path: str | os.PathLike[str] | None = None,
        warning_presenter: Callable[[str, str], None] | None = None,
        uncalibrated_warning_presenter: Callable[[str], None] | None = None,
        summary_factory: Callable[[Mapping[str, Any]], Any] = AnalysisResultSummaryWindow,
    ) -> None:
        self.model = model
        self.parent = parent
        self.logger = logger
        self.geometry_path = None if geometry_path is None else Path(geometry_path)
        self.warning_presenter = warning_presenter or self._default_warning
        self.uncalibrated_warning_presenter = (
            uncalibrated_warning_presenter or self.present_uncalibrated_warning
        )
        self.summary_factory = summary_factory
        self.summary_window: Any = None
        self.feedback_dialogs: list[Any] = []
        self.window_keys: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
        self.geometry_flush_timer = QTimer(parent)
        self.geometry_flush_timer.setSingleShot(True)
        self.geometry_flush_timer.timeout.connect(self.flush_geometry)
        self.model.replace_geometry(self.load_geometry())

    @property
    def analysis_windows(self) -> list[Any]:
        return self.model.analysis_instances

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self.logger, level, None)
        if callable(callback):
            callback(message)

    def _default_warning(self, title: str, text: str) -> None:
        MessageBox.warning(self.parent, title, text)

    def present_uncalibrated_warning(self, text: str) -> None:
        dialog = MessageBox(self.parent)
        dialog.setIcon(MessageBox.Warning)
        dialog.setWindowTitle("提示")
        dialog.setText(text)
        check_box = CheckBox("不在提示")
        dialog.setCheckBox(check_box)
        dialog.setWindowModality(Qt.WindowModal)
        self.feedback_dialogs.append(dialog)

        def finished(_result: int) -> None:
            if check_box.isChecked():
                save_uncalibrated_microphone_warning_suppressed(
                    logger=self.logger
                )
            if dialog in self.feedback_dialogs:
                self.feedback_dialogs.remove(dialog)

        dialog.finished.connect(finished)
        dialog.open()

    def reset_output(self) -> None:
        self.model.reset_output()
        self.summary_window = None

    def close_windows(self) -> None:
        for window in list(self.model.analysis_instances):
            try:
                if window is not None:
                    window.close()
            except (RuntimeError, TypeError) as error:
                self._log("warning", f"analysis window close failed: {error}")
        self.model.analysis_instances.clear()
        self.model.analysis_registry.clear()
        summary = self.summary_window
        self.summary_window = None
        if summary is not None:
            try:
                summary.close()
            except (RuntimeError, TypeError) as error:
                self._log("warning", f"analysis summary close failed: {error}")
        for dialog in tuple(self.feedback_dialogs):
            try:
                dialog.close()
            except (RuntimeError, TypeError) as error:
                self._log("warning", f"analysis feedback close failed: {error}")
        self.feedback_dialogs.clear()
        self.window_keys.clear()

    def show_channel_mismatch(
        self,
        analysis_name: str,
        *,
        error: Exception | None = None,
        mismatch_info: Mapping[str, Any] | None = None,
    ) -> None:
        configured_channel_text = "未知"
        active_channels_text = "未知"
        if isinstance(mismatch_info, Mapping):
            raw_channel = mismatch_info.get("raw_channel")
            active_channels = mismatch_info.get("active_input_channels", ())
            try:
                configured_channel_text = f"In{int(raw_channel) + 1}"
            except (TypeError, ValueError, OverflowError):
                configured_channel_text = str(raw_channel)
            try:
                active_channels_text = ", ".join(
                    f"In{int(channel) + 1}" for channel in active_channels
                ) or "无"
            except (TypeError, ValueError, OverflowError):
                active_channels_text = str(active_channels)
        detail_text = "" if error is None else f"\n\n详细信息: {error}"
        self.warning_presenter(
            "通道配置不匹配",
            f"{analysis_name} 配置通道与本次录制通道不一致。\n"
            f"当前配置通道: {configured_channel_text}\n"
            f"本次录制通道: {active_channels_text}\n"
            "请在分析参数中重新选择通道后再分析。"
            f"{detail_text}",
        )

    def present_calibration_warnings(
        self,
        messages: list[str],
        *,
        missing_message: str,
        suppress_missing: bool,
        record_only_channels: list[int],
        channel_formatter: Callable[[int], str],
    ) -> None:
        has_missing = missing_message in messages
        remaining = [message for message in messages if message != missing_message]
        if has_missing and not suppress_missing:
            if record_only_channels:
                text = "\n".join(
                    [
                        missing_message,
                        "未校准通道：",
                        *(f"• {channel_formatter(channel)}" for channel in record_only_channels),
                        *remaining,
                    ]
                )
            elif len(messages) == 1:
                text = messages[0]
            else:
                text = "\n".join(f"• {message}" for message in messages)
            self.uncalibrated_warning_presenter(text)
            return
        if len(remaining) == 1:
            text = remaining[0]
        else:
            text = "\n".join(f"• {message}" for message in remaining)
        if text:
            self.warning_presenter("提示", text)

    def show_instance(
        self,
        instance: Any,
        *,
        key: str | None,
        default_geometry: Mapping[str, int],
    ) -> None:
        geometry = self.get_geometry(key) if key else None
        if geometry is None:
            geometry = dict(default_geometry)
            if key:
                self.set_geometry(key, geometry)
        instance.setGeometry(
            int(geometry["x"]),
            int(geometry["y"]),
            int(geometry["w"]),
            int(geometry["h"]),
        )
        instance.setMinimumSize(QSize(200, 155))
        if key:
            self.window_keys[instance] = key
            instance.installEventFilter(self.parent)
        instance.show()

    def show_summary(
        self,
        result_dict: Mapping[str, Any],
        width: int,
        height: int,
    ) -> None:
        if not isinstance(result_dict, Mapping) or not result_dict:
            return
        if self.summary_window is None:
            self.summary_window = self.summary_factory(result_dict)
        else:
            setter = getattr(self.summary_window, "set_results", None)
            if callable(setter):
                setter(result_dict)
            else:
                self.summary_window = self.summary_factory(result_dict)
        summary = self.summary_window
        setattr(summary, "_sequence_analysis_key", self.SUMMARY_KEY)
        default_geometry = {
            "x": int(width),
            "y": int(height),
            "w": ui_style_const.scale_size_px(600),
            "h": ui_style_const.scale_size_px(500),
        }
        geometry = self.get_geometry(self.SUMMARY_KEY)
        if geometry is None:
            geometry = default_geometry
            self.set_geometry(self.SUMMARY_KEY, geometry)
        summary.setGeometry(
            int(geometry["x"]),
            int(geometry["y"]),
            int(geometry["w"]),
            int(geometry["h"]),
        )
        summary.setMinimumSize(QSize(360, 220))
        if summary not in self.window_keys:
            self.window_keys[summary] = self.SUMMARY_KEY
            summary.installEventFilter(self.parent)
        summary.show()
        summary.raise_()
        activate = getattr(summary, "activateWindow", None)
        if callable(activate):
            activate()

    def geometry_event(self, obj: Any) -> bool:
        try:
            key = self.window_keys.get(obj)
            if not key:
                return False
            rect = obj.geometry()
            self.set_geometry(
                key,
                {
                    "x": rect.x(),
                    "y": rect.y(),
                    "w": rect.width(),
                    "h": rect.height(),
                },
            )
            return True
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            self._log("error", f"analysis geometry persist error: {error}")
            return False

    def load_geometry(self) -> dict[str, dict[str, int]]:
        if self.geometry_path is None:
            return {}
        try:
            if not self.geometry_path.exists():
                self.geometry_path.parent.mkdir(parents=True, exist_ok=True)
                self.geometry_path.write_text("{}", encoding="utf-8")
                return {}
            value = json.loads(self.geometry_path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                return {}
            normalized: dict[str, dict[str, int]] = {}
            for key, geometry in value.items():
                item = self.normalize_geometry(geometry)
                if item is None:
                    self._log(
                        "warning", f"Ignored invalid analysis geometry for {key!r}"
                    )
                    continue
                normalized[str(key)] = item
            return normalized
        except (OSError, ValueError, TypeError) as error:
            self._log("warning", f"Failed to load analysis window geometry: {error}")
            return {}

    def flush_geometry(self) -> None:
        if self.geometry_path is None or not self.model.geometry_dirty:
            return
        try:
            self.geometry_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.geometry_path.with_suffix(self.geometry_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.model.geometry, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.geometry_path)
            self.model.geometry_dirty = False
        except OSError as error:
            self._log("warning", f"Failed to save analysis window geometry: {error}")

    def set_geometry(self, key: str, geometry: Mapping[str, Any]) -> None:
        normalized = self.normalize_geometry(geometry)
        if not key or normalized is None:
            return
        self.model.set_geometry(key, normalized)
        if not self.geometry_flush_timer.isActive():
            self.geometry_flush_timer.start(200)

    def get_geometry(self, key: str | None) -> dict[str, int] | None:
        if not key:
            return None
        normalized = self.normalize_geometry(self.model.geometry.get(key))
        if normalized is None or not self.is_geometry_on_any_screen(normalized):
            return None
        return normalized

    @staticmethod
    def normalize_geometry(geometry: Mapping[str, Any] | None) -> dict[str, int] | None:
        return normalize_analysis_geometry(geometry)

    @staticmethod
    def is_geometry_on_any_screen(geometry: Mapping[str, int]) -> bool:
        try:
            x, y = int(geometry["x"]), int(geometry["y"])
            for screen in QApplication.screens():
                if screen.availableGeometry().contains(x, y):
                    return True
            return QApplication.primaryScreen() is None
        except (TypeError, ValueError, KeyError, RuntimeError):
            return False

    def clear_geometry(self) -> None:
        if self.geometry_flush_timer.isActive():
            self.geometry_flush_timer.stop()
        self.model.replace_geometry({})
        if self.geometry_path is not None and self.geometry_path.exists():
            self.geometry_path.unlink()
