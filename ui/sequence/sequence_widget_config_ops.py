import json
import os
import time

from PyQt5.QtCore import QEvent, QSignalBlocker
from PyQt5.QtWidgets import QApplication, QMessageBox, QLineEdit

from base.load_config import LoadUiConfig
from consts import error_code


class SequenceWidgetConfigOpsMixin:

    def eventFilter(self, obj, event):
        """
        Persist analysis window geometry on move/resize (no close handling).
        """
        try:
            if obj in self._analysis_window_key_by_obj:
                et = event.type()
                if et in (QEvent.Move, QEvent.Resize):
                    key = self._analysis_window_key_by_obj.get(obj)
                    if key:
                        rect = obj.geometry()
                        geo = {"x": rect.x(), "y": rect.y(), "w": rect.width(), "h": rect.height()}
                        self._set_analysis_window_geometry(key, geo)
        except Exception as e:
            # Never break Qt event loop
            self.default_logger.error(f"eventFilter geometry persist error: {e}")

        # 键盘事件捕获（扫码枪键盘楔入模式）
        try:
            # 型号：单击进入编辑态（默认只读）
            # 设计：只读时单击解锁；编辑时单击不反向上锁（否则用户无法用鼠标定位光标）。
            # 回到只读：依赖失去焦点（lineedit_*_lose_focus 已处理）。
            if event.type() == QEvent.MouseButtonPress:
                if obj is self.lineedit_type:
                    try:
                        if isinstance(obj, QLineEdit) and obj.isReadOnly():
                            obj.setReadOnly(False)
                            obj.setFocus()
                            obj.selectAll()
                            return True
                    except Exception:
                        pass

            if event.type() == QEvent.KeyPress and self.barcode_scanner_box.isChecked():
                now = time.monotonic()
                fw = QApplication.focusWidget()

                # 如果 HID 模式刚刚接收到条码，忽略所有键盘输入（避免 HID 和键盘模式同时工作的重复问题）
                ch = event.text()
                # "最简焦点方案"下，型号输入框永远不拦截（保证手动输入不受影响）
                if (
                    ch
                    and ch.isprintable()
                    and now < self._hid_mode_active_until
                    and fw is not self.lineedit_type
                ):
                    return True  # 吞掉事件

                # 焦点在 S/N 输入框
                if fw is self.lineedit_s_or_n:
                    # HID 模式激活窗口内，吞掉键盘输入（避免 HID + 键盘模式重复导致 S/N 内容翻倍）
                    if ch and ch.isprintable() and now < self._hid_mode_active_until:
                        return True  # 吞掉事件

                    # 在"待确认"状态下，下一次扫码先清空旧内容，避免拼接
                    if (
                        self._sn_clear_on_next_scan
                        and not self.player_status_flag
                        and ch
                        and ch.isprintable()
                        and not ch.isspace()
                    ):
                        try:
                            with QSignalBlocker(self.lineedit_s_or_n):
                                self.lineedit_s_or_n.clear()
                            self._barcode_first_char_ts = None
                            self._barcode_last_char_ts = None
                            self._barcode_debounce_timer.stop()
                            self._sn_clear_on_next_scan = False
                        except Exception:
                            pass
                    return super().eventFilter(obj, event)

                # 其余键盘事件交给 BarcodeRouter 做路由/吞键/缓冲
                try:
                    handled = self._barcode_router.handle_keypress(obj, event)
                    if handled is True:
                        return True
                except Exception:
                    # router 逻辑异常时不影响主流程
                    pass

        except Exception:
            # 出异常时不影响主流程
            return super().eventFilter(obj, event)

        return super().eventFilter(obj, event)

    def _load_analysis_window_geometry(self):
        """
        Load persisted analysis window geometries.
        """
        try:
            if not os.path.exists(self._analysis_window_geometry_path):
                os.makedirs(os.path.dirname(self._analysis_window_geometry_path), exist_ok=True)
                with open(self._analysis_window_geometry_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                return {}
            with open(self._analysis_window_geometry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.default_logger.warning(f"Failed to load analysis window geometry: {e}")
            return {}

    def _flush_analysis_window_geometry(self):
        """
        Flush geometry cache to disk (atomic write).
        """
        if not self._analysis_window_geometry_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._analysis_window_geometry_path), exist_ok=True)
            tmp_path = self._analysis_window_geometry_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._analysis_window_geometry, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._analysis_window_geometry_path)
            self._analysis_window_geometry_dirty = False
        except Exception as e:
            self.default_logger.warning(f"Failed to save analysis window geometry: {e}")

    def _set_analysis_window_geometry(self, key: str, geo: dict):
        """
        Update in-memory geometry and schedule a debounced flush.
        """
        if not key or not isinstance(geo, dict):
            return
        norm = self._normalize_geometry(geo)
        if norm is None:
            return
        self._analysis_window_geometry[key] = norm
        self._analysis_window_geometry_dirty = True
        # Debounce writes to avoid heavy IO while dragging
        if not self._analysis_window_geometry_flush_timer.isActive():
            self._analysis_window_geometry_flush_timer.start(200)

    def _get_analysis_window_geometry(self, key: str):
        """
        Get persisted geometry if valid and on-screen; otherwise None.
        """
        if not key:
            return None
        geo = self._analysis_window_geometry.get(key)
        geo = self._normalize_geometry(geo) if isinstance(geo, dict) else None
        if geo is None:
            return None
        if not self._is_geometry_on_any_screen(geo):
            return None
        return geo

    @staticmethod
    def _normalize_geometry(geo: dict):
        """
        Ensure geometry has x/y/w/h ints with sane bounds.
        """
        if not isinstance(geo, dict):
            return None
        try:
            x = int(geo.get("x"))
            y = int(geo.get("y"))
            w = int(geo.get("w"))
            h = int(geo.get("h"))
            # basic sanity
            if w < 200 or h < 150:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        except Exception:
            return None

    @staticmethod
    def _is_geometry_on_any_screen(geo: dict) -> bool:
        """
        Check whether the saved top-left point is within any screen's available geometry.
        """
        try:
            x, y = int(geo["x"]), int(geo["y"])
            for screen in QApplication.screens():
                ag = screen.availableGeometry()
                if ag.contains(x, y):
                    return True
            return True if QApplication.primaryScreen() is None else False
        except Exception:
            return False

    def del_geometry_config(self):
        if (
            hasattr(self, "_analysis_window_geometry_flush_timer")
            and self._analysis_window_geometry_flush_timer.isActive()
        ):
            self._analysis_window_geometry_flush_timer.stop()

        self._analysis_window_geometry = {}
        self._analysis_window_geometry_dirty = False

        file_path = self._analysis_window_geometry_path
        if os.path.exists(file_path):
            os.remove(file_path)

    def get_sequence_config_from_registry(self):
        """
        Retrieves the sequence configuration from the registry.
        """
        registry = LoadUiConfig._load_sequence_config_registry()
        # IMPORTANT: Do not auto-add "默认配置" into registry.
        # The combobox should either never show it, or always show it if it already exists in registry.
        user_keys = [k for k in (registry or {}).keys() if k not in ("using_config_path", "默认配置")]
        using_config_path = registry.get("using_config_path", None)
        default_path = registry.get("默认配置")
        # Fallback when using_config_path missing or points to a non-existent file.
        if (not using_config_path) or (isinstance(using_config_path, str) and not os.path.exists(using_config_path)):
            fallback_path = None
            # Prefer user saved/imported entries (if any), otherwise fallback to built-in default.
            if user_keys:
                for k in sorted(user_keys):
                    p = registry.get(k)
                    if isinstance(p, str) and os.path.exists(p):
                        fallback_path = p
                        break
            if not fallback_path and isinstance(default_path, str) and os.path.exists(default_path):
                fallback_path = default_path

            using_config_path = fallback_path
            if using_config_path:
                LoadUiConfig.update_using_config_path(using_config_path)
        return using_config_path, registry

    def update_using_file_combobox(self):
        """
        Updates the using file combobox.
        """
        self.using_config_path, self.registry = self.get_sequence_config_from_registry()
        # Updating items will trigger currentTextChanged multiple times (clear/add/setIndex).
        # Block signals to avoid re-entrant loads and transient empty-text callbacks.
        self.using_file_combobox.blockSignals(True)
        try:
            self.using_file_combobox.clear()
            self.add_file_to_using_file_combobox()
        finally:
            self.using_file_combobox.blockSignals(False)

    def add_file_to_using_file_combobox(self):
        """
        Adds file to the using file combobox.
        """
        selected_key = None
        using_path = self.registry.get("using_config_path")

        keys = [k for k in (self.registry or {}).keys() if k != "using_config_path"]
        # Do NOT filter "默认配置" by business logic here:
        # - If registry contains "默认配置", it must appear in combobox (always pinned on top).
        # - If registry does not contain it, it won't appear.
        ordered_keys = []
        if "默认配置" in keys:
            ordered_keys.append("默认配置")
            keys.remove("默认配置")
        ordered_keys.extend(sorted(keys))

        # Only show entries whose file path exists. If none exist, show "无配置".
        visible_count = 0
        for key in ordered_keys:
            value = self.registry.get(key)
            if not isinstance(value, str) or (value and not os.path.exists(value)):
                continue
            self.using_file_combobox.addItem(key, value)
            visible_count += 1
            if using_path and value == using_path:
                selected_key = key

        if visible_count == 0:
            self.using_file_combobox.addItem("无配置", None)
            selected_key = "无配置"

        # Select the item matching the current using_config_path (if any)
        if selected_key:
            idx = self.using_file_combobox.findText(selected_key)
            if idx >= 0:
                self.using_file_combobox.setCurrentIndex(idx)

    def on_using_file_combobox_changed(self, text):
        """
        Handles the change of the using file combobox.
        """
        # Prefer the item's userData (full file path). During combobox refresh,
        # `text` may temporarily be empty which would otherwise resolve to None.
        if self.player_status_flag:
            self.restore_previous_configuration()
            QMessageBox.warning(self, "警告", "正在录音，请稍后...")
            return

        path = None
        try:
            path = self.using_file_combobox.currentData()
        except Exception:
            path = None
        if not path:
            path = self.registry.get(text)

        self.using_config_path = path
        LoadUiConfig.update_using_config_path(self.using_config_path)
        self.get_sequence_config_from_json()
        self.init_data_struct_stimulus_config()
        self.update_player_btn_is_paused()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None

        # 1. 强制清除下拉框焦点
        self.using_file_combobox.clearFocus()
        # 2. 尝试聚焦 S/N 框 (提升体验)
        if self.lineedit_s_or_n.isEnabled():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()  # 全选，方便覆盖旧条码
            except Exception:
                pass
        else:
            self.setFocus()  # 给主窗口，依靠 BarcodeRouter 后台捕获

    def restore_previous_configuration(self):
        """恢复到之前的配置选项"""
        index = self.using_file_combobox.findData(self.using_config_path)
        if index >= 0:
            self.using_file_combobox.blockSignals(True)
            self.using_file_combobox.setCurrentIndex(index)
            self.using_file_combobox.blockSignals(False)
            self.default_logger.warning("已恢复到之前的配置选项")

    def get_sequence_config_from_json(self):
        """
        Retrieves the sequence configuration from a JSON file.

        This method attempts to load the sequence configuration from a JSON file by calling the `load_sequence_from_json()` method.
        If the loading is successful and the result is valid, it returns the configuration; otherwise, it returns an empty dictionary.

        Returns:
            dict: The sequence configuration if loading is successful and the result is valid; otherwise, an empty dictionary.
        """
        # Avoid noisy stdout prints; use logger if needed for debugging.
        # self.default_logger.debug(f"Loading sequence config: {self.using_config_path}")
        load_code, result = LoadUiConfig().load_sequence_config_from_json(self.using_config_path)
        if load_code == error_code.OK and result:
            self.sequence_config = result
            seq = self.sequence_config[0]["seq1"]
            self.analysis_config = seq.get("analysis_list", {})
            mode = seq["acq"]["mode"]
            if mode == "IMPORT_AUDIO":
                self.replayer_btn.setDisabled(True)
            if self.count_board:
                self.count_board.analysis_config = seq.get("analysis_list", {})
                self._refresh_test_mode_availability()
            self._set_sequence_config_available_state(True)
            self._missing_config_prompted = False
        else:
            self.sequence_config = []
            self.analysis_config = dict()
            self._set_sequence_config_available_state(False)
            # Only show prompt after login success (window shown).
            if getattr(self, "_missing_config_prompt_enabled", False) and not getattr(
                self, "_missing_config_prompted", False
            ):
                QMessageBox.warning(
                    self,
                    "提示",
                    "当前未找到可用配置文件。\n"
                    "请在上方【使用配置】下拉框中选择配置；\n"
                    "如无可选项，请到【功能-测试队列】中保存或导入配置。",
                )
                self._missing_config_prompted = True

    def _set_sequence_config_available_state(self, available: bool):
        """
        Enable/disable key actions based on whether a valid sequence config is loaded.
        """
        try:
            if available:
                self.update_player_btn_is_paused()
                # replay/data depend on runtime state; keep conservative defaults here
            else:
                self.player_btn.setDisabled(True)
                self.replayer_btn.setDisabled(True)
                self.data_btn.setDisabled(True)
        except Exception:
            # During early init some widgets may not be ready; ignore safely.
            pass
