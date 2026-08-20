import json
import os
import time

from PyQt5.QtCore import QEvent, QSignalBlocker
from PyQt5.QtWidgets import QApplication, QMessageBox, QLineEdit

from base.load_config import LoadUiConfig
from base.product_test_program_config import ProductTestProgramConfigManager
from consts import error_code
from consts.running_consts import DEFAULT_DIR


class SequenceWidgetConfigOpsMixin:

    def _is_sequence_config_path(self, path) -> bool:
        if not path or not isinstance(path, (str, bytes, os.PathLike)):
            return False
        load_code, data = LoadUiConfig.load_data_from_json(path)
        if load_code != error_code.OK or not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and isinstance(first.get("seq1"), dict)

    def _get_product_program_manager(self):
        manager = getattr(self, "product_program_manager", None)
        if manager is None:
            manager = ProductTestProgramConfigManager()
            self.product_program_manager = manager
        return manager

    def _get_product_program_registry(self):
        manager = self._get_product_program_manager()
        registry = manager.load_registry()
        self.product_program_registry = registry
        self.active_product_program_file = registry.get("active_file")
        return registry

    def _get_active_product_program_path(self):
        manager = self._get_product_program_manager()
        active_file = str(getattr(self, "active_product_program_file", "") or "").strip()
        if not active_file:
            active_file = str((manager.load_registry() or {}).get("active_file") or "").strip()
            self.active_product_program_file = active_file
        if not active_file:
            return None
        return os.path.join(manager.program_dir, active_file)

    def load_active_product_test_condition_configs(self):
        return LoadUiConfig.load_product_test_program_condition_configs(
            self._get_active_product_program_path()
        )

    def _active_product_program_test_mode_availability(self):
        manager = self._get_product_program_manager()
        registry = manager.load_registry()
        active_file = str((registry or {}).get("active_file") or "").strip()
        self.product_program_registry = registry
        self.active_product_program_file = active_file or None
        if not active_file:
            return False, "当前未选择有效的产品配置，无法进入测试模式"

        load_code, program_data = manager.load_program(active_file)
        if load_code != error_code.OK or not isinstance(program_data, dict):
            return False, "当前产品配置无法读取，无法进入测试模式"

        validation = manager.validate_program(program_data, active_file)
        if not validation.get("is_usable", False):
            details = "\n".join(
                f"- {message}"
                for message in validation.get("use_errors", [])
            )
            reason = "当前产品配置不可用，无法进入测试模式"
            return False, f"{reason}：\n{details}" if details else reason

        if not validation.get("is_test_mode_usable", False):
            details = "\n".join(
                f"- {message}"
                for message in validation.get("test_mode_errors", [])
            )
            reason = "以下工况未启用阈值判定，无法进入测试模式"
            suffix = "\n请启用所有工况的阈值，或使用标记模式。"
            message = (
                f"{reason}：\n{details}{suffix}"
                if details
                else f"{reason}。{suffix}"
            )
            return False, message

        return True, ""

    def _validate_active_product_program_acquisition_modes(self):
        manager = self._get_product_program_manager()
        config_path = self._get_active_product_program_path()
        if not config_path:
            return True, ""
        load_code, program_data = LoadUiConfig.load_data_from_json(config_path)
        if load_code != error_code.OK or not isinstance(program_data, dict):
            return True, ""
        errors = manager.validate_acquisition_modes(program_data)
        return not errors, "\n".join(errors)

    def load_active_product_test_pdf_report_config(self):
        return LoadUiConfig.load_product_test_program_pdf_report_config(
            self._get_active_product_program_path()
        )

    def load_active_product_test_close_trigger_state(self):
        return LoadUiConfig.load_product_test_program_close_trigger_state(
            self._get_active_product_program_path()
        )

    def _resolve_sequence_queue_path(self, queue_name):
        queue_name = str(queue_name or "").strip()
        if not queue_name:
            return None
        if self._is_sequence_config_path(queue_name):
            return os.path.abspath(queue_name)

        manager = self._get_product_program_manager()
        try:
            queue_info = (manager.load_queue_catalog() or {}).get(queue_name)
            queue_path = queue_info.get("path") if isinstance(queue_info, dict) else None
            if self._is_sequence_config_path(queue_path):
                return os.path.abspath(queue_path)
        except Exception:
            pass

        try:
            registry = LoadUiConfig._load_sequence_config_registry(manager.queue_registry_path)
        except Exception:
            registry = {}
        registered_path = registry.get(queue_name) if isinstance(registry, dict) else None
        if self._is_sequence_config_path(registered_path):
            return os.path.abspath(registered_path)

        file_name = queue_name if queue_name.lower().endswith(".json") else f"{queue_name}.json"
        queue_dir = os.path.join(DEFAULT_DIR, "ui", "ui_config", "analysis_sequence_config")
        candidate = os.path.join(queue_dir, file_name)
        if self._is_sequence_config_path(candidate):
            return os.path.abspath(candidate)
        return None

    def _apply_sequence_config_from_path(self, queue_path, update_registry=False):
        if not self._is_sequence_config_path(queue_path):
            return False, "测试队列配置文件不可用"

        load_code, result = LoadUiConfig().load_sequence_config_from_json(queue_path)
        if load_code != error_code.OK or not isinstance(result, list) or not result:
            return False, "测试队列配置读取失败"
        if not isinstance(result[0], dict) or "seq1" not in result[0]:
            return False, "测试队列配置格式错误"

        self.using_config_path = queue_path
        if update_registry:
            LoadUiConfig.update_using_config_path(self.using_config_path)
        self.sequence_config = result
        seq = self.sequence_config[0]["seq1"]
        self.analysis_config = seq.get("analysis_list", {})
        if self.count_board:
            self.count_board.analysis_config = self.analysis_config
            self._refresh_test_mode_availability()
        self._set_sequence_config_available_state(True)
        self.init_data_struct_stimulus_config()
        self.init_fft_and_stft_flag()
        return True, ""

    def _load_sequence_config_for_product_condition(self, condition_config):
        if not isinstance(condition_config, dict):
            return False, "工况配置格式错误"
        condition_name = str(
            condition_config.get("condition_name") or condition_config.get("name") or ""
        ).strip()
        queue_name = str(condition_config.get("test_queue") or "").strip()
        if not queue_name:
            return False, f"{condition_name or '当前工况'} 未绑定测试队列"
        queue_path = self._resolve_sequence_queue_path(queue_name)
        if not queue_path:
            return False, f"{condition_name or '当前工况'} 绑定的测试队列不存在: {queue_name}"
        return self._apply_sequence_config_from_path(queue_path, update_registry=False)

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
                    # 产品测试整轮锁定期间，键盘楔入扫码也必须直接忽略。
                    # QLineEdit 的只读状态只能阻止控件自行修改文本，不能阻止
                    # 下面的 _sn_clear_on_next_scan 分支主动调用 clear()。
                    is_product_round_locked = getattr(
                        self,
                        "_is_sn_locked_for_product_round",
                        None,
                    )
                    if (
                        callable(is_product_round_locked)
                        and is_product_round_locked()
                        and ch
                        and ch.isprintable()
                    ):
                        return True

                    # 循环已进入录音阶段（正转或反转）-> 吞掉任何可打印键盘输入。
                    # 原因：下面的 _sn_clear_on_next_scan 分支会主动调 clear()
                    # 来 "为下一次扫码腾空间"，那条路径绕过 setReadOnly，所以
                    # 必须在这里挡住，否则正转录完后 wedge 模式的二次扫码会
                    # 把循环中 pinned 的 S/N 清空。HID/serial 已在 _commit_barcode
                    # 的 Stage 2 gate 拒绝写入，不走这条分支。
                    #
                    # 刻意只在 forward/reverse 时吞键：扫码后尚未触发录音的
                    # "间隔期" 允许 wedge 重扫覆盖，方便操作员纠正误扫。
                    cur_direction_raw = getattr(self, "_current_trigger_direction", "") or ""
                    cycle_in_progress = str(cur_direction_raw).strip().lower() in ("forward", "reverse")
                    is_sn_locked = getattr(self, "_is_sn_locked_for_cycle", None)
                    if (
                        callable(is_sn_locked)
                        and is_sn_locked()
                        and cycle_in_progress
                        and ch
                        and ch.isprintable()
                    ):
                        return True  # 吞掉事件

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
        user_keys = [k for k in user_keys if self._is_sequence_config_path(registry.get(k))]
        using_config_path = registry.get("using_config_path", None)
        default_path = registry.get("默认配置")
        if not self._is_sequence_config_path(using_config_path):
            using_config_path = None
        if not self._is_sequence_config_path(default_path):
            default_path = None
        # Fallback when using_config_path missing or points to a non-existent file.
        if (not using_config_path) or (isinstance(using_config_path, str) and not os.path.exists(using_config_path)):
            fallback_path = None
            # Prefer user saved/imported entries (if any), otherwise fallback to built-in default.
            if user_keys:
                for k in sorted(user_keys):
                    p = registry.get(k)
                    if isinstance(p, str) and os.path.exists(p) and self._is_sequence_config_path(p):
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
        self._get_product_program_registry()
        # Updating items will trigger currentTextChanged multiple times (clear/add/setIndex).
        # Block signals to avoid re-entrant loads and transient empty-text callbacks.
        self.using_file_combobox.blockSignals(True)
        try:
            self.using_file_combobox.clear()
            self.add_file_to_using_file_combobox()
        finally:
            self.using_file_combobox.blockSignals(False)

    def on_product_test_program_updated(self, *_):
        """Reload product-test programs after the configuration dialog saves."""
        try:
            self.update_using_file_combobox()
            self._sync_product_test_conditions(clear_recent_history=True)
            refresh_serial_runtime = getattr(self, "refresh_serial_product_trigger_runtime", None)
            if callable(refresh_serial_runtime):
                refresh_serial_runtime()
            self.update_player_btn_is_paused()
        except Exception as error:
            self.default_logger.warning(
                f"Failed to refresh product test program after update: {error}"
            )

    def add_file_to_using_file_combobox(self):
        """
        Adds file to the using file combobox.
        """
        registry = self._get_product_program_registry()
        active_file = str((registry or {}).get("active_file") or "")
        selected_key = None
        visible_count = 0
        was_blocked = self.using_file_combobox.blockSignals(True)
        try:
            for item in (registry or {}).get("configs", []):
                if not isinstance(item, dict):
                    continue
                file_name = str(item.get("file") or "").strip()
                name = str(item.get("name") or "").strip()
                if not file_name or not name:
                    continue
                self.using_file_combobox.addItem(name, file_name)
                visible_count += 1
                if active_file and file_name == active_file:
                    selected_key = name

            if visible_count == 0:
                self.using_file_combobox.addItem("无配置", None)
                selected_key = "无配置"

            if selected_key:
                idx = self.using_file_combobox.findText(selected_key)
                if idx >= 0:
                    self.using_file_combobox.setCurrentIndex(idx)
        finally:
            self.using_file_combobox.blockSignals(was_blocked)

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

        product_file = None
        try:
            product_file = self.using_file_combobox.currentData()
        except Exception:
            product_file = None
        if product_file:
            manager = self._get_product_program_manager()
            load_code, program_data = manager.load_program(str(product_file))
            if load_code != error_code.OK or not isinstance(program_data, dict):
                self.restore_previous_configuration()
                QMessageBox.warning(self, "产品配置不可用", "产品配置文件无法读取")
                return
            validation = manager.validate_program(program_data, str(product_file))
            if not validation.get("is_usable", False):
                self.restore_previous_configuration()
                message = "\n".join(validation.get("use_errors", []))
                QMessageBox.warning(
                    self,
                    "产品配置不可用",
                    message or "产品配置校验失败",
                )
                return
            registry = manager.load_registry()
            registry["active_file"] = str(product_file)
            if not manager.save_registry(registry):
                self.restore_previous_configuration()
                QMessageBox.warning(
                    self,
                    "产品配置切换失败",
                    "无法切换使用配置：当前配置记录保存失败，请检查配置目录权限。",
                )
                return
            self.product_program_registry = registry
            self.active_product_program_file = str(product_file)
            sync_product_conditions = getattr(self, "_sync_product_test_conditions", None)
            if callable(sync_product_conditions):
                sync_product_conditions(clear_recent_history=True)
            refresh_serial_runtime = getattr(
                self,
                "refresh_serial_product_trigger_runtime",
                None,
            )
            if callable(refresh_serial_runtime):
                refresh_serial_runtime()
            self.update_player_btn_is_paused()
            reset_manual_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
            if callable(reset_manual_cycle):
                reset_manual_cycle(clear_waveforms=True)
            self.replayer_btn.setDisabled(True)
            self.data_btn.setDisabled(True)
            self.data_struct.store_wave_data = None
            self.data_struct.store_wave_data_multi = None
            self.using_file_combobox.clearFocus()
            if self.lineedit_s_or_n.isEnabled():
                try:
                    self.lineedit_s_or_n.setFocus()
                    self.lineedit_s_or_n.selectAll()
                except Exception:
                    pass
            else:
                self.setFocus()
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
        active_file = getattr(self, "active_product_program_file", None) or getattr(self, "using_config_path", None)
        index = self.using_file_combobox.findData(active_file)
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
            if not isinstance(result, list) or not result or not isinstance(result[0], dict) or "seq1" not in result[0]:
                self.sequence_config = []
                self.analysis_config = dict()
                self._set_sequence_config_available_state(False)
                return
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
        sync_product_conditions = getattr(self, "_sync_product_test_conditions", None)
        if callable(sync_product_conditions):
            sync_product_conditions()

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
