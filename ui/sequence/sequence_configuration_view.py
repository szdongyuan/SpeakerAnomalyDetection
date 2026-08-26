"""Thin toolbar and warning adapter for sequence configuration presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


MISSING_CONFIGURATION_MESSAGE = (
    "当前未找到可用配置文件。\n"
    "请在上方【使用配置】下拉框中选择配置；\n"
    "如无可选项，请到【功能-测试队列】中保存或导入配置。"
)
_CURRENT_SELECTION = object()


@dataclass(frozen=True, slots=True)
class ConfigurationViewState:
    action_availability: tuple[bool | None, bool | None, bool | None]
    presentation_state: tuple[bool, bool]
    selection_path: Any
    configuration_entries: tuple[tuple[str, Any], ...] | None
    combobox_had_focus: bool | None
    serial_input_had_focus: bool | None
    parent_had_focus: bool | None


def _default_warning(parent: Any, title: str, message: str) -> None:
    from ui.custom_ui_widget.widgets import MessageBox

    MessageBox.warning(parent, title, message)


class SequenceConfigurationView:
    """Present configuration state without loading files or owning domain data."""

    def __init__(
        self,
        *,
        parent: Any = None,
        combobox: Any = None,
        warning: Callable[[Any, str, str], None] | None = None,
        player_button: Any = None,
        replay_button: Any = None,
        data_button: Any = None,
        serial_input: Any = None,
    ) -> None:
        self.parent = parent
        self.combobox = combobox
        self._warning = warning or _default_warning
        self.player_button = player_button
        self.replay_button = replay_button
        self.data_button = data_button
        self.serial_input = serial_input
        self._availability_callback: Callable[[], Any] | None = None
        self._runtime_readiness_provider: (
            Callable[[str | None], tuple[bool, bool]] | None
        ) = None
        self._missing_config_prompt_enabled = False
        self._missing_config_prompted = False

    def warn(self, title: str, message: str) -> None:
        self._warning(self.parent, title, message)

    def populate_configuration_entries(
        self,
        entries: Iterable[tuple[str, str]],
        *,
        using_config_path: Any,
        clear_first: bool = False,
    ) -> None:
        combo = self.combobox
        if combo is None:
            return
        previous_blocked = combo.blockSignals(True) if clear_first else None
        try:
            if clear_first:
                combo.clear()
            visible = list(entries)
            if not visible:
                combo.addItem("无配置", None)
                selected_index = combo.findText("无配置")
            else:
                for name, path in visible:
                    combo.addItem(name, path)
                selected_index = combo.findData(using_config_path)
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
        finally:
            if clear_first:
                combo.blockSignals(previous_blocked)

    def selected_path(self, text: str, registry: dict[Any, Any]) -> Any:
        path = None
        if self.combobox is not None:
            try:
                path = self.combobox.currentData()
            except (AttributeError, RuntimeError):
                path = None
        return path or registry.get(text)

    def restore_selection(self, using_config_path: Any) -> bool:
        combo = self.combobox
        if combo is None:
            return False
        index = combo.findData(using_config_path)
        if index < 0:
            return False
        previous_blocked = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(previous_blocked)
        return True

    def focus_after_selection(self) -> None:
        if self.combobox is not None:
            self.combobox.clearFocus()
        serial_input = self.serial_input
        if serial_input is not None and serial_input.isEnabled():
            try:
                serial_input.setFocus()
                serial_input.selectAll()
                return
            except RuntimeError:
                pass
        if self.parent is not None:
            self.parent.setFocus()

    def bind_availability_callback(
        self, callback: Callable[[], Any] | None
    ) -> None:
        self._availability_callback = callback

    def bind_runtime_readiness_provider(
        self,
        provider: Callable[[str | None], tuple[bool, bool]] | None,
    ) -> None:
        self._runtime_readiness_provider = provider

    def refresh_availability(self) -> None:
        if self._availability_callback is not None:
            self._availability_callback()

    def set_sequence_config_available(
        self,
        available: bool,
        *,
        mode: str | None = None,
    ) -> None:
        if not available:
            for button in (
                self.player_button,
                self.replay_button,
                self.data_button,
            ):
                if button is not None:
                    button.setDisabled(True)
            return

        self.refresh_availability()
        replay_available, data_available = False, False
        if self._runtime_readiness_provider is not None:
            replay_available, data_available = self._runtime_readiness_provider(mode)
        replay_available = bool(replay_available and mode != "IMPORT_AUDIO")
        data_available = bool(data_available)

        if self.replay_button is not None:
            self.replay_button.setEnabled(replay_available)
        if self.data_button is not None:
            self.data_button.setEnabled(data_available)

    def set_data_enabled(self, enabled: bool) -> None:
        if self.data_button is not None:
            self.data_button.setEnabled(bool(enabled))

    def reset_runtime_action_buttons(self) -> None:
        if self.replay_button is not None:
            self.replay_button.setDisabled(True)
        if self.data_button is not None:
            self.data_button.setDisabled(True)

    def capture_action_availability(
        self,
    ) -> tuple[bool | None, bool | None, bool | None]:
        return tuple(
            button.isEnabled() if button is not None else None
            for button in (
                self.player_button,
                self.replay_button,
                self.data_button,
            )
        )

    def restore_action_availability(
        self,
        state: tuple[bool | None, bool | None, bool | None],
    ) -> None:
        for button, enabled in zip(
            (
                self.player_button,
                self.replay_button,
                self.data_button,
            ),
            state,
        ):
            if button is not None and enabled is not None:
                button.setEnabled(enabled)

    def capture_configuration_presentation_state(self) -> tuple[bool, bool]:
        return (
            self._missing_config_prompt_enabled,
            self._missing_config_prompted,
        )

    def restore_configuration_presentation_state(
        self, state: tuple[bool, bool]
    ) -> None:
        self._missing_config_prompt_enabled = bool(state[0])
        self._missing_config_prompted = bool(state[1])

    @staticmethod
    def _has_focus(widget: Any) -> bool | None:
        has_focus = getattr(widget, "hasFocus", None)
        return bool(has_focus()) if callable(has_focus) else None

    def capture_configuration_state(
        self, *, selection_path: Any = _CURRENT_SELECTION
    ) -> ConfigurationViewState:
        if selection_path is _CURRENT_SELECTION and self.combobox is not None:
            current_data = getattr(self.combobox, "currentData", None)
            if callable(current_data):
                selection_path = current_data()
        if selection_path is _CURRENT_SELECTION:
            selection_path = None
        configuration_entries = None
        if self.combobox is not None:
            count = getattr(self.combobox, "count", None)
            item_text = getattr(self.combobox, "itemText", None)
            item_data = getattr(self.combobox, "itemData", None)
            if callable(count) and callable(item_text) and callable(item_data):
                configuration_entries = tuple(
                    (str(item_text(index)), item_data(index))
                    for index in range(count())
                )
            elif hasattr(self.combobox, "items"):
                configuration_entries = tuple(self.combobox.items)
        return ConfigurationViewState(
            action_availability=self.capture_action_availability(),
            presentation_state=self.capture_configuration_presentation_state(),
            selection_path=selection_path,
            configuration_entries=configuration_entries,
            combobox_had_focus=self._has_focus(self.combobox),
            serial_input_had_focus=self._has_focus(self.serial_input),
            parent_had_focus=self._has_focus(self.parent),
        )

    def restore_configuration_state(self, state: ConfigurationViewState) -> None:
        self.restore_configuration_presentation_state(state.presentation_state)
        if state.configuration_entries is not None:
            self.populate_configuration_entries(
                state.configuration_entries,
                using_config_path=state.selection_path,
                clear_first=True,
            )
        else:
            self.restore_selection(state.selection_path)
        self.restore_action_availability(state.action_availability)
        for widget, captured_focus in (
            (self.combobox, state.combobox_had_focus),
            (self.serial_input, state.serial_input_had_focus),
            (self.parent, state.parent_had_focus),
        ):
            clear_focus = getattr(widget, "clearFocus", None)
            if captured_focus is not None and callable(clear_focus):
                clear_focus()
        if state.combobox_had_focus and self.combobox is not None:
            self.combobox.setFocus()
        elif state.serial_input_had_focus and self.serial_input is not None:
            self.serial_input.setFocus()
        elif state.parent_had_focus and self.parent is not None:
            self.parent.setFocus()

    def mark_configuration_loaded(self) -> None:
        self._missing_config_prompted = False

    def present_missing_configuration_prompt(
        self, sequence_config: Any, *, eligible: bool
    ) -> bool:
        if eligible:
            self._missing_config_prompt_enabled = True
        if (
            sequence_config
            or not self._missing_config_prompt_enabled
            or self._missing_config_prompted
        ):
            return False
        self.warn("提示", MISSING_CONFIGURATION_MESSAGE)
        self._missing_config_prompted = True
        return True
