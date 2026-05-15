from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from PyQt5.QtCore import QObject, QPoint
from PyQt5.QtWidgets import QWidget

from ui.custom_ui_widget.widgets import Action, Menu
from ui.sequence.channel_plot_workspace_model import ChannelPlotWorkspaceModel


class ChannelPlotWorkspaceController(QObject):
    def __init__(self, model: ChannelPlotWorkspaceModel, menu_parent: Optional[QWidget] = None):
        super().__init__(menu_parent)
        self._model = model
        self._menu_parent = menu_parent
        self._menu: Optional[Menu] = None
        self._channel_actions: Dict[int, Action] = {}
        self._menu_sync_guard = False

        self._model.channels_reset.connect(self._rebuild_menu)
        self._model.visibility_changed.connect(self._sync_channel_action)

        if self._menu_parent is not None:
            self._rebuild_menu()

    def set_menu_parent(self, menu_parent: QWidget) -> None:
        self._menu_parent = menu_parent
        if self._menu is not None:
            self._menu.setParent(menu_parent)
        self._rebuild_menu()

    def menu(self) -> Menu:
        if self._menu is None:
            self._menu = Menu(self._menu_parent)
        return self._menu

    def channel_actions(self) -> List[Action]:
        return [self._channel_actions[channel_index] for channel_index in self._model.channel_indices()]

    def action_for_channel(self, channel_index: int) -> Optional[Action]:
        return self._channel_actions.get(int(channel_index))

    def show_context_menu(self, global_pos: QPoint) -> None:
        if not self._model.channel_indices():
            return
        self._sync_all_actions()
        self.menu().exec_(global_pos)

    def reset_channels(self, channel_indices: Optional[Iterable[int]]) -> None:
        self._model.reset_channels(channel_indices)

    def set_channel_visible(self, channel_index: int, visible: bool) -> bool:
        return self._model.set_visible(channel_index, visible)

    def toggle_channel_visible(self, channel_index: int) -> bool:
        next_visible = not self._model.is_visible(channel_index)
        self._model.set_visible(channel_index, next_visible)
        return next_visible

    def hide_channel(self, channel_index: int) -> bool:
        return self.set_channel_visible(channel_index, False)

    def show_all_channels(self) -> bool:
        changed = False
        for channel_index in self._model.channel_indices():
            changed = self._model.set_visible(channel_index, True) or changed
        return changed

    def _rebuild_menu(self) -> None:
        if self._menu is None:
            self._menu = Menu(self._menu_parent)
        menu = self._menu
        menu.clear()
        self._channel_actions.clear()

        for channel_index in self._model.channel_indices():
            action = Action(self._channel_text(channel_index), menu)
            action.setCheckable(True)
            action.toggled.connect(
                lambda checked, current_channel=channel_index: self._on_channel_action_toggled(
                    current_channel, checked
                )
            )
            menu.addAction(action)
            self._channel_actions[channel_index] = action

        self._sync_all_actions()

    def _sync_all_actions(self) -> None:
        self._menu_sync_guard = True
        try:
            for channel_index, action in self._channel_actions.items():
                visible = self._model.is_visible(channel_index)
                if action.isChecked() != visible:
                    action.setChecked(visible)
        finally:
            self._menu_sync_guard = False

    def _sync_channel_action(self, channel_index: int, visible: bool) -> None:
        action = self._channel_actions.get(int(channel_index))
        if action is None:
            return

        self._menu_sync_guard = True
        try:
            if action.isChecked() != bool(visible):
                action.setChecked(bool(visible))
        finally:
            self._menu_sync_guard = False

    def _on_channel_action_toggled(self, channel_index: int, checked: bool) -> None:
        if self._menu_sync_guard:
            return
        self.set_channel_visible(channel_index, checked)

    @staticmethod
    def _channel_text(channel_index: int) -> str:
        return f"In{int(channel_index) + 1}"
