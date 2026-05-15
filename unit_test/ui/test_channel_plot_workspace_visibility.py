import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.channel_plot_workspace_model import ChannelPlotWorkspaceModel


class _FakeTimer:
    def __init__(self):
        self._active = False
        self.start_calls = []
        self.stop_calls = 0

    def isActive(self):
        return self._active

    def start(self, interval_ms):
        self._active = True
        self.start_calls.append(interval_ms)

    def stop(self):
        self._active = False
        self.stop_calls += 1


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _process_events(app):
    app.processEvents()


def test_reset_channels_tracks_order_and_defaults_to_visible():
    model = ChannelPlotWorkspaceModel()
    reset_events = []
    model.channels_reset.connect(lambda: reset_events.append(model.channel_indices()))

    model.reset_channels([3, 1, 2])

    assert model.channel_indices() == [3, 1, 2]
    assert model.visible_channels() == [3, 1, 2]
    assert model.hidden_channels() == []
    assert model.is_visible(3) is True
    assert model.is_visible(1) is True
    assert model.is_visible(2) is True
    assert reset_events == [[3, 1, 2]]


def test_set_visible_updates_lists_and_only_emits_on_state_change():
    model = ChannelPlotWorkspaceModel([1, 4, 7])
    visibility_events = []
    model.visibility_changed.connect(lambda channel_index, visible: visibility_events.append((channel_index, visible)))

    assert model.set_visible(4, False) is True
    assert model.visible_channels() == [1, 7]
    assert model.hidden_channels() == [4]
    assert model.is_visible(4) is False

    assert model.set_visible(4, False) is False
    assert visibility_events == [(4, False)]


def test_reset_channels_restores_hidden_channels_to_visible():
    model = ChannelPlotWorkspaceModel([0, 2, 5])
    reset_events = []
    model.channels_reset.connect(
        lambda: reset_events.append((model.channel_indices(), model.visible_channels()))
    )

    model.set_visible(2, False)

    model.reset_channels([0, 2, 5])

    assert model.channel_indices() == [0, 2, 5]
    assert model.visible_channels() == [0, 2, 5]
    assert model.hidden_channels() == []
    assert model.is_visible(2) is True
    assert reset_events == [([0, 2, 5], [0, 2, 5])]


def test_reset_channels_keeps_workspace_default_channel_for_empty_input():
    model = ChannelPlotWorkspaceModel()

    model.reset_channels([])

    assert model.channel_indices() == [0]
    assert model.visible_channels() == [0]
    assert model.hidden_channels() == []


def test_unknown_channel_raises_key_error():
    model = ChannelPlotWorkspaceModel([1, 2])

    with pytest.raises(KeyError):
        model.is_visible(3)

    with pytest.raises(KeyError):
        model.set_visible(3, False)


def test_single_canvas_mode_uses_one_hidden_title_full_area(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1, 2])
    _process_events(qapp)

    workspace.set_single_canvas_mode(2)
    _process_events(qapp)

    wins = workspace.subwindows()
    assert len(wins) == 1
    assert wins[0].channel_index == 2
    assert wins[0].title_bar.isHidden() is True
    assert wins[0].pos().x() == 0
    assert wins[0].pos().y() == 0
    assert wins[0].width() == workspace.canvas.width()
    assert wins[0].height() == workspace.canvas.height()

    workspace.close()
    _process_events(qapp)


def test_single_canvas_mode_before_show_fills_after_show(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.set_single_canvas_mode(1)
    workspace.show()
    _process_events(qapp)

    wins = workspace.subwindows()
    assert len(wins) == 1
    assert wins[0].channel_index == 1
    assert wins[0].title_bar.isHidden() is True
    assert wins[0].pos().x() == 0
    assert wins[0].pos().y() == 0
    assert wins[0].width() == workspace.canvas.width()
    assert wins[0].height() == workspace.canvas.height()

    workspace.close()
    _process_events(qapp)


def test_single_canvas_mode_tracks_resize(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_single_canvas_mode(1)
    _process_events(qapp)

    win = workspace.subwindows()[0]
    workspace.resize(720, 520)
    _process_events(qapp)

    assert win.title_bar.isHidden() is True
    assert win.pos().x() == 0
    assert win.pos().y() == 0
    assert win.width() == workspace.canvas.width()
    assert win.height() == workspace.canvas.height()

    workspace.close()
    _process_events(qapp)


def test_single_canvas_mode_suppresses_context_menu_and_hide_paths(qapp, monkeypatch):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_single_canvas_mode(0)
    _process_events(qapp)

    calls = []
    monkeypatch.setattr(workspace.controller(), "show_context_menu", lambda global_pos: calls.append(global_pos))
    workspace._show_context_menu_from(workspace.canvas, workspace.canvas.rect().center())
    assert calls == []

    win = workspace.subwindows()[0]
    workspace.controller().hide_channel(0)
    _process_events(qapp)
    assert workspace.model().visible_channels() == [0]
    assert workspace.subwindows() == [win]
    assert win.isVisible() is True

    win.hide_requested.emit(0)
    _process_events(qapp)
    assert workspace.model().visible_channels() == [0]
    assert workspace.subwindows() == [win]
    assert win.isVisible() is True

    action = workspace.controller().action_for_channel(0)
    if action is not None:
        action.trigger()
        _process_events(qapp)
        assert workspace.model().visible_channels() == [0]
        assert win.isVisible() is True

    assert workspace.model().set_visible(0, False) is True
    _process_events(qapp)
    assert workspace.model().visible_channels() == [0]
    assert win.isVisible() is True

    workspace.close()
    _process_events(qapp)


def test_set_channels_exits_single_canvas_mode_and_restores_titles(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_single_canvas_mode(1)
    _process_events(qapp)

    assert len(workspace.subwindows()) == 1
    assert workspace.subwindows()[0].title_bar.isHidden() is True

    workspace.set_channels([0, 1])
    _process_events(qapp)

    wins = workspace.subwindows()
    assert [w.channel_index for w in wins] == [0, 1]
    assert all(w.title_bar.isVisible() for w in wins)
    assert wins[0].geometry().intersects(wins[1].geometry()) is False

    workspace.close()
    _process_events(qapp)


def test_close_button_hides_channel_and_unchecks_controller_menu(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1, 2])
    _process_events(qapp)

    channel_two_window = workspace.subwindow(1)
    assert channel_two_window is not None
    assert workspace.model().visible_channels() == [0, 1, 2]
    assert workspace.controller().action_for_channel(1).isChecked() is True

    channel_two_window.title_bar.close_btn.click()
    _process_events(qapp)

    assert workspace.model().visible_channels() == [0, 2]
    assert workspace.model().hidden_channels() == [1]
    assert workspace.controller().action_for_channel(1).isChecked() is False
    assert channel_two_window.isHidden() is True

    workspace.close()
    _process_events(qapp)


def test_controller_menu_keeps_full_channel_list_and_can_restore_hidden_channel(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1, 2, 3])
    _process_events(qapp)

    workspace.subwindow(2).title_bar.close_btn.click()
    _process_events(qapp)

    actions = workspace.controller().channel_actions()
    assert [action.text() for action in actions] == ["In1", "In2", "In3", "In4"]
    assert [action.isChecked() for action in actions] == [True, True, False, True]

    workspace.controller().action_for_channel(2).trigger()
    _process_events(qapp)

    assert workspace.model().visible_channels() == [0, 1, 2, 3]
    assert workspace.controller().action_for_channel(2).isChecked() is True
    assert workspace.subwindow(2).isVisible() is True

    workspace.close()
    _process_events(qapp)


def test_hiding_window_keeps_siblings_in_place_and_freed_area_can_be_reused(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1, 2])
    _process_events(qapp)

    channel_one_window = workspace.subwindow(0)
    channel_two_window = workspace.subwindow(1)
    channel_three_window = workspace.subwindow(2)
    assert channel_one_window is not None
    assert channel_two_window is not None
    assert channel_three_window is not None

    sibling_positions = {
        0: channel_one_window.pos(),
        2: channel_three_window.pos(),
    }
    hidden_geometry = channel_two_window.geometry()

    assert channel_three_window.request_move(channel_two_window.pos()) is False

    channel_two_window.title_bar.close_btn.click()
    _process_events(qapp)

    assert channel_one_window.pos() == sibling_positions[0]
    assert channel_three_window.pos() == sibling_positions[2]
    assert channel_two_window.isHidden() is True

    workspace.controller().action_for_channel(1).trigger()
    _process_events(qapp)

    assert channel_one_window.pos() == sibling_positions[0]
    assert channel_three_window.pos() == sibling_positions[2]
    assert channel_two_window.isVisible() is True
    assert channel_two_window.geometry() == hidden_geometry

    channel_two_window.title_bar.close_btn.click()
    _process_events(qapp)

    assert channel_three_window.request_move(hidden_geometry.topLeft()) is True
    assert channel_three_window.pos() == hidden_geometry.topLeft()

    workspace.close()
    _process_events(qapp)


def test_showing_hidden_window_after_resize_uses_current_tile_size_and_safe_restore(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1, 2])
    _process_events(qapp)

    channel_one_window = workspace.subwindow(0)
    channel_two_window = workspace.subwindow(1)
    channel_three_window = workspace.subwindow(2)
    assert channel_one_window is not None
    assert channel_two_window is not None
    assert channel_three_window is not None

    channel_two_window.title_bar.close_btn.click()
    _process_events(qapp)
    assert channel_two_window.isHidden() is True

    workspace.resize(720, 520)
    _process_events(qapp)

    sibling_positions_after_resize = {
        0: channel_one_window.pos(),
        2: channel_three_window.pos(),
    }
    viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    expected_spec = workspace._calc_tile_spec(viewport_w, viewport_h, 3)

    workspace.controller().action_for_channel(1).trigger()
    _process_events(qapp)

    assert channel_one_window.pos() == sibling_positions_after_resize[0]
    assert channel_three_window.pos() == sibling_positions_after_resize[2]
    assert channel_two_window.isVisible() is True
    assert channel_two_window.width() == expected_spec.win_w
    assert channel_two_window.height() == expected_spec.win_h
    assert workspace.canvas.rect().contains(channel_two_window.geometry())
    assert channel_two_window.geometry().intersects(channel_one_window.geometry()) is False
    assert channel_two_window.geometry().intersects(channel_three_window.geometry()) is False

    workspace.close()
    _process_events(qapp)


def test_showing_hidden_window_in_narrow_workspace_clamps_tile_width_and_restores_in_bounds(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0, 1])
    _process_events(qapp)

    channel_one_window = workspace.subwindow(0)
    channel_two_window = workspace.subwindow(1)
    assert channel_one_window is not None
    assert channel_two_window is not None

    channel_two_window.title_bar.close_btn.click()
    _process_events(qapp)
    assert channel_two_window.isHidden() is True

    workspace.resize(200, 520)
    _process_events(qapp)

    viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    expected_spec = workspace._calc_tile_spec(viewport_w, viewport_h, 2)
    usable_w = max(1, viewport_w - 2 * expected_spec.pad)

    assert viewport_w >= 50
    assert usable_w < 220
    assert expected_spec.win_w == usable_w

    workspace.controller().action_for_channel(1).trigger()
    _process_events(qapp)

    assert channel_two_window.isVisible() is True
    assert channel_two_window.width() == expected_spec.win_w
    assert channel_two_window.height() == expected_spec.win_h
    assert channel_two_window.geometry().right() <= workspace.canvas.rect().right()
    assert workspace.canvas.rect().contains(channel_two_window.geometry())
    assert channel_two_window.geometry().intersects(channel_one_window.geometry()) is False

    workspace.close()
    _process_events(qapp)


def test_widening_across_column_threshold_reflows_windows_side_by_side(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(600, 640)
    workspace.show()
    workspace.set_channels([0, 1])
    _process_events(qapp)

    channel_one_window = workspace.subwindow(0)
    channel_two_window = workspace.subwindow(1)
    assert channel_one_window is not None
    assert channel_two_window is not None

    narrow_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    narrow_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    narrow_spec = workspace._calc_tile_spec(narrow_viewport_w, narrow_viewport_h, 2)

    assert narrow_spec.cols == 1
    assert channel_one_window.x() == channel_two_window.x()
    assert channel_two_window.y() > channel_one_window.y()

    workspace.resize(960, 640)
    _process_events(qapp)

    wide_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    wide_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    wide_spec = workspace._calc_tile_spec(wide_viewport_w, wide_viewport_h, 2)

    assert wide_spec.cols == 2
    assert channel_one_window.x() == wide_spec.pad
    assert channel_one_window.y() == wide_spec.pad
    assert channel_two_window.x() == wide_spec.pad + wide_spec.win_w + wide_spec.gap
    assert channel_two_window.y() == wide_spec.pad
    assert channel_one_window.geometry().intersects(channel_two_window.geometry()) is False

    workspace.close()
    _process_events(qapp)


def test_taller_resize_across_row_threshold_reflows_stacked_windows(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(600, 440)
    workspace.show()
    workspace.set_channels([0, 1])
    _process_events(qapp)

    channel_one_window = workspace.subwindow(0)
    channel_two_window = workspace.subwindow(1)
    assert channel_one_window is not None
    assert channel_two_window is not None

    short_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    short_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    short_spec = workspace._calc_tile_spec(short_viewport_w, short_viewport_h, 2)

    assert short_spec.cols == 1
    assert short_spec.height_rows == 1
    assert channel_one_window.x() == channel_two_window.x()
    assert channel_one_window.y() == short_spec.pad
    assert channel_two_window.y() == short_spec.pad + short_spec.win_h + short_spec.gap

    previous_second_y = channel_two_window.y()

    workspace.resize(600, 560)
    _process_events(qapp)

    tall_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    tall_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    tall_spec = workspace._calc_tile_spec(tall_viewport_w, tall_viewport_h, 2)

    assert tall_spec.cols == 1
    assert tall_spec.height_rows == 2
    assert channel_one_window.y() == tall_spec.pad
    assert channel_two_window.y() == tall_spec.pad + tall_spec.win_h + tall_spec.gap
    assert channel_two_window.y() < previous_second_y
    assert channel_one_window.geometry().intersects(channel_two_window.geometry()) is False

    workspace.close()
    _process_events(qapp)


def test_four_windows_only_fill_viewport_height_after_two_row_threshold(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 440)
    workspace.show()
    workspace.set_channels([0, 1, 2, 3])
    _process_events(qapp)

    windows = [workspace.subwindow(channel_index) for channel_index in range(4)]
    assert all(window is not None for window in windows)
    channel_one_window, channel_two_window, channel_three_window, channel_four_window = windows

    short_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    short_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    short_spec = workspace._calc_tile_spec(short_viewport_w, short_viewport_h, 4)
    short_usable_h = max(1, short_viewport_h - 2 * short_spec.pad)
    short_fill_h = int((short_usable_h - short_spec.gap) / 2)

    assert short_spec.cols == 2
    assert short_spec.height_rows == 1
    assert short_spec.win_h == max(220, short_usable_h)
    assert short_spec.win_h > short_fill_h
    assert channel_one_window.height() == short_spec.win_h
    assert channel_two_window.height() == short_spec.win_h
    assert channel_three_window.height() == short_spec.win_h
    assert channel_four_window.height() == short_spec.win_h
    assert channel_one_window.y() == short_spec.pad
    assert channel_two_window.y() == short_spec.pad
    assert channel_three_window.y() == short_spec.pad + short_spec.win_h + short_spec.gap
    assert channel_four_window.y() == short_spec.pad + short_spec.win_h + short_spec.gap
    assert channel_three_window.y() > short_viewport_h
    assert workspace.canvas.minimumHeight() > short_viewport_h

    short_second_row_y = channel_three_window.y()

    workspace.resize(960, 640)
    _process_events(qapp)

    tall_viewport_w = int(workspace.scroll.viewport().width() or workspace.width() or 0)
    tall_viewport_h = int(workspace.scroll.viewport().height() or workspace.height() or 0)
    tall_spec = workspace._calc_tile_spec(tall_viewport_w, tall_viewport_h, 4)
    tall_usable_h = max(1, tall_viewport_h - 2 * tall_spec.pad)
    tall_fill_h = int((tall_usable_h - tall_spec.gap) / 2)

    assert tall_spec.cols == 2
    assert tall_spec.height_rows == 2
    assert tall_spec.win_h == tall_fill_h
    assert channel_one_window.height() == tall_spec.win_h
    assert channel_two_window.height() == tall_spec.win_h
    assert channel_three_window.height() == tall_spec.win_h
    assert channel_four_window.height() == tall_spec.win_h
    assert channel_one_window.y() == tall_spec.pad
    assert channel_two_window.y() == tall_spec.pad
    assert channel_three_window.y() == tall_spec.pad + tall_spec.win_h + tall_spec.gap
    assert channel_four_window.y() == tall_spec.pad + tall_spec.win_h + tall_spec.gap
    assert channel_three_window.y() < short_second_row_y
    assert workspace.canvas.minimumHeight() == tall_viewport_h

    workspace.close()
    _process_events(qapp)


def test_deferred_retry_scheduling_is_coalesced_with_positive_delay(qapp, monkeypatch):
    workspace = ChannelPlotWorkspace()
    tile_timer = _FakeTimer()
    restore_timer = _FakeTimer()
    workspace._tile_retry_timer = tile_timer
    workspace._restore_retry_timer = restore_timer
    monkeypatch.setattr(workspace, "_restore_layout_is_ready", lambda: True)

    workspace._schedule_tile_retry(keep_positions=True)
    workspace._schedule_tile_retry(keep_positions=False)
    workspace._schedule_tile_retry(keep_positions=True)

    assert len(tile_timer.start_calls) == 1
    assert tile_timer.start_calls[0] > 0
    assert workspace._tile_retry_keep_positions is False

    workspace._schedule_restore_retry(2)
    workspace._schedule_restore_retry(2)
    workspace._schedule_restore_retry(1)

    assert len(restore_timer.start_calls) == 1
    assert restore_timer.start_calls[0] > 0
    assert workspace._restore_retry_channels == {1, 2}

    workspace.close()
    _process_events(qapp)


def test_restore_retry_waits_for_ready_layout_before_starting_timer(qapp, monkeypatch):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    restore_timer = _FakeTimer()
    workspace._restore_retry_timer = restore_timer
    workspace.set_channels([0, 1])
    _process_events(qapp)

    channel_two_window = workspace.subwindow(1)
    assert channel_two_window is not None

    workspace.controller().hide_channel(1)
    _process_events(qapp)
    assert channel_two_window.isHidden() is True

    ready = False
    monkeypatch.setattr(workspace, "_restore_layout_is_ready", lambda: ready)
    monkeypatch.setattr(workspace, "_restore_subwindow_position", lambda subwindow: ready)

    workspace.controller().action_for_channel(1).trigger()
    _process_events(qapp)

    assert workspace._restore_retry_channels == {1}
    assert restore_timer.start_calls == []
    assert channel_two_window.isHidden() is True

    workspace._run_restore_retries()
    assert workspace._restore_retry_channels == {1}
    assert restore_timer.start_calls == []

    ready = True
    workspace.resize(900, 600)
    _process_events(qapp)

    assert len(restore_timer.start_calls) == 1
    assert restore_timer.start_calls[0] > 0

    restore_timer._active = False
    workspace._run_restore_retries()

    assert workspace._restore_retry_channels == set()
    assert channel_two_window.isVisible() is True

    workspace.close()
    _process_events(qapp)


def test_reset_while_restore_retry_pending_clears_stale_retry_and_ignores_removed_channel(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    restore_timer = _FakeTimer()
    workspace._restore_retry_timer = restore_timer
    workspace.set_channels([0, 1])
    _process_events(qapp)

    workspace._schedule_restore_retry(1)

    assert workspace._restore_retry_channels == {1}
    assert restore_timer.isActive() is True

    workspace.set_channels([0])
    _process_events(qapp)

    assert workspace._restore_retry_channels == set()
    assert restore_timer.isActive() is False
    assert restore_timer.stop_calls == 1

    workspace._restore_retry_channels = {1}
    workspace._run_restore_retries()
    assert workspace._restore_retry_channels == set()

    workspace.close()
    _process_events(qapp)


def test_plot_surface_keeps_pyqtgraph_menu_enabled(qapp):
    workspace = ChannelPlotWorkspace()
    workspace.resize(960, 640)
    workspace.show()
    workspace.set_channels([0])
    _process_events(qapp)

    subwindow = workspace.subwindow(0)
    assert subwindow is not None
    assert subwindow.plot_widget.menuEnabled() is True
    assert subwindow.plot_widget.contextMenuPolicy() != Qt.CustomContextMenu

    calls = []
    workspace.controller().show_context_menu = lambda global_pos: calls.append(global_pos)
    workspace._show_context_menu_from(workspace.canvas, subwindow.geometry().center())
    assert calls == []

    workspace.close()
    _process_events(qapp)
