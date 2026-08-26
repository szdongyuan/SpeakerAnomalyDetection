import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QSizePolicy, QSplitter, QVBoxLayout, QWidget

import ui.sequence.sequencement_count_board as count_board_module
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequencement_count_board import SequenceCountBoard


class _SequenceWindowHarness(SequenceWindow):
    def __init__(self, count_board):
        QWidget.__init__(self)
        self.count_board = count_board
        self.toolsbar = QWidget()

    def add_file_to_using_file_combobox(self):
        pass

    def showEvent(self, event):
        QWidget.showEvent(self, event)

    def closeEvent(self, event):
        QWidget.closeEvent(self, event)


def _build_sequence_window_harness(count_board, size=(900, 500)):
    window = _SequenceWindowHarness(count_board)
    layout = SequenceWindow.create_waveform_layout(window)
    window.setLayout(layout)
    window.resize(*size)
    window.show()
    return window, layout


def _scaled_px(value):
    return count_board_module.ui_style_const.scale_size_px(value)


def _splitter_available_width(window):
    sizes = window.waveform_splitter.sizes()
    if sizes:
        return sum(sizes)
    return max(0, window.waveform_splitter.width() - window.waveform_splitter.handleWidth())


def _vertical_gap(parent, upper_widget, lower_widget):
    upper_top = upper_widget.mapTo(parent, upper_widget.rect().topLeft()).y()
    lower_top = lower_widget.mapTo(parent, lower_widget.rect().topLeft()).y()
    return lower_top - (upper_top + upper_widget.height())


def _mode_row_layout(board):
    return board.content_widget.layout().itemAt(0).layout()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def count_board_runtime(tmp_path):
    return tmp_path


def test_count_board_collapse_toggle_is_idempotent(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()
    emissions = []
    board.collapsed_changed.connect(emissions.append)

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.collapse_toggle_btn.isVisible() is True

    board.set_collapsed(True)

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert board.collapse_toggle_btn.isVisible() is True
    assert emissions == [True]

    board.set_collapsed(True)

    assert emissions == [True]

    board.toggle_collapsed()

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert emissions == [True, False]

    board.close()


def test_count_board_collapse_toggle_button_background_is_transparent(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()

    button_style = board.collapse_toggle_btn.styleSheet()

    assert "background-color: transparent" in button_style
    assert "border: none" in button_style

    board.close()


def test_count_board_collapse_preserves_mode_and_count_values(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()
    board.on_test_btn_clicked()
    board.total_line_edit.setText("12")
    board.ok_line_edit.setText("10")
    board.ng_line_edit.setText("2")

    board.set_collapsed(True)
    board.set_collapsed(False)

    assert board.mode == "test"
    assert board.stacked_widget.currentIndex() == 0
    assert board.total_line_edit.text() == "12"
    assert board.ok_line_edit.text() == "10"
    assert board.ng_line_edit.text() == "2"

    board.close()


def test_count_board_expanded_minimum_tracks_button_width_and_splitter_gap(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert board._minimum_content_width == board.btn_width + count_board_module.ui_style_const.scale_size_px(20)
    assert board.content_widget.minimumSizeHint().width() == board._minimum_content_width
    assert board.lineedit_width == board._minimum_content_width - board.label_width
    assert board.expanded_width_hint() == (
        board._minimum_content_width + board._collapse_bar_width + board._splitter_gap_width
    )
    assert board.collapsed_width_hint() == board._collapse_bar_width + board._splitter_gap_width
    assert board.layout().contentsMargins().right() == board._splitter_gap_width

    board.close()


def test_count_board_minimum_width_is_enforced_by_widget_geometry(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.resize(board.expanded_width_hint(), 420)
    board.show()
    qapp.processEvents()

    expected_content_width = board.btn_width + _scaled_px(20)
    expected_lineedit_width = expected_content_width - board.label_width
    expected_board_width = expected_content_width + board._collapse_bar_width + _scaled_px(5)

    assert board.width() >= expected_board_width
    assert board.minimumWidth() == expected_board_width
    assert board.content_widget.width() >= expected_content_width
    assert board.mark_total_edit.minimumWidth() == expected_lineedit_width
    assert board.mark_total_edit.width() >= expected_lineedit_width

    board.close()


def test_count_board_starts_with_expanded_safe_minimum_until_compact_resize_enabled(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()

    expanded_width = board.expanded_width_hint()
    collapsed_width = board.collapsed_width_hint()

    assert board.minimumSizeHint().width() >= expanded_width
    assert board.minimumWidth() >= expanded_width

    board.set_compact_resize_enabled(True)

    assert board.minimumSizeHint().width() == collapsed_width
    assert board.minimumWidth() == collapsed_width

    board.close()


def test_count_board_line_edits_expand_horizontally_but_keep_fixed_height(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.resize(board.expanded_width_hint() + 160, 500)
    board.show()
    qapp.processEvents()

    assert board.mark_total_edit.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert board.mark_total_edit.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
    assert board.mark_total_edit.height() == board.lineedit_height
    assert board.mark_total_edit.minimumWidth() == board.lineedit_width
    assert board.mark_total_edit.width() > board.lineedit_width

    board.close()


def test_count_board_vertical_layout_stays_compact_when_tall(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.resize(board.expanded_width_hint() + 160, 700)
    board.show()
    qapp.processEvents()

    test_layout = board.stacked_widget.widget(0).layout()
    mark_layout = board.stacked_widget.widget(1).layout()
    total_top = board.mark_total_edit.mapTo(board, board.mark_total_edit.rect().topLeft()).y()
    ng_top = board.ng_btn.mapTo(board, board.ng_btn.rect().topLeft()).y()

    assert all(test_layout.itemAt(index).spacerItem() is None for index in range(test_layout.count() - 1))
    assert test_layout.itemAt(test_layout.count() - 1).spacerItem() is not None
    assert mark_layout.itemAt(mark_layout.count() - 1).spacerItem() is not None
    assert test_layout.spacing() == count_board_module.ui_style_const.scale_size_px(7)
    assert mark_layout.spacing() == count_board_module.ui_style_const.scale_size_px(7)
    assert total_top < 90
    assert ng_top < 320

    board.close()


def test_count_board_line_edit_row_gaps_match_scaled_default_in_mark_and_test_modes(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.resize(board.expanded_width_hint() + 160, 700)
    board.show()
    qapp.processEvents()

    expected_gap = _scaled_px(7)

    board.on_mark_btn_clicked()
    qapp.processEvents()
    assert [
        _vertical_gap(board, board.mark_total_edit, board.mark_ok_edit),
        _vertical_gap(board, board.mark_ok_edit, board.mark_ng_edit),
    ] == [expected_gap, expected_gap]

    board.on_test_btn_clicked()
    qapp.processEvents()
    assert [
        _vertical_gap(board, board.total_line_edit, board.ok_line_edit),
        _vertical_gap(board, board.ok_line_edit, board.ng_line_edit),
    ] == [expected_gap, expected_gap]

    board.close()


def test_mode_buttons_fit_within_minimum_content_width(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.show()
    qapp.processEvents()

    mode_row_width = board.mode_label_width + board.test_btn.width() + board.mark_btn.width()

    assert board.test_btn.width() == board.mode_button_width
    assert board.mark_btn.width() == board.mode_button_width
    assert mode_row_width <= board._minimum_content_width

    board.close()


def test_mode_row_keeps_label_left_and_buttons_right_aligned(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    board.resize(board.expanded_width_hint() + 160, 420)
    board.show()
    qapp.processEvents()

    mode_layout = _mode_row_layout(board)
    mode_label = mode_layout.itemAt(0).widget()
    content_rect = board.content_widget.contentsRect()

    assert mode_label.geometry().left() == content_rect.left()
    assert board.mark_btn.geometry().left() == board.test_btn.geometry().right() + 1
    assert abs(board.mark_btn.geometry().right() - content_rect.right()) <= 1
    assert board.test_btn.geometry().left() > mode_label.geometry().right()

    board.close()


def test_sequence_waveform_layout_uses_horizontal_splitter(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, layout = _build_sequence_window_harness(board)
    qapp.processEvents()

    assert isinstance(layout, QHBoxLayout)
    assert isinstance(window.waveform_splitter, QSplitter)
    assert window.waveform_splitter.orientation() == Qt.Horizontal
    assert "#b8b8b8" in window.waveform_splitter.styleSheet()
    assert f"margin-left: {_scaled_px(3)}px" in window.waveform_splitter.styleSheet()
    assert f"margin-right: {_scaled_px(3)}px" in window.waveform_splitter.styleSheet()
    assert window.waveform_splitter.widget(0) is board
    assert isinstance(window.waveform_splitter.widget(1), ChannelPlotWorkspace)
    assert window.channel_workspace is window.waveform_splitter.widget(1)

    handle = window.waveform_splitter.handle(1)
    handle_rect = handle.geometry()
    image = window.waveform_splitter.grab().toImage()
    layout_gap_left = window.waveform_splitter.widget(0).geometry().right() + 1
    layout_gap_right = window.waveform_splitter.widget(1).geometry().left() - 1
    layout_gap_width = layout_gap_right - layout_gap_left + 1
    scan_y = handle_rect.center().y()
    gray_offsets = [
        offset
        for offset in range(layout_gap_width)
        if image.pixelColor(layout_gap_left + offset, scan_y).name() == "#b8b8b8"
    ]
    assert gray_offsets
    assert gray_offsets == list(range(gray_offsets[0], gray_offsets[-1] + 1))

    left_empty_width = gray_offsets[0]
    gray_width = len(gray_offsets)
    right_empty_width = layout_gap_width - gray_offsets[-1] - 1
    side_padding = _scaled_px(3)
    assert (
        layout_gap_width,
        left_empty_width,
        gray_width,
        right_empty_width,
    ) == (
        side_padding * 2 + 1,
        side_padding,
        1,
        side_padding,
    )
    assert window.waveform_splitter.handleWidth() == 1

    window.close()


def test_sequence_layout_keeps_toolbar_tight_to_top(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window = _SequenceWindowHarness(board)
    SequenceWindow.init_ui(window)
    window.resize(900, 500)
    window.show()
    qapp.processEvents()

    layout = window.layout()
    assert isinstance(layout, QVBoxLayout)
    assert layout.alignment() & Qt.AlignTop
    assert layout.spacing() == 0
    assert window.toolsbar.geometry().top() <= 1

    window.close()


def test_initial_splitter_size_preserves_expanded_count_board_width(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()
    qapp.processEvents()

    expanded_width = board.expanded_width_hint()
    assert window.waveform_splitter.width() > expanded_width + 1
    assert window.waveform_splitter.sizes()[0] >= expanded_width
    assert window.waveform_splitter.sizes()[0] <= window._count_board_max_width

    window.close()


def test_constrained_initial_splitter_layout_collapses_count_board(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board, size=(120, 220))
    qapp.processEvents()
    qapp.processEvents()

    assert _splitter_available_width(window) < board.expanded_width_hint() + 1
    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert window.waveform_splitter.sizes()[0] <= window._count_board_collapse_threshold

    window.close()


def test_count_board_splitter_left_side_is_capped(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()

    max_width = window._count_board_max_width
    window._last_count_board_expanded_width = max_width + 200
    window._apply_count_board_splitter_sizes(False)
    qapp.processEvents()

    assert board.maximumWidth() == max_width
    assert window.waveform_splitter.sizes()[0] <= max_width
    assert max_width > board.expanded_width_hint()

    window.close()


def test_constrained_programmatic_expansion_keeps_content_hidden(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board, size=(120, 220))
    qapp.processEvents()
    qapp.processEvents()

    board.set_collapsed(True)
    qapp.processEvents()
    assert _splitter_available_width(window) < board.expanded_width_hint() + 1

    board.set_collapsed(False)
    qapp.processEvents()

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert board.content_widget.width() < board._minimum_content_width

    window.resize(900, 500)
    window.waveform_splitter.setSizes([board.expanded_width_hint(), 600])
    qapp.processEvents()

    board.set_collapsed(False)
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.content_widget.width() >= board._minimum_content_width

    window.close()


def test_constrained_toggle_expansion_keeps_content_hidden(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board, size=(120, 220))
    qapp.processEvents()
    qapp.processEvents()

    board.set_collapsed(True)
    qapp.processEvents()
    assert _splitter_available_width(window) < board.expanded_width_hint() + 1

    board.collapse_toggle_btn.click()
    qapp.processEvents()

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert board.content_widget.width() < board._minimum_content_width

    window.resize(900, 500)
    window.waveform_splitter.setSizes([board.expanded_width_hint(), 600])
    qapp.processEvents()

    board.collapse_toggle_btn.click()
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.content_widget.width() >= board._minimum_content_width

    window.close()


def test_splitter_drag_collapses_and_expands_count_board(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()

    window.waveform_splitter.setSizes([board.expanded_width_hint(), 600])
    window.waveform_splitter.moveSplitter(window._count_board_collapsed_width, 1)
    qapp.processEvents()

    assert board.is_collapsed() is True
    assert window.waveform_splitter.sizes()[0] <= window._count_board_collapse_threshold

    expand_width = board.expanded_width_hint() + 1
    assert expand_width <= window._count_board_max_width
    window.waveform_splitter.moveSplitter(expand_width, 1)
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert window._last_count_board_expanded_width >= expand_width

    window.close()


def test_splitter_drag_to_intermediate_width_collapses_instead_of_clipping(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()

    intermediate_width = window._count_board_expand_threshold + _scaled_px(40)
    assert window._count_board_expand_threshold < intermediate_width < board.expanded_width_hint()

    window.waveform_splitter.setSizes([intermediate_width, 600])
    window._on_waveform_splitter_moved(intermediate_width, 1)

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert board.content_widget.width() < board._minimum_content_width

    window.close()


def test_direct_splitter_set_sizes_collapses_instead_of_clipping_count_board(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()
    qapp.processEvents()

    safe_expanded_width = board.expanded_width_hint() + _scaled_px(24)
    window.waveform_splitter.setSizes([safe_expanded_width, 600])
    qapp.processEvents()
    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.content_widget.width() >= board._minimum_content_width

    intermediate_width = window._count_board_expand_threshold + _scaled_px(40)
    assert window._count_board_expand_threshold < intermediate_width < board.expanded_width_hint()

    window.waveform_splitter.setSizes([intermediate_width, 600])
    qapp.processEvents()
    qapp.processEvents()

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert window.waveform_splitter.sizes()[0] <= window._count_board_collapse_threshold
    assert not (board.content_widget.isVisible() and board.content_widget.width() < board._minimum_content_width)

    window.close()


def test_parent_resize_narrower_collapses_expanded_count_board(qapp, count_board_runtime):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.content_widget.width() >= board._minimum_content_width

    window.resize(120, 220)
    qapp.processEvents()
    qapp.processEvents()

    assert board.is_collapsed() is True
    assert board.content_widget.isVisible() is False
    assert window.waveform_splitter.sizes()[0] <= window._count_board_collapse_threshold
    assert not (board.content_widget.isVisible() and board.content_widget.width() < board._minimum_content_width)

    window.resize(900, 500)
    qapp.processEvents()
    window.waveform_splitter.setSizes([board.expanded_width_hint(), 600])
    board.set_collapsed(False)
    qapp.processEvents()

    assert board.is_collapsed() is False
    assert board.content_widget.isVisible() is True
    assert board.content_widget.width() >= board._minimum_content_width

    window.close()


def test_splitter_resize_event_filter_is_quiet_with_lightweight_harness(qapp, count_board_runtime, capsys):
    board = SequenceCountBoard({})
    window, _layout = _build_sequence_window_harness(board)
    qapp.processEvents()
    qapp.processEvents()
    capsys.readouterr()

    window.resize(120, 220)
    qapp.processEvents()
    qapp.processEvents()

    captured = capsys.readouterr()
    assert board.is_collapsed() is True
    assert window.waveform_splitter.sizes()[0] <= window._count_board_collapse_threshold
    assert "Traceback" not in captured.err
    assert "_analysis_window_key_by_obj" not in captured.err
    assert "default_logger" not in captured.err

    window.close()
