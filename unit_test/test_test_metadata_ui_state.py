from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QApplication, QWidget

from ui.sequence import sequence_widget_test_metadata_ops as metadata_ops
from ui.sequence.sequence_tools_bar import SequenceToolsBar


class _MetadataStateHost(metadata_ops.SequenceWidgetTestMetadataOpsMixin, QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.toolsbar = SequenceToolsBar()
        self.left_panel = Mock()
        self.default_logger = Mock()
        self._init_test_round_metadata()


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(
        metadata_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: None),
    )
    monkeypatch.setattr(metadata_ops, "save_recorded_data_to_json", Mock())
    widget = _MetadataStateHost()
    yield widget
    widget.toolsbar.close()
    widget.close()


def test_restores_saved_ui_metadata_without_locking_fields(host, monkeypatch):
    monkeypatch.setattr(
        metadata_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(
            lambda logger: {
                "sample_number": "  SAMPLE-009  ",
                "current_test_round": 8,
            }
        ),
    )

    host._restore_test_metadata_ui_state()

    assert host.toolsbar.sample_number_lineedit.text() == "SAMPLE-009"
    assert host.toolsbar.current_round_spinbox.value() == 8
    assert not host.toolsbar.sample_number_lineedit.isReadOnly()
    assert not host.toolsbar.current_round_spinbox.isReadOnly()


@pytest.mark.parametrize("saved_round", [0, 10000, "7", True])
def test_invalid_saved_round_does_not_replace_ui_default(host, monkeypatch, saved_round):
    host.toolsbar.current_round_spinbox.setValue(1)
    monkeypatch.setattr(
        metadata_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: {"current_test_round": saved_round}),
    )

    host._restore_test_metadata_ui_state()

    assert host.toolsbar.current_round_spinbox.value() == 1


def test_editing_ui_metadata_persists_into_existing_sequence_state(host, monkeypatch):
    save = Mock()
    monkeypatch.setattr(metadata_ops, "save_recorded_data_to_json", save)
    host.toolsbar.sample_number_lineedit.setText("  SAMPLE-010  ")
    host.toolsbar.current_round_spinbox.setValue(9)

    host.toolsbar.sample_number_lineedit.editingFinished.emit()

    save.assert_called_once_with(
        sample_number="SAMPLE-010",
        current_test_round=9,
    )

    save.reset_mock()
    host.toolsbar.current_round_spinbox.editingFinished.emit()
    save.assert_called_once_with(
        sample_number="SAMPLE-010",
        current_test_round=9,
    )


def test_begin_round_persists_current_ui_metadata(host, monkeypatch):
    save = Mock()
    monkeypatch.setattr(metadata_ops, "save_recorded_data_to_json", save)
    host.toolsbar.sample_number_lineedit.setText("SAMPLE-001")
    host.toolsbar.current_round_spinbox.setValue(7)

    assert host._begin_test_round_metadata()

    save.assert_called_once_with(
        sample_number="SAMPLE-001",
        current_test_round=7,
    )


def test_sequence_logger_exists_before_metadata_initialization(monkeypatch):
    from ui.sequence.sequence_widget import SequenceWindow

    logger = Mock()

    class _InitializationChecked(Exception):
        pass

    def check_logger_and_stop(sequence):
        assert sequence.default_logger is logger
        raise _InitializationChecked

    monkeypatch.setattr(
        "ui.sequence.sequence_widget.LogManager.set_log_handler",
        lambda name: logger,
    )
    monkeypatch.setattr(
        SequenceWindow,
        "_init_test_round_metadata",
        check_logger_and_stop,
    )

    with pytest.raises(_InitializationChecked):
        SequenceWindow()
