import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest

from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
)

DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 2,
}

ROOT = Path(__file__).resolve().parents[1]


def _load_method(path, class_name, method_name, extra_globals=None):
    module_tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[method_node],
        decorator_list=[],
    )
    namespace = dict(extra_globals or {})
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[test_class], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return getattr(namespace["TestClass"], method_name)


def test_runtime_refresh_resolves_factor_for_current_hardware():
    sequence = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[1],
        v2pa_factor=0.0,
    )

    resolve = mock.Mock(return_value=2.5)
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {"get_mic_v2pa_factor": resolve},
    )

    update_factor(sequence)

    resolve.assert_called_once_with(DEVICE, [1])
    assert sequence.v2pa_factor == 2.5


def test_legacy_scalar_refresh_is_single_channel_only():
    sequence = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[0, 2],
        v2pa_factor=9.0,
    )
    resolve = mock.Mock(return_value=2.5)
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {"get_mic_v2pa_factor": resolve},
    )

    update_factor(sequence)

    resolve.assert_not_called()
    assert sequence.v2pa_factor == 0.0


@pytest.mark.parametrize(
    "error_type",
    [MicCalibrationFormatError, MicCalibrationIOError],
)
def test_legacy_scalar_refresh_contains_calibration_file_errors(error_type):
    error = error_type("broken calibration")
    resolve = mock.Mock(side_effect=error)
    default_logger = mock.Mock()
    dialogs = SimpleNamespace(critical=mock.Mock(), warning=mock.Mock())
    sequence = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[1],
        v2pa_factor=9.0,
        default_logger=default_logger,
    )
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {
            "get_mic_v2pa_factor": resolve,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": dialogs,
        },
    )

    update_factor(sequence)

    resolve.assert_called_once_with(DEVICE, [1])
    assert sequence.v2pa_factor == 0.0
    default_logger.error.assert_called_once()
    log_args = default_logger.error.call_args.args
    assert error_type.__name__ in " ".join(str(arg) for arg in log_args)
    assert str(error) in " ".join(str(arg) for arg in log_args)
    dialogs.critical.assert_not_called()
    dialogs.warning.assert_not_called()


def test_legacy_scalar_refresh_propagates_unknown_errors():
    error = RuntimeError("programming error")
    resolve = mock.Mock(side_effect=error)
    default_logger = mock.Mock()
    dialogs = SimpleNamespace(critical=mock.Mock(), warning=mock.Mock())
    sequence = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[1],
        v2pa_factor=9.0,
        default_logger=default_logger,
    )
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {
            "get_mic_v2pa_factor": resolve,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": dialogs,
        },
    )

    with pytest.raises(RuntimeError) as raised:
        update_factor(sequence)

    assert raised.value is error
    assert sequence.v2pa_factor == 9.0
    default_logger.error.assert_not_called()
    dialogs.critical.assert_not_called()
    dialogs.warning.assert_not_called()


@pytest.mark.parametrize(
    "error_type",
    [MicCalibrationFormatError, MicCalibrationIOError],
)
def test_main_window_initialization_contains_scalar_refresh_errors(error_type):
    error = error_type("broken calibration")
    resolve = mock.Mock(side_effect=error)
    default_logger = mock.Mock()
    dialogs = SimpleNamespace(critical=mock.Mock(), warning=mock.Mock())
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {
            "get_mic_v2pa_factor": resolve,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": dialogs,
        },
    )

    class FakeWidget:
        def __init__(self):
            self.layout = None
            self.mouse_tracking = False

        def setLayout(self, layout):
            self.layout = layout

        def setMouseTracking(self, enabled):
            self.mouse_tracking = enabled

    class FakeLayout:
        def __init__(self):
            self.widgets = []
            self.alignment = None
            self.margins = None
            self.spacing = None

        def addWidget(self, widget):
            self.widgets.append(widget)

        def setAlignment(self, alignment):
            self.alignment = alignment

        def setContentsMargins(self, *margins):
            self.margins = margins

        def setSpacing(self, spacing):
            self.spacing = spacing

    class FakeSequenceWindow:
        def __init__(self, *, recording_bridge):
            self.recording_bridge = recording_bridge
            self.v2pa_factor = 9.0
            self.default_logger = default_logger
            self.update_v2pa_factor = MethodType(update_factor, self)

    title_bar = object()
    menu_bar = object()
    menu_row = object()
    window = SimpleNamespace(
        recording_bridge=object(),
        mic=DEVICE,
        mic_channels=[1],
        speaker={"name": "Speaker"},
        speaker_channels=[0],
        init_menu=mock.Mock(return_value=menu_bar),
        set_title=mock.Mock(return_value=title_bar),
        _create_menu_row=mock.Mock(return_value=menu_row),
        setCentralWidget=mock.Mock(),
    )
    init_sequence = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "init_sequence_widget",
        {
            "QWidget": FakeWidget,
            "QVBoxLayout": FakeLayout,
            "SequenceWindow": FakeSequenceWindow,
            "Qt": SimpleNamespace(AlignTop="align-top"),
        },
    )

    init_sequence(window)

    assert window.sequence_window.recording_bridge is window.recording_bridge
    assert window.sequence_window.v2pa_factor == 0.0
    resolve.assert_called_once_with(DEVICE, [1])
    central_widget = window.setCentralWidget.call_args.args[0]
    assert central_widget.layout.widgets == [title_bar, menu_row, window.sequence_window]
    assert central_widget.layout.alignment == "align-top"
    assert central_widget.layout.margins == (0, 0, 0, 0)
    assert central_widget.layout.spacing == 0
    assert central_widget.mouse_tracking is True
    default_logger.error.assert_called_once()
    log_text = " ".join(str(arg) for arg in default_logger.error.call_args.args)
    assert error_type.__name__ in log_text
    assert str(error) in log_text
    dialogs.critical.assert_not_called()
    dialogs.warning.assert_not_called()


def test_main_window_passes_current_input_to_calibration_and_refreshes_on_success():
    events = []

    class FakeCalibrationWindow:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self.input_calibration_flag = True
            self.speaker = None

        def exec(self):
            events.append(("exec", None))

    sequence = SimpleNamespace(
        update_v2pa_factor=lambda: events.append(("refresh", None))
    )
    window = SimpleNamespace(
        recording_bridge=object(),
        mic=DEVICE,
        mic_channels=[1],
        speaker={"name": "Speaker"},
        sequence_window=sequence,
    )

    open_calibration = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "on_calibration_window_init",
        {"CalibrationWindow": FakeCalibrationWindow},
    )
    open_calibration(window)

    assert events == [
        (
            "init",
            {"input_device": DEVICE, "input_channels": [1], "recording_bridge": window.recording_bridge},
        ),
        ("exec", None),
        ("refresh", None),
    ]


@pytest.mark.parametrize(
    "error_type",
    [MicCalibrationFormatError, MicCalibrationIOError],
)
def test_successful_calibration_contains_scalar_refresh_errors(error_type):
    error = error_type("broken calibration")
    resolve = mock.Mock(side_effect=error)
    default_logger = mock.Mock()
    dialogs = SimpleNamespace(critical=mock.Mock(), warning=mock.Mock())
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {
            "get_mic_v2pa_factor": resolve,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": dialogs,
        },
    )
    sequence = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[1],
        v2pa_factor=9.0,
        default_logger=default_logger,
    )
    sequence.update_v2pa_factor = MethodType(update_factor, sequence)
    created_dialogs = []

    class FakeCalibrationWindow:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.input_calibration_flag = True
            self.speaker = None
            self.exec = mock.Mock()
            created_dialogs.append(self)

    speaker = {"name": "Speaker"}
    window = SimpleNamespace(
        recording_bridge=object(),
        mic=DEVICE,
        mic_channels=[1],
        speaker=speaker,
        sequence_window=sequence,
    )
    open_calibration = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "on_calibration_window_init",
        {"CalibrationWindow": FakeCalibrationWindow},
    )

    open_calibration(window)

    assert len(created_dialogs) == 1
    dialog = created_dialogs[0]
    assert dialog.kwargs == {
        "input_device": DEVICE,
        "input_channels": [1],
        "recording_bridge": window.recording_bridge,
    }
    assert dialog.speaker == speaker
    dialog.exec.assert_called_once_with()
    assert sequence.v2pa_factor == 0.0
    resolve.assert_called_once_with(DEVICE, [1])
    default_logger.error.assert_called_once()
    log_text = " ".join(str(arg) for arg in default_logger.error.call_args.args)
    assert error_type.__name__ in log_text
    assert str(error) in log_text
    dialogs.critical.assert_not_called()
    dialogs.warning.assert_not_called()


def test_main_window_does_not_refresh_after_failed_or_cancelled_calibration():
    class FakeCalibrationWindow:
        def __init__(self, **kwargs):
            self.input_calibration_flag = False
            self.speaker = None

        def exec(self):
            return None

    sequence = SimpleNamespace(update_v2pa_factor=mock.Mock())
    window = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[1],
        speaker={"name": "Speaker"},
        sequence_window=sequence,
    )

    open_calibration = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "on_calibration_window_init",
        {"CalibrationWindow": FakeCalibrationWindow},
    )
    open_calibration(window)

    sequence.update_v2pa_factor.assert_not_called()


def test_hardware_change_refreshes_factor_after_new_input_is_installed():
    new_device = {**DEVICE, "name": "New Microphone"}
    events = []

    def refresh_factor():
        events.append((sequence.mic, list(sequence.mic_channels)))

    sequence = SimpleNamespace(
        player_status_flag=False,
        mic=None,
        speaker=None,
        mic_channels=[],
        speaker_channels=[],
        update_v2pa_factor=refresh_factor,
        refresh_channel_windows=mock.Mock(),
    )
    window = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[0],
        speaker={"index": 2, "name": "Speaker", "hostapi": 3},
        speaker_channels=[],
        sequence_window=sequence,
        update_statusbar=mock.Mock(),
    )
    save_selection = mock.Mock()
    open_hardware = mock.Mock(
        return_value=(
            True,
            window.speaker,
            [],
            new_device,
            [1],
        )
    )
    fake_device_manager = SimpleNamespace(
        get_api_info=mock.Mock(return_value={"name": "Test API"})
    )
    open_window = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "on_hardware_window_init",
        {
            "QMessageBox": SimpleNamespace(warning=mock.Mock()),
            "SoundDeviceManager": fake_device_manager,
            "open_hardware_selection_window": open_hardware,
            "save_if_changed": save_selection,
        },
    )

    open_window(window)

    assert events == [(new_device, [1])]
    save_selection.assert_called_once_with(new_device, window.speaker, [1], [])
    sequence.refresh_channel_windows.assert_called_once_with()


@pytest.mark.parametrize(
    "error_type",
    [MicCalibrationFormatError, MicCalibrationIOError],
)
def test_hardware_change_contains_scalar_refresh_errors(error_type):
    new_device = {**DEVICE, "name": "New Microphone"}
    error = error_type("broken calibration")
    resolve = mock.Mock(side_effect=error)
    default_logger = mock.Mock()
    dialogs = SimpleNamespace(critical=mock.Mock(), warning=mock.Mock())
    update_factor = _load_method(
        ROOT / "ui" / "sequence" / "sequence_widget_streaming_ops.py",
        "SequenceWidgetStreamingOpsMixin",
        "update_v2pa_factor",
        {
            "get_mic_v2pa_factor": resolve,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": dialogs,
        },
    )
    sequence = SimpleNamespace(
        player_status_flag=False,
        mic=None,
        speaker=None,
        mic_channels=[],
        speaker_channels=[],
        v2pa_factor=9.0,
        default_logger=default_logger,
        refresh_channel_windows=mock.Mock(),
    )
    sequence.update_v2pa_factor = MethodType(update_factor, sequence)
    speaker = {"index": 2, "name": "Speaker", "hostapi": 3}
    window = SimpleNamespace(
        mic=DEVICE,
        mic_channels=[0],
        speaker=speaker,
        speaker_channels=[],
        sequence_window=sequence,
        update_statusbar=mock.Mock(),
    )
    save_selection = mock.Mock()
    open_hardware = mock.Mock(
        return_value=(
            True,
            speaker,
            [],
            new_device,
            [1],
        )
    )
    fake_device_manager = SimpleNamespace(
        get_api_info=mock.Mock(return_value={"name": "Test API"})
    )
    open_window = _load_method(
        ROOT / "main_window.py",
        "MainWindow",
        "on_hardware_window_init",
        {
            "QMessageBox": dialogs,
            "SoundDeviceManager": fake_device_manager,
            "open_hardware_selection_window": open_hardware,
            "save_if_changed": save_selection,
        },
    )

    open_window(window)

    assert sequence.v2pa_factor == 0.0
    resolve.assert_called_once_with(new_device, [1])
    save_selection.assert_called_once_with(new_device, speaker, [1], [])
    window.update_statusbar.assert_called_once_with()
    sequence.refresh_channel_windows.assert_called_once_with()
    default_logger.error.assert_called_once()
    log_text = " ".join(str(arg) for arg in default_logger.error.call_args.args)
    assert error_type.__name__ in log_text
    assert str(error) in log_text
    dialogs.critical.assert_not_called()
    dialogs.warning.assert_not_called()
