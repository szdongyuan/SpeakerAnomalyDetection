import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
            {"input_device": DEVICE, "input_channels": [1]},
        ),
        ("exec", None),
        ("refresh", None),
    ]


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
