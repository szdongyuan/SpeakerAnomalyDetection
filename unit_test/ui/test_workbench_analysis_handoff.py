import pytest

from unit_test.ui.test_workbench_recording_startup import startup
from base.workbench_analysis_input import WorkbenchAnalysisInputError
from base.motor_workbench_config import MotorWorkbenchConfigRepository
from base.workbench_analysis_input import build_analysis_payload
from unit_test.base.workbench_config_fixtures import write_workbench_files
from unit_test.ui.test_motor_workbench_controller import config, workbench, recording_workbench


def test_repository_returns_separate_objects(tmp_path):
    files = write_workbench_files(tmp_path)
    repository = MotorWorkbenchConfigRepository(files['programs'], files['queues'], files['layout'])
    result = repository.load_active()
    assert result.recording.context is result.analysis.context
    assert result.ui.layout.columns == 4


def test_success_hands_off_bound_input_after_original_completion(recording_workbench):
    w = recording_workbench
    inputs = []
    def receive(value):
        assert w.completions == [value.completion]
        assert w.controller._in_flight is None
        assert w.controller._condition_index == 1
        inputs.append(value)
    w.controller.bind_analysis_input(receive)
    w.controller.start_recording()
    analysis = w.controller.get_analysis_config()
    w.complete()
    assert inputs[0].context.analysis is analysis
    assert len(build_analysis_payload(inputs)['tasks']) == 2
    w.controller.bind_analysis_input(None)
    w.controller.start_recording()
    w.complete()
    assert len(inputs) == 1


def test_empty_queries(workbench):
    workbench.controller.refresh_configuration()
    assert workbench.controller.get_recording_config() is None
    assert workbench.controller.get_analysis_config() is None
    assert workbench.controller.get_ui_config() is None


def test_completed_callback_refresh_cannot_replace_bound_context(recording_workbench):
    from dataclasses import replace
    from unit_test.ui.test_motor_workbench_controller import configuration_with
    w = recording_workbench
    inputs = []
    original = w.controller.get_analysis_config()
    def completed(result):
        w.completions.append(result)
        w.repository.result = configuration_with(w.config, project_name="changed")
        w.controller.refresh_configuration()
    w.controller.bind_recording(replace(w.controller._recording, completed=completed))
    w.controller.bind_analysis_input(inputs.append)
    w.controller.start_recording()
    w.complete()
    assert inputs[0].context.analysis is original
    assert inputs[0].context.analysis != w.controller.get_analysis_config()
    assert build_analysis_payload(inputs)["tasks"]


def test_request_preparation_refresh_keeps_analysis_version(recording_workbench, monkeypatch):
    from unit_test.ui.test_motor_workbench_controller import configuration_with
    from ui.sequence import motor_workbench_controller as module
    w = recording_workbench
    inputs = []
    version = w.controller._analysis_version
    build = module.build_recording_request
    def build_and_refresh(*args, **kwargs):
        request = build(*args, **kwargs)
        w.repository.result = configuration_with(w.config, project_name="changed")
        w.controller.refresh_configuration()
        return request
    monkeypatch.setattr(module, "build_recording_request", build_and_refresh)
    w.controller.bind_analysis_input(inputs.append)
    w.controller.start_recording()
    w.complete()
    assert inputs[0].context.analysis_version == version


def test_synchronous_analysis_callback_error_propagates_after_completion(recording_workbench):
    import pytest
    w = recording_workbench
    def fail(value):
        raise ValueError("consumer bug")
    w.controller.bind_analysis_input(fail)
    w.outcome.on_start = w.complete
    with pytest.raises(ValueError, match="consumer bug"):
        w.controller.start_recording()
    assert len(w.completions) == 1
    assert w.controller._in_flight is None
    assert w.controller._condition_index == 1
    assert not w.problems


# The backend fakes only the acquisition boundary. Requests, WAV publication,
# Qt delivery, MainWindow signals and analysis assembly use production code.
def test_main_window_two_conditions_publish_eight_analysis_tasks(startup, tmp_path, monkeypatch, qt_app):
    import json
    from pathlib import Path
    from unittest.mock import Mock

    import soundfile as sf
    from PyQt5.QtCore import QThread, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QMessageBox
    import main_window
    from base import load_config
    from base.recording_service import RecordingService
    from ui.recording_service_bridge import RecordingServiceBridge
    from ui.sequence.sequence_widget import SequenceWindow
    from unit_test.base.workbench_config_fixtures import write_json
    from unit_test.ui.test_motor_workbench_integration import _wait_for_qt

    files = write_workbench_files(tmp_path, all_stations=True)
    product = json.loads(files['product'].read_text(encoding='utf-8'))
    # Four logical stations share the fake soundcard's three physical channels.
    for index, station in enumerate(product['stations']):
        for column, channel in enumerate(station['channels']):
            channel['hardware_channel']['physical_channel_index'] = (index + column) % 3
    write_json(files['product'], product)
    expected_test_queues = {}
    for name in ('测试队列A', '测试队列B'):
        queue = json.loads(files[name].read_text(encoding='utf-8'))
        queue[0]['seq1']['acq']['detail'].update(
            total_time=12 / 48000, startup_trim_ms=0, audio_validation={'enabled': False})
        expected_test_queues[name] = queue[0]['seq1']['analysis_list']
        write_json(files[name], queue)
    original_files = {path: path.read_bytes() for path in files.values()}
    repository = MotorWorkbenchConfigRepository(files['programs'], files['queues'], files['layout'])
    monkeypatch.setattr(main_window, 'MotorWorkbenchConfigRepository', lambda: repository)
    monkeypatch.setattr(load_config, 'SEQUENCE_CONFIG_REGISTRY_PATH', str(files['queues']))
    monkeypatch.setattr(QMessageBox, 'information', Mock())
    monkeypatch.setattr(QMessageBox, 'warning', Mock())
    forbidden = Mock(side_effect=AssertionError('workbench must not start analysis'))
    for name in ('_start_process_recording', '_start_next_queued_analysis'):
        monkeypatch.setattr(SequenceWindow, name, forbidden)
    service = RecordingService(
        backend_factory='unit_test.base.recording_process_fakes:workbench_soundcard_dependencies',
        backend_options={'trace_dir': str(tmp_path), 'frames': 12, 'chunk_frames': 4})
    bridge = RecordingServiceBridge(service)

    def login(window):
        _wait_for_qt(lambda: window.workbench_recording_device.snapshot is not None)
        return 'Engineer', 'test'

    startup.login(login)
    try:
        window = main_window.MainWindow(recording_bridge=bridge)
        window.sequence_window.lineedit_type.setText("M-02")
        window.sequence_window.toolsbar.sample_number_lineedit.setText("007")
        completions, inputs = [], []
        window.workbench_recording_completed.connect(completions.append)

        def receive(item):
            assert QThread.currentThread() == qt_app.thread()
            assert completions[-1] is item.completion
            assert window.workbench_controller._in_flight is None
            inputs.append(item)

        window.workbench_analysis_input_ready.connect(receive)
        button = window.sequence_window.player_btn
        _wait_for_qt(button.isEnabled)
        for count in (1, 2):
            QTest.mouseClick(button, Qt.LeftButton)
            _wait_for_qt(lambda: len(completions) == count and button.isEnabled())
            assert len(inputs) == count
        payload = build_analysis_payload(inputs)
        assert set(payload) == {'stations', 'conditions', 'test_queues', 'tasks'}
        assert len(payload['tasks']) == 8
        assert len(startup.analysis_service.submissions) == 2
        assert sum(len(batch['tasks']) for batch in startup.analysis_service.submissions) == 8
        assert all(Path(task['wav_path']).is_file()
                   for batch in startup.analysis_service.submissions for task in batch['tasks'])
        assert inputs[0].context.analysis_version == inputs[1].context.analysis_version
        expected_paths = {(result.condition_id, wav.station_id): wav.path
                          for result in completions for wav in result.stations}
        assert len(set(expected_paths.values())) == 8
        stations = {station['station_id']: station for station in payload['stations']}
        conditions = {condition['condition_id']: condition for condition in payload['conditions']}
        assert {key: value['test_queue'] for key, value in conditions.items()} == {
            'c1': '测试队列A', 'c2': '测试队列B'}
        assert all(set(condition) == {'condition_id', 'condition_name', 'test_queue', 'segmented_analysis'}
                   for condition in conditions.values())
        for task in payload['tasks']:
            assert task['wav_path'] == expected_paths[task['condition_id'], task['station_id']]
            assert Path(task['wav_path']).is_file()
            info = sf.info(task['wav_path'])
            assert (info.frames, info.channels, info.samplerate, info.subtype) == (12, 2, 48000, 'PCM_24')
            assert len(stations[task['station_id']]['channels']) == 2
            test_queue = payload['test_queues'][conditions[task['condition_id']]['test_queue']]
            assert set(test_queue) == {'analysis_channel_basis', 'analysis_config'}
            assert test_queue['analysis_channel_basis'] == 'wav'
            assert task['analysis_result_directory'] is None
        assert {name: test_queue['analysis_config'] for name, test_queue in payload['test_queues'].items()} == expected_test_queues
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert {path: path.read_bytes() for path in files.values()} == original_files
        forbidden.assert_not_called()
        QMessageBox.warning.assert_not_called()
    finally:
        bridge.shutdown()
        assert service.closed.wait(10)


@pytest.fixture
def file_recording_workbench(recording_workbench, tmp_path):
    from dataclasses import replace
    from base.recording_process_protocol import FrozenConfig
    w = recording_workbench
    w.files = write_workbench_files(tmp_path, all_stations=True)
    w.repository = MotorWorkbenchConfigRepository(w.files['programs'], w.files['queues'], w.files['layout'])
    w.controller._repository = w.repository
    device = w.device.snapshot.device.to_dict()
    device['max_input_channels'] = 8
    w.device.snapshot = replace(w.device.snapshot, device=FrozenConfig.snapshot(device), channels=tuple(range(8)))
    w.controller.refresh_configuration()
    return w


@pytest.mark.parametrize('change,resets,changes_version', [
    ('title', False, False), ('layout', False, False), ('preview', False, False),
    ('spl', False, True), ('fft', False, True),
    ('duration', True, False), ('quality', True, False),
    ('channels', True, True), ('association', True, True), ('source', True, False),
])
def test_inflight_file_refresh_matrix(file_recording_workbench, change, resets, changes_version):
    import json
    from unit_test.base.workbench_config_fixtures import write_json
    w = file_recording_workbench
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    w.controller.start_recording()
    w.complete()
    assert w.controller._condition_index == 1
    w.controller.start_recording()
    request = w.requests[-1]
    context = w.controller._in_flight.analysis_context
    original_analysis = context.analysis.describe()
    version = context.analysis_version
    if change in ('title', 'layout'):
        path = w.files['layout']
        value = json.loads(path.read_text(encoding='utf-8'))
        if change == 'title':
            value['cells'][0][0]['title'] = '更新标题'
        else:
            value['cells'].reverse()
    elif change in ('channels', 'association'):
        path = w.files['product']
        value = json.loads(path.read_text(encoding='utf-8'))
        if change == 'channels':
            value['stations'][0]['channels'].reverse()
        else:
            value['conditions'][1]['applicable_station_ids'] = ['s1', 's3']
    elif change == 'source':
        copied = w.files['product'].with_name('copy.json')
        copied.write_bytes(w.files['product'].read_bytes())
        path = w.files['programs']
        value = json.loads(path.read_text(encoding='utf-8'))
        value['active_file'] = copied.name
        value['configs'].append({'file': copied.name, 'project_name': '电机'})
    else:
        path = w.files['测试队列B']
        value = json.loads(path.read_text(encoding='utf-8'))
        seq = value[0]['seq1']
        if change == 'spl':
            seq['analysis_list']['声压级']['scalar_upper_value'] = 99
        elif change == 'fft':
            seq['analysis_list']['频谱']['window'] = 'blackman'
        elif change == 'duration':
            seq['acq']['detail']['total_time'] = 8
        elif change == 'quality':
            seq['acq']['detail']['audio_validation']['min_rms_dbfs'] = -50
        else:
            seq['acq']['detail']['recording_preview_time_mode'] = 'relative_latest'
    write_json(path, value)
    w.controller.refresh_configuration()
    assert w.controller._configuration_valid
    assert w.controller._condition_index == (0 if resets else 1)
    assert (w.controller._analysis_version != version) is changes_version
    assert w.controller._in_flight.analysis_context is context
    assert w.requests[-1] is request
    w.complete()
    old = inputs[-1]
    assert old.context is context
    assert old.context.analysis.describe() == original_analysis
    old_payload = build_analysis_payload([old])
    assert old_payload['test_queues']['测试队列B'] == original_analysis['test_queues']['测试队列B']
    assert [(task['station_id'], task['wav_path']) for task in old_payload['tasks']] == [
        (station.station_id, station.path) for station in old.completion.stations]
    assert w.controller._condition_index == 0
    # Re-record both conditions so the new c2 input resolves the same test_queue name.
    for _ in range(2):
        w.controller.start_recording()
        w.complete()
    new = inputs[-1]
    assert new.context.condition_id == 'c2'
    current = w.controller.get_analysis_config().describe()
    assert build_analysis_payload([new])['test_queues']['测试队列B'] == current['test_queues']['测试队列B']
    if changes_version:
        with pytest.raises(WorkbenchAnalysisInputError, match='version.*separate batches'):
            build_analysis_payload([old, new])
    else:
        assert len(build_analysis_payload([old, new])['tasks']) == 8
    with pytest.raises(WorkbenchAnalysisInputError, match='duplicate'):
        build_analysis_payload([old, old])


@pytest.mark.parametrize('outcome', ['failure', 'cancelled', 'rejected'])
def test_unsuccessful_recording_releases_context_without_analysis(recording_workbench, outcome):
    w = recording_workbench
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    if outcome == 'rejected':
        w.outcome.error = RuntimeError('service rejected request')
    w.controller.start_recording()
    if outcome != 'rejected':
        w.complete(outcome=outcome)
    assert w.controller._in_flight is None
    assert w.controller._condition_index == 0
    assert inputs == w.completions == []


def test_unsubscribed_completion_is_not_retained_or_replayed(recording_workbench):
    import gc
    import weakref
    w = recording_workbench
    w.controller.start_recording()
    context_ref = weakref.ref(w.controller._in_flight.analysis_context)
    w.complete()
    # The fake service logs callbacks for assertions; release that external owner.
    w.callbacks.clear()
    gc.collect()
    assert context_ref() is None
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    assert inputs == []
    w.controller.start_recording()
    w.complete()
    assert len(inputs) == 1
    assert inputs[0].completion is w.completions[-1]
