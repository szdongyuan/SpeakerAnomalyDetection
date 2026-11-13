"""Raw CSV backend contracts independent of the retired sequence recorder."""
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from base.raw_audio_csv_service import RawAudioCsvService
from base.raw_audio_csv_protocol import CsvExportRequest


def submit(service, tmp_path, channels=(0, 2)):
    source = tmp_path / 'recording.wav'
    if not source.exists():
        sf.write(source, np.zeros((3, 2), dtype=np.float32), 8000)
    request = CsvExportRequest('task', 'recording', str(source), str(source.with_suffix('.csv')),
                               channels, 'original-group', 'original-record')
    token = service.reserve(request.recording_id).reservation
    assert service.commit(token, request) == 'accepted'
    return request


def test_parent_never_formats_csv_and_draining_accepts_reserved_recording(tmp_path, monkeypatch):
    from base import raw_audio_csv_exporter
    def parent_export_forbidden(*args, **kwargs):
        raise AssertionError('parent formatted CSV')
    monkeypatch.setattr(raw_audio_csv_exporter, 'export_raw_audio_csv', parent_export_forbidden)
    service = RawAudioCsvService()
    events = []
    service.subscribe(events.append)
    source = tmp_path / 'recording.wav'
    sf.write(source, np.zeros((3, 2), dtype=np.float32), 8000)
    request = CsvExportRequest('task', 'recording', str(source), str(source.with_suffix('.csv')),
                              (0, 2), 'original-group', 'original-record')
    token = service.reserve(request.recording_id).reservation
    try:
        service.begin_shutdown()
        assert service.commit(token, request) == 'accepted'
        assert service.closed.wait(15)
        terminal, = [event for event in events if event.kind == 'terminal']
        assert terminal.result.frames == 3
        assert terminal.result.worker_pid != __import__('os').getpid()
        assert terminal.task.request.owner_group == 'original-group'
        assert terminal.task.request.owner_record == 'original-record'
        assert not Path(request.csv_path).exists()
        with ZipFile(request.csv_path + '.zip') as archive:
            assert archive.read('recording.csv').decode('utf-8-sig').splitlines()[0] == 'time_s,CH1,CH3'
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)


def test_scheduled_pcm24_csv_child_keeps_saved_samples_and_channel_labels(tmp_path):
    from base.save_data import save_audio_simple
    service = RawAudioCsvService()
    events = []
    service.subscribe(events.append)
    source = np.array([[0.5, 2.5], [2**-25, -2**-25], [0.001953125, -3]], dtype=np.float32)
    path = tmp_path / 'recording.wav'
    save_audio_simple(str(path), source, 8000)
    try:
        request = submit(service, tmp_path, (7, 1))
        service.begin_shutdown()
        assert service.closed.wait(15)
        terminal, = [event for event in events if event.kind == 'terminal']
        assert terminal.result.frames == 3
        with ZipFile(request.csv_path + '.zip') as archive:
            rows = list(csv.reader(io.StringIO(archive.read('recording.csv').decode('utf-8-sig'))))
        assert rows[0] == ['time_s', 'CH8', 'CH2']
        assert [row[1:] for row in rows[1:]] == [
            ['0.5', '0.999999881'], ['0', '-1.1920929e-07'], ['0.001953125', '-1']]
        saved, _ = sf.read(path, dtype='float32', always_2d=True)
        np.testing.assert_array_equal(np.array([row[1:] for row in rows[1:]], dtype=np.float32), saved)
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)
