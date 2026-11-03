import json
import struct

import numpy as np
import pytest
import soundfile as sf
from scipy.io import wavfile

from base.vkinging_wav import VkingingVoltageWriter, preflight_vkinging_wav_size
from base.wav_calibration_metadata import inspect_wav_calibration_metadata


def metadata(channels=(15, 0, 8), rate=51200, bits=32):
    return {'backend': 'vkinging', 'schema_version': 1,
            'acquisition': {'hardware_id': 'machine-A', 'model': None, 'unit': 'V',
                            'input_mode': 'IEPE', 'range_min': -5.0, 'range_max': 5.0,
                            'sample_rate': rate, 'bit_depth': bits},
            'recorded_channels': [
                {'wav_channel_index': i, 'physical_channel_index': ch, 'calibrated': True,
                 'v2pa_factor': 2.5 + i, 'standard_spl': None,
                 'calibration_conditions': {'input_mode': 'IEPE', 'unit': 'V',
                                            'range_min': -10.0, 'range_max': 10.0}}
                for i, ch in enumerate(channels)]}


def chunks(data):
    result = []
    offset = 12
    while offset + 8 <= len(data):
        name, size = struct.unpack_from('<4sI', data, offset)
        end = offset + 8 + size + size % 2
        result.append((name, data[offset:end]))
        offset = end
    return result


def rebuild(parts):
    payload = b'WAVE' + b''.join(part for _, part in parts)
    return b'RIFF' + struct.pack('<I', len(payload)) + payload


def make_writer(tmp_path, bits=32, channels=(15, 0, 8), meta=None):
    return VkingingVoltageWriter(tmp_path / 'final.wav', sample_rate=51200, bit_depth=bits,
                                physical_channels=channels, target_frames=2,
                                metadata=meta if meta is not None else metadata(channels, bits=bits))


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('channels', [(15, 0, 8), tuple(range(32))])
def test_voltage_roundtrip_and_commit(tmp_path, bits, channels):
    values = np.tile(np.array([[2.5], [-3.0]], dtype='float64'), (1, len(channels)))
    writer = make_writer(tmp_path, bits, channels)
    writer.write_chunk(values[:1])
    writer.write_chunk(values[1:])
    assert writer.frames_written == 2
    assert not writer.resources_released
    writer.close()
    assert writer.resources_released
    prepared = writer.finalize()
    assert not (tmp_path / 'final.wav').exists()
    assert prepared['frames'] == 2
    result = writer.commit()
    actual, rate = sf.read(result['path'], dtype='float64', always_2d=True)
    np.testing.assert_array_equal(actual, values)
    scipy_rate, scipy_values = wavfile.read(result['path'])
    np.testing.assert_array_equal(scipy_values, values)
    assert scipy_rate == rate == 51200
    assert sf.info(result['path']).subtype == ('FLOAT' if bits == 32 else 'DOUBLE')
    parts = chunks((tmp_path / 'final.wav').read_bytes())
    names = [name for name, _ in parts]
    assert names.index(b'vkin') < names.index(b'data')
    marker = next(part for name, part in parts if name == b'vkin')
    assert marker[8:] == b'vkinging;unit=V;schema=1\0\0'
    inspection = inspect_wav_calibration_metadata(result['path'])
    assert (inspection.source, inspection.unit, inspection.status) == ('vkinging', 'V', 'vkinging_valid')
    assert [ch['physical_channel_index'] for ch in inspection.metadata['recorded_channels']] == list(channels)


@pytest.mark.parametrize('change,status', [
    ('remove_json', 'vkinging_invalid'), ('remove_marker', 'vkinging_invalid'),
    ('bad_json', 'vkinging_invalid'), ('bad_marker', 'vkinging_invalid'),
    ('schema', 'unsupported'), ('marker_schema', 'unsupported'),
    ('riff_long', 'vkinging_invalid'), ('riff_short', 'vkinging_invalid'),
    ('count', 'vkinging_invalid'), ('rate', 'vkinging_invalid'),
])
def test_provenance_damage_never_becomes_legacy(tmp_path, change, status):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    prepared = writer.finalize()
    from pathlib import Path
    path = Path(prepared['path'])
    parts = chunks(path.read_bytes())
    if change == 'remove_json':
        parts = [(k, v) for k, v in parts if k != b'LIST']
    elif change == 'remove_marker':
        parts = [(k, v) for k, v in parts if k != b'vkin']
    elif change in ('bad_json', 'schema', 'count', 'rate'):
        old, new = {'bad_json': (b'"backend":', b'!backend!:'),
                    'schema': (b'"schema_version":1', b'"schema_version":9'),
                    'count': (b'"wav_channel_index":2', b'"wav_channel_index":3'),
                    'rate': (b'"sample_rate":51200', b'"sample_rate":48000')}[change]
        parts = [(k, v.replace(old, new)) for k, v in parts]
    elif change in ('bad_marker', 'marker_schema'):
        old, new = (b'unit=V', b'unit=X') if change == 'bad_marker' else (b'schema=1', b'schema=9')
        parts = [(k, v.replace(old, new) if k == b'vkin' else v) for k, v in parts]
    data = rebuild(parts)
    if change.startswith('riff_'):
        data = data[:4] + struct.pack('<I', len(data) - 8 + (100 if change == 'riff_long' else -100)) + data[8:]
    path.write_bytes(data)
    result = inspect_wav_calibration_metadata(path)
    assert result.source == 'vkinging'
    assert result.status == status
    assert result.reason


def test_all_provenance_stripped_is_legacy(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    from pathlib import Path
    path = Path(writer.finalize()['path'])
    path.write_bytes(rebuild([(k, v) for k, v in chunks(path.read_bytes()) if k not in (b'vkin', b'LIST')]))
    assert inspect_wav_calibration_metadata(path).status == 'legacy_absent'


def test_frozen_metadata_and_uncalibrated_channels(tmp_path):
    data = metadata()
    data['recorded_channels'][1].update(calibrated=False, v2pa_factor=None, standard_spl=None, calibration_conditions=None)
    from base.vkinging_input import _freeze_snapshot
    writer = make_writer(tmp_path, meta=_freeze_snapshot(data))
    writer.write_chunk(np.ones((2, 3)))
    assert inspect_wav_calibration_metadata(writer.finalize()['path']).status == 'vkinging_uncalibrated'


def test_existing_target_and_short_capture_cannot_publish(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((1, 3)))
    with pytest.raises(ValueError, match='frames'):
        writer.finalize()
    assert writer.resources_released
    assert not (tmp_path / 'final.wav').exists()
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    writer.finalize()
    target = tmp_path / 'final.wav'
    target.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        writer.commit()
    assert target.read_bytes() == b'keep me'


def test_preflight_uses_serialized_metadata_and_rejects_riff_overflow(tmp_path):
    small = preflight_vkinging_wav_size(sample_rate=51200, bit_depth=32, channels=3,
                                      target_frames=2, metadata=metadata())
    data = metadata()
    data['acquisition']['model'] = 'x' * 200
    large = preflight_vkinging_wav_size(sample_rate=51200, bit_depth=32, channels=3,
                                      target_frames=2, metadata=data)
    assert large == small + 198  # JSON null (4 bytes) becomes a quoted 200-byte string.
    with pytest.raises(ValueError, match='RIFF'):
        VkingingVoltageWriter(tmp_path / 'too-large.wav', sample_rate=51200, bit_depth=32,
                              physical_channels=(15, 0, 8), target_frames=2**32, metadata=metadata())
    assert list(tmp_path.iterdir()) == []


def test_assembly_cancellation_retains_owned_paths(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    def cancelled():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        writer.finalize(checkpoint=cancelled)
    assert writer.resources_released
    assert writer.owned_paths
    assert not (tmp_path / 'final.wav').exists()


def test_missing_soundfile_has_no_pcm_fallback(monkeypatch, tmp_path):
    from base import vkinging_wav
    def missing():
        raise ImportError('soundfile missing')
    monkeypatch.setattr(vkinging_wav, '_soundfile', missing)
    with pytest.raises(ImportError, match='soundfile missing'):
        make_writer(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_write_failure_remains_failed_after_close(tmp_path):
    writer = make_writer(tmp_path)
    real = writer._handle
    class BrokenWrite:
        def write(self, samples):
            raise OSError('disk full')
        def close(self):
            real.close()
    writer._handle = BrokenWrite()
    with pytest.raises(OSError, match='disk full'):
        writer.write_chunk(np.ones((2, 3)))
    writer.abort()
    assert writer.resources_released
    with pytest.raises(RuntimeError):
        writer.finalize()
    assert writer.frames_written == 0
    assert not (tmp_path / 'final.wav').exists()


def test_uncertain_close_prevents_publication(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    real = writer._handle
    class BrokenClose:
        def close(self):
            real.close()
            raise OSError('close failed')
    writer._handle = BrokenClose()
    with pytest.raises(OSError, match='close failed'):
        writer.finalize()
    assert not writer.resources_released
    with pytest.raises(RuntimeError):
        writer.commit()
    assert not (tmp_path / 'final.wav').exists()


def test_metadata_validation_failure_cannot_commit(monkeypatch, tmp_path):
    from base import vkinging_wav
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    def invalid(path):
        from base.wav_calibration_metadata import WavCalibrationInspection
        return WavCalibrationInspection('vkinging', 'vkinging_invalid', reason='broken JSON')
    monkeypatch.setattr(vkinging_wav, 'inspect_wav_calibration_metadata', invalid)
    with pytest.raises(ValueError, match='broken JSON'):
        writer.finalize()
    assert writer.resources_released
    with pytest.raises(RuntimeError):
        writer.commit()
    assert len(writer.owned_paths) == 2


def test_valid_legacy_audio_bytes_are_not_provenance(tmp_path):
    path = tmp_path / 'legacy.wav'
    # IEEE floating sample bytes can incidentally spell an unknown chunk ID.
    samples = np.frombuffer(b'vkin' + b'\0' * 8, dtype='<f4')
    sf.write(path, samples, 48000, subtype='FLOAT')
    assert inspect_wav_calibration_metadata(path).status == 'legacy_absent'


def test_cancel_checkpoint_before_publication(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    writer.finalize()
    def cancelled():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        writer.commit(checkpoint=cancelled)
    assert not (tmp_path / 'final.wav').exists()


def test_assembly_close_failure_preserves_primary_and_release_evidence(monkeypatch, tmp_path):
    import builtins
    from base import vkinging_wav
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    writer.close()
    real_open = builtins.open
    class BrokenOutput:
        def __init__(self, handle):
            self.handle = handle
        def write(self, data):
            raise OSError('primary copy failure')
        def close(self):
            self.handle.close()
            raise OSError('secondary close failure')
    def open_file(path, mode):
        handle = real_open(path, mode)
        return BrokenOutput(handle) if mode == 'wb' else handle
    monkeypatch.setattr(vkinging_wav, 'open', open_file, raising=False)
    with pytest.raises(OSError, match='primary copy failure') as caught:
        writer.finalize()
    assert any('secondary close failure' in note for note in caught.value.__notes__)
    assert not writer.resources_released
    assert not (tmp_path / 'final.wav').exists()


def test_source_close_failure_is_uncertain_and_preserves_cancellation(monkeypatch, tmp_path):
    import builtins
    from base import vkinging_wav
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    writer.close()
    real_open = builtins.open
    class BrokenInput:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            self.handle.close()
            raise OSError('source close failure')
    def open_file(path, mode):
        handle = real_open(path, mode)
        return BrokenInput(handle) if mode == 'rb' else handle
    monkeypatch.setattr(vkinging_wav, 'open', open_file, raising=False)
    def cancelled():
        assert not writer.resources_released
        raise InterruptedError('primary cancellation')
    with pytest.raises(InterruptedError, match='primary cancellation') as caught:
        writer.finalize(checkpoint=cancelled)
    assert any('source close failure' in note for note in caught.value.__notes__)
    assert not writer.resources_released


def test_escaped_json_backend_still_declares_vkinging(tmp_path):
    from pathlib import Path
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    path = Path(writer.finalize()['path'])
    parts = [(name, part) for name, part in chunks(path.read_bytes()) if name != b'vkin']
    from base.wav_calibration_metadata import _build_info_comment_list_chunk
    parts = [(name, _build_info_comment_list_chunk('mic_calibration=' + json.dumps(metadata()).replace(
        '"backend": "vkinging"', '"backend": "vkin\\u0067ing"')) if name == b'LIST' else part)
             for name, part in parts]
    path.write_bytes(rebuild(parts))
    result = inspect_wav_calibration_metadata(path)
    assert result.source == 'vkinging'
    assert result.status == 'vkinging_invalid'


def test_commit_checks_cancellation_after_validation(monkeypatch, tmp_path):
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    writer.finalize()
    checks = []
    validate = writer._validate_audio
    def validation(*args, **kwargs):
        validate(*args, **kwargs)
        checks.append('validated')
    monkeypatch.setattr(writer, '_validate_audio', validation)
    def cancelled_after_validation():
        if checks:
            raise InterruptedError('cancelled during validation')
    with pytest.raises(InterruptedError):
        writer.commit(checkpoint=cancelled_after_validation)
    assert not (tmp_path / 'final.wav').exists()


def test_assembly_reads_audio_in_bounded_blocks(monkeypatch, tmp_path):
    import builtins
    from base import vkinging_wav
    frames = 20000
    writer = VkingingVoltageWriter(tmp_path / 'large.wav', sample_rate=51200, bit_depth=64,
                                  physical_channels=(15, 0, 8), target_frames=frames,
                                  metadata=metadata(bits=64))
    writer.write_chunk(np.full((frames, 3), 2.5))
    real_open = builtins.open
    read_sizes = []
    class BoundedInput:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, size=-1):
            assert 0 <= size <= 65536
            read_sizes.append(size)
            return self.handle.read(size)
    def open_file(path, mode):
        handle = real_open(path, mode)
        return BoundedInput(handle) if mode == 'rb' else handle
    monkeypatch.setattr(vkinging_wav, 'open', open_file, raising=False)
    writer.finalize()
    assert read_sizes.count(65536) > 1
    assert writer.resources_released


@pytest.mark.parametrize('case', ['physical_duplicate', 'calibrated_not_bool', 'factor_string',
                                 'factor_nan', 'bad_range', 'missing_version'])
def test_metadata_schema_rejects_invalid_values(tmp_path, case):
    data = metadata()
    if case == 'physical_duplicate':
        data['recorded_channels'][1]['physical_channel_index'] = 15
    elif case == 'calibrated_not_bool':
        data['recorded_channels'][0]['calibrated'] = 1
    elif case == 'factor_string':
        data['recorded_channels'][0]['v2pa_factor'] = '2.5'
    elif case == 'factor_nan':
        data['recorded_channels'][0]['v2pa_factor'] = float('nan')
    elif case == 'bad_range':
        data['acquisition']['range_max'] = 9
    elif case == 'missing_version':
        del data['schema_version']
    with pytest.raises(ValueError):
        make_writer(tmp_path, meta=data)
    assert list(tmp_path.iterdir()) == []


def test_inspection_rejects_fact_frame_count_mismatch(tmp_path):
    from pathlib import Path
    writer = make_writer(tmp_path)
    writer.write_chunk(np.ones((2, 3)))
    path = Path(writer.finalize()['path'])
    assert path.stat().st_size == writer.expected_size
    parts = [(name, part[:8] + struct.pack('<I', 999) if name == b'fact' else part)
             for name, part in chunks(path.read_bytes())]
    path.write_bytes(rebuild(parts))
    result = inspect_wav_calibration_metadata(path)
    assert result.source == 'vkinging'
    assert result.status == 'vkinging_invalid'
