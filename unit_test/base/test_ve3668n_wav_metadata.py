"""VE voltage/provenance contracts; all WAVs are disposable FLOAT fixtures."""
from copy import deepcopy
import io
import json
import struct
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from base import wav_calibration_metadata as wav_metadata_io
from base.recording_process_protocol import FrozenConfig
from consts.ve3668n_consts import VE_RANGE_LIMITS
from unit_test.base.ve3668n_fakes import wav_metadata
from unit_test.base.test_wav_calibration_metadata import (
    _append_raw_chunk, _chunk, _install_bounded_open, _metadata_list_chunk,
)


@pytest.mark.parametrize("sources", [("none", "none"), ("measured", "measured"),
                                     ("measured", "none")])
@pytest.mark.parametrize("limit", VE_RANGE_LIMITS)
def test_none_measured_mixed_round_trip_preserves_voltage_and_original_rate(tmp_path, sources, limit):
    payload = wav_metadata(sources)
    payload["acquisition"].update(range_min=-limit, range_max=limit)
    path = tmp_path / "voltage.wav"
    audio = np.array([[2.5, -7.0], [-4.0, 1.5], [0.125, 9.0]], dtype=np.float32)
    wavfile.write(path, 44100, audio)  # IEEE FLOAT format 3, not Python wave PCM.
    original = path.read_bytes()

    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) == payload
    assert wav_metadata_io.append_wav_calibration_metadata(path, payload)
    assert wav_metadata_io.read_wav_calibration_metadata(path) == payload
    assert path.read_bytes()[8:len(original)] == original[8:]
    rate, actual = wavfile.read(path)
    assert rate == 44100
    np.testing.assert_array_equal(actual, audio)
    for channel in payload["recorded_channels"]:
        if channel["calibrated"]:
            assert channel["v2pa_factor"] == 10.0
            assert channel["calibration"]["sample_rate"] == 51200


def _section(payload, section):
    if section == "root":
        return payload
    if section == "acquisition":
        return payload[section]
    channel = payload["recorded_channels"][0]
    return channel if section == "channel" else channel["calibration"]


@pytest.mark.parametrize("section", ["root", "acquisition", "channel", "calibration"])
@pytest.mark.parametrize("field", ["unknown", "sensitivity", "nominal_factor"])
def test_schema_rejects_unknown_fields_at_every_level(section, field):
    payload = wav_metadata()
    _section(payload, section)[field] = 1000.0
    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) is None


@pytest.mark.parametrize("section,field", [
    (section, field)
    for section in ("root", "acquisition", "channel", "calibration")
    for field in _section(wav_metadata(), section)
    if field != "backend"  # A missing marker intentionally retains legacy semantics.
])
def test_schema_requires_every_declared_field(section, field):
    payload = wav_metadata()
    del _section(payload, section)[field]
    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) is None


@pytest.mark.parametrize("section,field,value", [
    *(('root', 'schema_version', value) for value in (0, 2, True, 1.0, '1', None)),
    *(('root', 'recorded_channels', value) for value in (None, [], {}, 'channels')),
    *(('acquisition', field, value) for field, value in (
        ('model', 'VE3668N-extra'), ('model', None), ('machine_id', ''),
        ('machine_id', True), ('input_mode', 'voltage'), ('unit', 'Pa'),
        ('range_min', -1.0), ('range_max', '10'), ('range_max', float('inf')),
    )),
    *((section, 'sample_rate', value) for section in ('acquisition', 'calibration')
      for value in (0, -1, True, False, 44100.0, '51200', None)),
    *(('channel', 'wav_channel_index', value) for value in (-1, True, 0.0, '0', 2)),
    *(('channel', 'physical_input_channel', value) for value in (-1, 8, True, 7.0, '7')),
    *(('channel', 'v2pa_factor', value) for value in (
        None, 0, -10, float('nan'), float('inf'), float('-inf'), True, '10', 10**400,
    )),
    *(('channel', 'calibrated', value) for value in (False, 1, 'true', None)),
    *(('channel', 'factor_source', value) for value in ('none', 'nominal', None, True)),
    ('channel', 'calibration', None),
    *(('calibration', 'standard_spl', value) for value in (None, True, '94', float('nan'))),
    *(('calibration', 'duration_seconds', value) for value in (
        None, 0, -1, True, '10', float('inf'),
    )),
    *(('calibration', 'calibrated_at', value) for value in (
        None, True, '', 'not-a-time', '2026-08-28', '2026-08-28T10:00:00',
    )),
])
def test_schema_rejects_invalid_types_values_and_source_combinations(section, field, value):
    payload = wav_metadata()
    _section(payload, section)[field] = value
    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) is None


@pytest.mark.parametrize("field,value", [
    ('calibrated', True), ('calibrated', 0), ('calibrated', None),
    ('v2pa_factor', 1.0), ('v2pa_factor', False),
    ('calibration', {}), ('calibration', False), ('factor_source', 'measured'),
])
def test_none_is_exactly_false_null_null(field, value):
    payload = wav_metadata(('none', 'none'))
    payload['recorded_channels'][0][field] = value
    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) is None


@pytest.mark.parametrize("field", ['wav_channel_index', 'physical_input_channel'])
def test_duplicate_channel_identity_is_rejected(field):
    payload = wav_metadata()
    payload['recorded_channels'][1][field] = payload['recorded_channels'][0][field]
    assert wav_metadata_io.normalize_wav_calibration_metadata(payload) is None


@pytest.mark.parametrize("capture_rate,calibration_rate", [(4000, 192000), (192000, 4000)])
def test_historical_positive_rates_are_not_new_acquisition_whitelist(tmp_path, capture_rate,
                                                                  calibration_rate):
    payload = wav_metadata(sample_rate=capture_rate)
    payload['recorded_channels'][0]['calibration']['sample_rate'] = calibration_rate
    path = tmp_path / 'historical.wav'
    wavfile.write(path, capture_rate, np.zeros((8, 2), dtype=np.float32))
    assert wav_metadata_io.append_wav_calibration_metadata(path, payload)
    assert wav_metadata_io.read_wav_calibration_metadata(path) == payload


def test_frozen_mapping_is_serializable_and_normalized_snapshots_are_independent(tmp_path):
    from base.ve3668n_wav_metadata import validate_ve_wav_metadata

    payload = wav_metadata()
    expected = deepcopy(payload)
    frozen = FrozenConfig.snapshot(payload)
    normalized = wav_metadata_io.normalize_wav_calibration_metadata(frozen)
    assert normalized == expected
    assert json.loads(json.dumps(validate_ve_wav_metadata(frozen), allow_nan=False)) == expected
    payload['acquisition']['sample_rate'] = 48000
    normalized['recorded_channels'][0]['calibration']['sample_rate'] = 22050
    assert frozen.to_dict() == expected
    path = tmp_path / 'frozen.wav'
    wavfile.write(path, 44100, np.zeros((8, 2), dtype=np.float32))
    assert wav_metadata_io.append_wav_calibration_metadata(path, frozen)
    assert wav_metadata_io.read_wav_calibration_metadata(path) == expected


def test_ve_factor_resolution_is_file_local_and_explicit():
    from base import ve3668n_wav_metadata

    payload = FrozenConfig.snapshot(wav_metadata())
    measured = ve3668n_wav_metadata.resolve_ve_wav_channel_v2pa_factor(payload, 0)
    assert measured.factor == 10.0
    assert measured.state == 'measured'
    none = ve3668n_wav_metadata.resolve_ve_wav_channel_v2pa_factor(payload, 1)
    assert none.factor == 1.0
    assert none.state == 'none'
    assert none.diagnostic
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(payload, 0).factor == 10.0
    resolution = wav_metadata_io.resolve_wav_channel_v2pa_factor(payload, 1)
    assert resolution.factor == 1.0
    assert resolution.has_valid_metadata
    assert not resolution.used_file_metadata
    assert payload.to_dict() == wav_metadata()


@pytest.mark.parametrize('index', [2, -1, True, 0.0, '0'])
def test_known_ve_invalid_channel_never_uses_legacy_index_coercion_or_factor_one(index):
    from base import ve3668n_wav_metadata

    payload = wav_metadata()
    result = ve3668n_wav_metadata.resolve_ve_wav_channel_v2pa_factor(payload, index)
    assert result.factor is None
    assert result.state == 'invalid'
    with pytest.raises(ValueError, match='VE.*invalid'):
        wav_metadata_io.resolve_wav_channel_v2pa_factor(payload, index)


def test_known_invalid_ve_never_falls_back_to_one_even_without_trusted_metadata():
    payload = wav_metadata()
    payload['schema_version'] = 99
    with pytest.raises(ValueError, match='VE.*invalid'):
        wav_metadata_io.resolve_wav_channel_v2pa_factor(payload, 0)
    result = wav_metadata_io.WavCalibrationMetadataReadResult(
        wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID, None,
        declared_backend='vkinging',
    )
    with pytest.raises(ValueError, match='VE.*invalid'):
        wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0)


def test_read_result_two_argument_construction_remains_legacy_compatible():
    result = wav_metadata_io.WavCalibrationMetadataReadResult(
        wav_metadata_io.WavCalibrationMetadataReadStatus.ABSENT, None,
    )
    assert result.declared_backend is None
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(None, 0).factor == 1.0


def _float_wav(tmp_path):
    path = tmp_path / 'metadata.wav'
    wavfile.write(path, 44100, np.zeros((8, 2), dtype=np.float32))
    return path


def _assert_invalid_ve(path):
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID
    assert result.metadata is None
    assert result.declared_backend == 'vkinging'
    with pytest.raises(ValueError, match='VE.*invalid'):
        wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0)


@pytest.mark.parametrize('raw', [
    b'{"backend":"vkinging",',
    b'{"backend":"vkinging", "bad":\xff}',
    b'{"bad":\xff, "backend" : "vkinging"}',
    b'{"backend":"vkinging","schema_version":99}',
    b'{"backend":"vkinging","backend":"sounddevice","recorded_channels":[]}',
    b'{"back\\u0065nd":"vkin\\u0067ing",',
])
@pytest.mark.parametrize('legacy_position', ['before', 'after', 'same_list_before', 'same_list_after'])
def test_invalid_ve_hint_cannot_be_masked_by_valid_legacy_comments(tmp_path, raw, legacy_position):
    path = _float_wav(tmp_path)
    legacy = json.dumps({'recorded_channels': [
        {'wav_channel_index': 0, 'calibrated': False},
    ]}).encode('utf-8')
    comments = [raw, legacy] if legacy_position.endswith('after') else [legacy, raw]
    if legacy_position.startswith('same_list'):
        _append_raw_chunk(path, _chunk(b'LIST', b'INFO' + b''.join(
            _chunk(b'ICMT', b'mic_calibration=' + comment + b'\0') for comment in comments
        )))
    else:
        for comment in comments:
            _append_raw_chunk(path, _metadata_list_chunk(comment))
    _assert_invalid_ve(path)


@pytest.mark.parametrize('order', ['before', 'after'])
def test_valid_ve_provenance_is_not_replaced_by_legacy_none(tmp_path, order):
    path = _float_wav(tmp_path)
    ve = _metadata_list_chunk(json.dumps(wav_metadata()).encode())
    legacy = _metadata_list_chunk(b'{"recorded_channels":[{"wav_channel_index":0}]}')
    for chunk in ((legacy, ve) if order == 'before' else (ve, legacy)):
        _append_raw_chunk(path, chunk)
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.declared_backend == 'vkinging'
    assert result.metadata == wav_metadata()


@pytest.mark.parametrize('corruption', ['chunk_overrun', 'list_overrun', 'bad_padding', 'trailing'])
def test_recognized_ve_survives_later_scanner_structure_errors(tmp_path, corruption):
    path = _float_wav(tmp_path)
    ve_comment = _chunk(b'ICMT', b'mic_calibration=' + json.dumps(wav_metadata()).encode())
    if corruption == 'list_overrun':
        chunk = _chunk(b'LIST', b'INFO' + ve_comment + b'ICMT' + struct.pack('<I', 500) + b'x')
    else:
        chunk = _chunk(b'LIST', b'INFO' + ve_comment)
        chunk += {
            'chunk_overrun': b'JUNK' + struct.pack('<I', 100) + b'x',
            'bad_padding': _chunk(b'JUNK', b'x', padding=b'!'),
            'trailing': b'x',
        }[corruption]
    _append_raw_chunk(path, chunk)
    _assert_invalid_ve(path)


@pytest.mark.parametrize('corruption', ['oversized_riff', 'truncated_tail', 'truncated_comment'])
@pytest.mark.parametrize('backend', ['vkinging', None])
def test_invalid_riff_end_retains_only_reachable_backend_provenance(tmp_path, corruption, backend):
    path = _float_wav(tmp_path)
    payload = wav_metadata() if backend else {'recorded_channels': [
        {'wav_channel_index': 0, 'calibrated': True, 'v2pa_factor': 10.0},
    ]}
    comment_start = len(path.read_bytes())
    _append_raw_chunk(path, _metadata_list_chunk(json.dumps(payload).encode()))
    intact = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert intact.status is wav_metadata_io.WavCalibrationMetadataReadStatus.VALID
    assert intact.declared_backend == backend
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(intact.metadata, 0).factor == 10.0

    if corruption == 'truncated_tail':
        _append_raw_chunk(path, _chunk(b'JUNK', b'abcd'))
        damaged = path.read_bytes()[:-2]
    else:
        original = path.read_bytes()
        if corruption == 'oversized_riff':
            damaged = original[:4] + struct.pack('<I', len(original) - 8 + 100) + original[8:]
        else:
            marker = b'"backend": "vkinging"' if backend else b'"recorded_channels":'
            damaged = original[:original.index(marker, comment_start) + len(marker)]
    path.write_bytes(damaged)

    if backend:
        _assert_invalid_ve(path)
    else:
        result = wav_metadata_io.inspect_wav_calibration_metadata(path)
        assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID
        assert result.metadata is None
        assert result.declared_backend is None
        assert wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0).factor == 1.0
    assert wav_metadata_io.read_wav_calibration_metadata(path) is None
    assert not wav_metadata_io.append_wav_calibration_metadata(path, payload)
    assert path.read_bytes() == damaged


@pytest.mark.parametrize('comment_size', [90000, 1024 * 1024 + 1, 0xFFFFFFFF])
@pytest.mark.parametrize('hint_in_probe', [True, False])
def test_invalid_riff_end_probes_only_bounded_comment_provenance(tmp_path, monkeypatch,
                                                              comment_size, hint_in_probe):
    path = _float_wav(tmp_path)
    body = b'{"backend":"vkinging",'
    if not hint_in_probe:
        body = b' ' * 65536 + body
    comment = b'mic_calibration=' + body
    comment += b'x' * (min(comment_size, 100000) - len(comment))
    if comment_size == 0xFFFFFFFF:
        chunk = (b'LIST' + struct.pack('<I', 0xFFFFFFFF) + b'INFO'
                 + b'ICMT' + struct.pack('<I', comment_size) + comment)
    else:
        comment += b'x' * (comment_size - len(comment))
        chunk = _chunk(b'LIST', b'INFO' + _chunk(b'ICMT', comment))
    _append_raw_chunk(path, chunk)
    original = path.read_bytes()
    path.write_bytes(original[:4] + struct.pack('<I', 0xFFFFFFFF) + original[8:])
    wrappers = _install_bounded_open(monkeypatch, path)

    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID
    assert result.metadata is None
    assert result.declared_backend == ('vkinging' if hint_in_probe else None)
    assert sum(sum(wrapper.read_sizes) for wrapper in wrappers) < 70000


@pytest.mark.parametrize('location', ['audio', 'unprefixed', 'other_list', 'nested', 'quoted',
                                     'outside_riff', 'undersized_riff'])
def test_invalid_riff_end_never_scans_audio_or_out_of_bounds_markers(monkeypatch, location):
    hint = _metadata_list_chunk(b'{"backend":"vkinging",')
    chunks = _chunk(b'fmt ', struct.pack('<HHIIHH', 3, 2, 44100, 352800, 8, 32))
    audio = hint if location == 'audio' else b'\0' * 16
    audio_start = 12 + len(chunks) + 8
    chunks += _chunk(b'data', audio)
    if location == 'unprefixed':
        chunks += _chunk(b'LIST', b'INFO' + _chunk(b'ICMT', b'{"backend":"vkinging",'))
    elif location == 'other_list':
        chunks += _chunk(b'LIST', b'adtl' + _chunk(b'ICMT', b'mic_calibration={"backend":"vkinging",'))
    elif location == 'nested':
        chunks += _metadata_list_chunk(b'{"nested":{"backend":"vkinging"},')
    elif location == 'quoted':
        chunks += _metadata_list_chunk(json.dumps({'example': '{"backend":"vkinging"}'}).encode())
    # An incomplete chunk header also keeps the shorter declared-RIFF control invalid.
    chunks += b'JUNK'
    if location == 'outside_riff':
        riff_end = 12 + len(chunks)
        chunks += hint
    elif location == 'undersized_riff':
        riff_end = 10
        chunks += hint
    else:
        riff_end = 12 + len(chunks) + 100
    raw = b'RIFF' + struct.pack('<I', riff_end - 8) + b'WAVE' + chunks

    class GuardedReader(io.BytesIO):
        def read(self, size=-1):
            start = self.tell()
            assert 0 <= size <= 65536
            assert start + size <= max(12, min(riff_end, len(raw)))
            assert not (start < audio_start + len(audio) and start + size > audio_start)
            return super().read(size)

    reader = GuardedReader(raw)
    monkeypatch.setattr(wav_metadata_io, 'open', lambda *args, **kwargs: reader, raising=False)
    result = wav_metadata_io.inspect_wav_calibration_metadata('memory.wav')
    assert reader.closed
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID
    assert result.metadata is None
    assert result.declared_backend is None
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0).factor == 1.0


def test_oversized_ve_comment_only_probes_bounded_hint_and_cannot_fall_back(tmp_path, monkeypatch):
    path = _float_wav(tmp_path)
    oversized = b'mic_calibration={"backend":"vkinging",' + b'x' * (2 * 1024 * 1024)
    _append_raw_chunk(path, _chunk(b'LIST', b'INFO' + _chunk(b'ICMT', oversized)))
    _append_raw_chunk(path, _metadata_list_chunk(
        b'{"recorded_channels":[{"wav_channel_index":0}]}'
    ))
    wrappers = _install_bounded_open(monkeypatch, path)
    _assert_invalid_ve(path)
    assert sum(sum(wrapper.read_sizes) for wrapper in wrappers) < 70000


def test_large_but_allowed_comment_uses_small_reads(tmp_path, monkeypatch):
    path = _float_wav(tmp_path)
    raw = b' ' * 70000 + json.dumps(wav_metadata()).encode()
    _append_raw_chunk(path, _metadata_list_chunk(raw))
    wrappers = _install_bounded_open(monkeypatch, path)
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.metadata == wav_metadata()
    assert result.declared_backend == 'vkinging'
    assert all(size <= 65536 for wrapper in wrappers for size in wrapper.read_sizes)


@pytest.mark.parametrize('body,recognized', [
    (rb'{"example":"\"backend\":\"vkinging\""}', False),
    (rb'{"nested":{"backend":"vkinging"}}', False),
    (rb'{"nested":[{"backend":"vkinging"}]}', False),
    (rb'[{"backend":"vkinging"}]', False),
    (rb'{"backend":["vkinging"]}', False),
    (rb'{"backend":{"backend":"vkinging"}}', False),
    (rb'{"example":"quotes \" and backslash \\","back\u0065nd":"vkin\u0067ing"}', True),
    (rb'{"example":"\"backend\":\"vkinging\"', False),
    (rb'{"example":"\"backend\":\"vkinging\"' + b'\\', False),
    (b'{"example":"trailing escape' + b'\\', False),
    (rb'{"backend":"vkinging","example":"\"unfinished', True),
    (rb'{"back\u0065nd":"vkin\u0067ing","example":"\"' + b'\\', True),
    (b'\xff' + rb'{"back\u0065nd":"vkin\u0067ing"}', True),
    (b'{"example":"\xff","backend":"vkinging"}', True),
    (b'{"\xff":"bad key","backend":"vkinging"}', True),
    (b'{"backend":"vkinging\xff"}', False),
    (b'{"back\xffend":"vkinging"}', False),
])
def test_backend_hint_respects_escaped_json_string_boundaries(tmp_path, body, recognized):
    path = _float_wav(tmp_path)
    _append_raw_chunk(path, _metadata_list_chunk(body))
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.INVALID
    assert result.metadata is None
    assert result.declared_backend == ('vkinging' if recognized else None)
    if recognized:
        with pytest.raises(ValueError, match='VE.*invalid'):
            wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0)
    else:
        assert wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0).factor == 1.0


@pytest.mark.parametrize('backend_last', [False, True])
def test_complete_ve_json_with_escaped_strings_retains_exact_snapshot(tmp_path, backend_last):
    path = _float_wav(tmp_path)
    payload = wav_metadata()
    payload['acquisition']['machine_id'] = '"quoted" \\ device' * 2048
    if backend_last:
        payload['backend'] = payload.pop('backend')
    body = json.dumps(payload).encode().replace(
        b'"backend"', rb'"back\u0065nd"',
    ).replace(b'"vkinging"', rb'"vkin\u0067ing"')
    _append_raw_chunk(path, _metadata_list_chunk(body))
    original = path.read_bytes()
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.VALID
    assert result.declared_backend == 'vkinging'
    assert result.metadata == payload
    assert path.read_bytes() == original


@pytest.mark.parametrize('key,escaped_key', [
    (b'"backend"', rb'"back\u0065nd"'),
    (b'"machine_id"', rb'"machine\u005fid"'),
])
def test_escaped_duplicate_ve_keys_remain_invalid(tmp_path, key, escaped_key):
    path = _float_wav(tmp_path)
    body = json.dumps(wav_metadata()).encode()
    body = body.replace(key + b':', key + b': "ignored", ' + escaped_key + b':')
    _append_raw_chunk(path, _metadata_list_chunk(body))
    _assert_invalid_ve(path)


def test_quoted_backend_example_preserves_legacy_duplicate_key_normalization(tmp_path):
    path = _float_wav(tmp_path)
    body = (
        rb'{"example":"\"backend\":\"vkinging\"","nested":{"backend":"vkinging"},'
        rb'"recorded_channels":[{"wav_channel_index":"0","calibrated":true,'
        rb'"v2pa_factor":"2","v2pa_factor":"3.5"}]}'
    )
    _append_raw_chunk(path, _metadata_list_chunk(body))
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.VALID
    assert result.declared_backend is None
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(result.metadata, '0').factor == 3.5


@pytest.mark.parametrize('corruption', ['ordinary', 'oversized', 'truncated_riff',
                                      'truncated_comment'])
def test_escaped_string_corruption_is_time_bounded(tmp_path, corruption):
    path = _float_wav(tmp_path)
    # An unclosed string used to restart the regex at every escaped quote,
    # taking over ten seconds even for a single bounded 64 KiB hint probe.
    body = b'{"x":"' + b'\\"' * 32750 + b'\\'
    if corruption == 'oversized':
        body += b'x' * wav_metadata_io.MAX_CALIBRATION_COMMENT_SIZE
    if corruption == 'truncated_comment':
        comment = b'mic_calibration=' + body
        chunk = _chunk(b'LIST', b'INFO' + b'ICMT' + struct.pack('<I', 0xFFFFFFFF) + comment)
    else:
        chunk = _metadata_list_chunk(body)
    _append_raw_chunk(path, chunk)
    if corruption == 'truncated_riff':
        raw = path.read_bytes()
        path.write_bytes(raw[:4] + struct.pack('<I', len(raw) + 100) + raw[8:])
    original = path.read_bytes()

    # Isolate the CPU-bound parser so a regression cannot hang the test runner.
    # Ten seconds includes interpreter startup and gives the linear scan ample
    # headroom; this is a hang guard, not a millisecond performance assertion.
    script = (
        'import json, sys\n'
        'from base.wav_calibration_metadata import inspect_wav_calibration_metadata\n'
        'result = inspect_wav_calibration_metadata(sys.argv[1])\n'
        'print(json.dumps([result.status.value, result.metadata, result.declared_backend]))\n'
    )
    try:
        completed = subprocess.run(
            [sys.executable, '-B', '-X', 'utf8', '-c', script, str(path)],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f'{corruption} escaped-string metadata scan exceeded 10 seconds', pytrace=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ['invalid', None, None]
    assert path.read_bytes() == original


@pytest.mark.parametrize('location', ['audio', 'outside_riff', 'unprefixed', 'other_list', 'nested'])
def test_backend_hint_is_never_guessed_outside_prefixed_metadata(tmp_path, location):
    path = _float_wav(tmp_path)
    hint = b'mic_calibration={"backend":"vkinging",'
    if location == 'audio':
        _append_raw_chunk(path, _chunk(b'data', hint))
    elif location == 'outside_riff':
        path.write_bytes(path.read_bytes() + _chunk(b'LIST', b'INFO' + _chunk(b'ICMT', hint)))
    elif location == 'unprefixed':
        _append_raw_chunk(path, _chunk(b'LIST', b'INFO' + _chunk(b'ICMT', hint[16:])))
    elif location == 'other_list':
        _append_raw_chunk(path, _chunk(b'LIST', b'adtl' + _chunk(b'ICMT', hint)))
    else:
        _append_raw_chunk(path, _metadata_list_chunk(b'{"nested":{"backend":"vkinging"},'))
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.declared_backend is None


@pytest.mark.parametrize('field,value', [('sample_rate', 48000), ('channels', 1)])
def test_ve_header_mismatch_rejects_import_and_append_without_mutating_original(tmp_path, field, value):
    path = _float_wav(tmp_path)
    payload = wav_metadata()
    if field == 'sample_rate':
        payload['acquisition'][field] = value
    else:
        payload['recorded_channels'] = payload['recorded_channels'][:value]
    original = path.read_bytes()
    assert not wav_metadata_io.append_wav_calibration_metadata(path, payload)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('.*.tmp'))
    _append_raw_chunk(path, _metadata_list_chunk(json.dumps(payload).encode()))
    _assert_invalid_ve(path)


@pytest.mark.parametrize('fmt_case', ['missing', 'short', 'duplicate', 'after_metadata'])
def test_ve_fmt_header_is_bounded_and_required_even_when_after_metadata(tmp_path, fmt_case):
    path = _float_wav(tmp_path)
    original = path.read_bytes()
    # scipy writes fmt first; rebuild only these disposable test containers.
    fmt_size = struct.unpack('<I', original[16:20])[0]
    fmt_end = 20 + fmt_size + fmt_size % 2
    fmt = original[12:fmt_end]
    tail = original[fmt_end:]
    metadata = _metadata_list_chunk(json.dumps(wav_metadata()).encode())
    chunks = {
        'missing': tail + metadata,
        'short': _chunk(b'fmt ', b'\x03\x00') + tail + metadata,
        'duplicate': fmt + fmt + tail + metadata,
        'after_metadata': tail + metadata + fmt,
    }[fmt_case]
    path.write_bytes(b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WAVE' + chunks)
    if fmt_case == 'after_metadata':
        assert wav_metadata_io.inspect_wav_calibration_metadata(path).metadata == wav_metadata()
    else:
        _assert_invalid_ve(path)


@pytest.mark.parametrize('prefix', [b'\xff', b'\xef\xbb\xbf'])
def test_recognizable_backend_hint_survives_invalid_bytes_before_json(tmp_path, prefix):
    path = _float_wav(tmp_path)
    _append_raw_chunk(path, _metadata_list_chunk(prefix + b'{"backend":"vkinging",'))
    _assert_invalid_ve(path)


@pytest.mark.parametrize('boundary', ['list', 'riff', 'padding'])
def test_corrupt_comment_sizes_preserve_hint_only_inside_declared_bounds(tmp_path, boundary):
    path = _float_wav(tmp_path)
    body = b'mic_calibration={"backend":"vkinging",'
    if boundary == 'list':
        chunk = _chunk(b'LIST', b'INFO' + b'ICMT' + struct.pack('<I', 999999) + body)
    elif boundary == 'riff':
        chunk = b'LIST' + struct.pack('<I', 999999) + b'INFO' + _chunk(b'ICMT', body)
    else:
        body += b'x' if len(body) % 2 == 0 else b''
        chunk = _chunk(b'LIST', b'INFO' + b'ICMT' + struct.pack('<I', len(body)) + body)
    _append_raw_chunk(path, chunk)
    _assert_invalid_ve(path)


@pytest.mark.parametrize('legacy_last', [False, True])
def test_unrecognized_oversized_legacy_comment_retains_last_valid_compatibility(tmp_path,
                                                                           monkeypatch,
                                                                           legacy_last):
    path = _float_wav(tmp_path)
    legacy_payload = {'recorded_channels': [
        {'wav_channel_index': '0', 'calibrated': True, 'v2pa_factor': '3.5'},
    ]}
    legacy = _metadata_list_chunk(json.dumps(legacy_payload).encode())
    oversized = _metadata_list_chunk(b'x' * (2 * 1024 * 1024))
    for chunk in ((oversized, legacy) if legacy_last else (legacy, oversized)):
        _append_raw_chunk(path, chunk)
    _install_bounded_open(monkeypatch, path)
    result = wav_metadata_io.inspect_wav_calibration_metadata(path)
    assert result.declared_backend is None
    assert result.status is wav_metadata_io.WavCalibrationMetadataReadStatus.VALID
    assert wav_metadata_io.resolve_wav_channel_v2pa_factor(result.metadata, '0').factor == 3.5


@pytest.mark.parametrize('invalid_first', [True, False])
@pytest.mark.parametrize('invalid_kind', ['header', 'schema'])
def test_invalid_ve_cannot_be_masked_even_by_another_valid_ve_comment(tmp_path, invalid_first,
                                                                 invalid_kind):
    path = _float_wav(tmp_path)
    valid = wav_metadata()
    invalid = wav_metadata()
    if invalid_kind == 'header':
        invalid['acquisition']['sample_rate'] = 48000
    else:
        invalid['schema_version'] = 2
    for payload in ((invalid, valid) if invalid_first else (valid, invalid)):
        _append_raw_chunk(path, _metadata_list_chunk(json.dumps(payload).encode()))
    _assert_invalid_ve(path)


@pytest.mark.parametrize('source', ['measured', 'none'])
def test_two_argument_read_result_with_ve_payload_is_still_known_ve(source):
    result = wav_metadata_io.WavCalibrationMetadataReadResult(
        wav_metadata_io.WavCalibrationMetadataReadStatus.VALID, wav_metadata((source,)),
    )
    resolution = wav_metadata_io.resolve_wav_channel_v2pa_factor(result, 0)
    assert resolution.factor == (10.0 if source == 'measured' else 1.0)
    assert resolution.has_valid_metadata
    assert resolution.used_file_metadata is (source == 'measured')


@pytest.mark.parametrize('stage', ['source', 'temporary', 'validation'])
def test_ve_append_close_failures_preserve_original_and_report_uncertain_ownership(tmp_path,
                                                                              monkeypatch, stage):
    from unit_test.base.recording_process_fakes import MetadataFileFaults

    path = _float_wav(tmp_path)
    original = path.read_bytes()
    faults = MetadataFileFaults(stage)
    faults.install(monkeypatch)
    try:
        result = wav_metadata_io.append_wav_calibration_metadata_result(path, wav_metadata())
        assert not result.appended and not result.handles_released
        assert result.cleanup_paths == tuple(faults.temporary_paths)
        assert result.retained_handles and result.close_errors
        assert path.read_bytes() == original
    finally:
        faults.release_all()


@pytest.mark.parametrize('boundary,oversized_riff', [
    ('read', False), ('close', False), ('comment_read', False), ('read', True), ('close', True),
])
def test_io_error_after_ve_identification_retains_backend_and_closes_reader(tmp_path, monkeypatch,
                                                                       boundary, oversized_riff):
    path = _float_wav(tmp_path)
    body = json.dumps(wav_metadata()).encode()
    if boundary == 'comment_read':
        body += b' ' * 70000
    _append_raw_chunk(path, _metadata_list_chunk(body))
    original_end = len(path.read_bytes())
    _append_raw_chunk(path, _chunk(b'JUNK', b'\0\0'))
    if oversized_riff:
        original = path.read_bytes()
        path.write_bytes(original[:4] + struct.pack('<I', len(original) - 8 + 100) + original[8:])
    raw_open = open
    readers = []
    errors = []

    class FailingReader:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def read(self, size):
            if boundary == 'read' and self.raw.tell() >= original_end:
                errors.append(boundary)
                raise OSError('simulated read failure after VE')
            if boundary == 'comment_read' and self.raw.tell() >= 65536:
                errors.append(boundary)
                raise OSError('simulated mid-comment read failure after VE hint')
            return self.raw.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.raw.close()
            if boundary == 'close':
                errors.append(boundary)
                raise OSError('simulated close failure after VE')

    def failing_open(file, mode='r', *args, **kwargs):
        raw = raw_open(file, mode, *args, **kwargs)
        readers.append(raw)
        return FailingReader(raw)

    monkeypatch.setattr('base.wav_calibration_metadata.open', failing_open, raising=False)
    _assert_invalid_ve(path)
    assert errors == [boundary]
    assert readers and all(reader.closed for reader in readers)
