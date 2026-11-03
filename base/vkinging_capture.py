"""Synchronous raw-voltage acquisition on the service's sole SDK owner.

The supplied SDK is never constructed or closed here. Only the capture task
and WAV writer belong to this handler; finalization/commit run via service
hooks after that SDK has closed. No audio arrays leave this module.
"""
import ctypes
from pathlib import Path
import sys
import tempfile
import time
import uuid

import numpy as np

from base.vkinging_discovery import discover_vkinging_devices
from base.vkinging_input import resolve_vkinging_input_config
from base.vkinging_service import CaptureResult, _exception_text
from base.vkinging_wav import VkingingVoltageWriter, preflight_vkinging_wav_size
from base.wav_calibration_metadata import normalize_vkinging_metadata
from consts.vkinging_consts import VKINGING_WAV_SCHEMA_VERSION


def _checkpoint(control):
    if control.cancelled:
        raise RuntimeError('capture cancelled')


def _metadata(request):
    acquisition = {'hardware_id': request.device['hardware_id'],
                   'model': request.device.get('model'),
                   **resolve_vkinging_input_config(
                       {'vkinging_range_index': request.range_index}, request.device)}
    if request.purpose == 'main':
        if request.calibration_snapshot is None:
            raise ValueError('main capture requires a frozen calibration_snapshot')
        metadata = normalize_vkinging_metadata(request.calibration_snapshot)
        if (metadata['acquisition'] != acquisition
                or tuple(ch['physical_channel_index'] for ch in metadata['recorded_channels']) != request.channels):
            raise ValueError('calibration_snapshot differs from requested acquisition or channels')
        return metadata
    return normalize_vkinging_metadata({
        'backend': 'vkinging', 'schema_version': VKINGING_WAV_SCHEMA_VERSION,
        'acquisition': acquisition,
        'recorded_channels': [
            {'wav_channel_index': i, 'physical_channel_index': channel,
             'calibrated': False, 'v2pa_factor': None, 'standard_spl': None,
             'calibration_conditions': None}
            for i, channel in enumerate(request.channels)]})


def _scan_prefix(buffer, count, requested, channels):
    if type(count) is not int or not 0 <= count <= requested:
        raise ValueError('read count must be an integer between zero and requested frames')
    if (not isinstance(buffer, ctypes.Array) or buffer._type_ is not ctypes.c_double
            or len(buffer) != requested * channels):
        raise ValueError('read buffer must have exactly R*C native doubles')
    prefix = np.ctypeslib.as_array(buffer)[:count * channels].reshape(count, channels)
    if not np.isfinite(prefix).all():
        raise ValueError('read buffer valid prefix must be finite and fully written')
    return prefix


def _file_hooks(writer):
    def prove_exception_release(control):
        # Each hook opens fresh third-party I/O handles. Its own writer proof,
        # rather than the earlier capture proof, governs exceptional exits.
        if sys.exception() is not None:
            if writer.resources_released:
                control.confirm_released()
            else:
                control.report_cleanup_failure(
                    f'WAV release uncertain; owned paths: {writer.owned_paths}')

    def commit(control):
        try:
            _checkpoint(control)
            return writer.commit(checkpoint=lambda: _checkpoint(control))
        finally:
            prove_exception_release(control)

    def finalize(control):
        try:
            _checkpoint(control)
            descriptor = writer.finalize(checkpoint=lambda: _checkpoint(control))
            _checkpoint(control)
            return CaptureResult(descriptor, resources_released=writer.resources_released, commit=commit)
        finally:
            prove_exception_release(control)

    return finalize


def capture_vkinging(sdk, request, control, *, clock=time.monotonic,
                     block_frames=None, temp_directory=None):
    """Capture exact per-channel frames, returning descriptor-only file hooks.

    clock/block_frames/temp_directory are injection points for deterministic
    tests. Defaults bound reads to 0.1 seconds/4096 frames with a <=0.2 second
    native timeout. Deadlines are checked without needing a GUI tick.
    """
    task = 'vkinging_' + uuid.uuid4().hex
    created, writer, writer_attempted = False, None, False
    cleanup_errors = []
    try:
        _checkpoint(control)
        control.set_phase('starting')
        metadata = _metadata(request)
        if request.purpose == 'main' and request.output_path is None:
            raise ValueError('main capture requires output_path')
        channels = len(request.channels)
        frames = min(4096, max(1, request.sample_rate // 10)) if block_frames is None else block_frames
        if (type(frames) is not int or not 1 <= frames < 2**31
                or frames * channels >= 2**31
                or frames * channels > sys.maxsize // ctypes.sizeof(ctypes.c_double)):
            raise ValueError('read buffer capacity/count exceeds native bounds')
        preflight_vkinging_wav_size(
            sample_rate=request.sample_rate, bit_depth=request.bit_depth, channels=channels,
            target_frames=request.target_samples, metadata=metadata)
        discovery = discover_vkinging_devices(sdk)
        matches = [device for device in discovery.devices
                   if device['hardware_id'] == request.device['hardware_id']]
        if len(matches) != 1:
            raise ValueError(f'current device identity unavailable: {request.device["hardware_id"]}; '
                             + '; '.join(discovery.diagnostics))
        current = matches[0]
        if not set(request.channels).issubset(current['physical_channels']):
            raise ValueError('selected physical channels no longer available')
        routes = ','.join(f'{current["name"]}/AIN{channel + 1}' for channel in request.channels)
        _checkpoint(control)
        output = request.output_path
        if output is None:
            # UUID destination is absent; writer creates its own exclusive
            # partials. Calibration therefore requires no GUI-created file.
            output = Path(temp_directory or tempfile.gettempdir()) / ('vkinging-calibration-' + uuid.uuid4().hex + '.wav')
        writer_attempted = True
        writer = VkingingVoltageWriter(
            output, sample_rate=request.sample_rate, bit_depth=request.bit_depth,
            physical_channels=request.channels, target_frames=request.target_samples, metadata=metadata)
        _checkpoint(control)
        sdk.create_task(task)
        created = True
        sdk.create_iepe_voltage_channel(task, routes,
                                       range_min=metadata['acquisition']['range_min'],
                                       range_max=metadata['acquisition']['range_max'])
        sdk.configure_sample_clock(task, request.sample_rate)
        _checkpoint(control)
        sdk.start_task(task)
        sdk.verify_actual_sample_rate(current['name'], request.sample_rate)
        _checkpoint(control)
        control.set_phase('capturing')
        started = last_progress = clock()
        budget = (request.target_samples + request.discard_initial_samples) / request.sample_rate + 5
        raw_frames = kept = 0

        def check_deadline():
            _checkpoint(control)
            now = clock()
            if now - started >= budget:
                raise TimeoutError('capture_timeout: overall acquisition deadline exceeded')
            if now - last_progress >= 3:
                raise TimeoutError('read_stall: no frames for three seconds')

        while kept < request.target_samples:
            check_deadline()
            buffer, count = sdk.read_task_data(
                task, channel_count=channels, samples_per_channel=frames, timeout_seconds=.1)
            check_deadline()
            values = _scan_prefix(buffer, count, frames, channels)
            if not count:
                continue
            skip = min(count, max(0, request.discard_initial_samples - raw_frames))
            raw_frames += count
            last_progress = clock()
            control.report_progress(raw_frames)
            take = min(count - skip, request.target_samples - kept)
            if take:
                writer.write_chunk(values[skip:skip + take])
                kept += take
        check_deadline()
    finally:
        # SDK calls and writer.close are external contract boundaries. Every
        # release is attempted; errors stay diagnosable without replacing a
        # primary capture failure, and ANY failed release blocks reuse.
        primary = sys.exception()
        if created:
            control.set_phase('stopping')
            for operation in ('stop_task', 'clear_task'):
                try:
                    getattr(sdk, operation)(task)
                except Exception as exc:
                    diagnostic = f'{operation}: {_exception_text(exc)}'
                    cleanup_errors.append(diagnostic)
                    control.report_cleanup_failure(diagnostic)
                    if primary is not None:
                        primary.add_note(diagnostic)
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                diagnostic = f'WAV close: {_exception_text(exc)}; owned paths: {writer.owned_paths}'
                cleanup_errors.append(diagnostic)
                control.report_cleanup_failure(diagnostic)
                if primary is not None:
                    primary.add_note(diagnostic)
        released = (not cleanup_errors and (writer.resources_released if writer is not None else not writer_attempted))
        if primary is not None and released:
            control.confirm_released()
    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors))
    return CaptureResult(resources_released=released, finalize=_file_hooks(writer))
