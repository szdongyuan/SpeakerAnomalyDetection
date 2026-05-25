import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np


SCHEDULE_ALGORITHM = "shared_boundary_sample_v1"
FADEOUT_TAIL_DURATION_S = 0.00875
FADEOUT_TAIL_EXPONENT = 1.3
_VALID_FREQUENCY_MODES = {"octave", "custom_linear", "custom_log"}
_VALID_RESOLUTIONS = {"R3", "R10", "R20", "R40", "R80"}
_CANONICAL_RESOLVER_METADATA_KEYS = {
    "stimulus_method",
    "frequency_mode",
    "stimulus_type",
    "num_steps",
    "effective_start_freq",
    "effective_stop_freq",
    "frequencies",
    "min_duration",
    "min_cycles",
    "amplitude",
    "repeat_times",
    "sample_rate",
    "schedule_sample_rate",
    "schedule_provenance",
    "transition_hz",
    "total_time",
    "safe_max_freq",
    "frequency_clamped",
    "per_repetition_sample_count",
    "alignment_sample_count",
    "playback_sample_count",
    "fadeout_tail_duration_s",
    "fadeout_tail_exponent",
    "fadeout_tail_sample_count",
    "segments",
    "step_durations",
    "schedule_algorithm",
}


@dataclass(frozen=True)
class FrequencyStepSegment:
    repetition_index: int
    step_index: int
    frequency_hz: float
    start_sample: int
    end_sample: int
    boundary_zero_sample: int
    base_duration_s: float
    actual_duration_s: float
    sample_count: int


@dataclass(frozen=True)
class FrequencySteppedFrequencyResult:
    frequencies_hz: np.ndarray
    frequency_mode: str
    start_freq: float
    stop_freq: float
    effective_start_freq: float
    effective_stop_freq: float
    num_steps: int
    resolution: object
    safe_max_freq: float
    frequency_clamped: bool


@dataclass(frozen=True)
class FrequencySteppedSignalResult:
    data: np.ndarray
    sample_rate: int
    frequencies_hz: np.ndarray
    segments: tuple[FrequencyStepSegment, ...]
    per_repetition_sample_count: int
    transition_hz: float
    total_time: float
    metadata: dict


def build_frequency_stepped_frequencies(
    *,
    sample_rate,
    frequency_mode=None,
    stimulus_type=None,
    start_freq=None,
    stop_freq=None,
    num_steps=None,
    resolution=None,
    frequencies=None,
):
    sample_rate = _positive_int(sample_rate, "sample_rate")
    mode = _resolve_frequency_mode(frequency_mode, stimulus_type)
    resolution_value = _validate_resolution(resolution) if mode == "octave" and resolution is not None else None
    safe_max_freq = float(sample_rate) * 0.475

    if frequencies is not None:
        raw_frequency_values = _positive_frequency_array(frequencies)
        if mode == "octave":
            _validate_octave_explicit_frequencies(raw_frequency_values, resolution_value)
        frequency_values, frequency_clamped = _clamp_frequencies(raw_frequency_values, safe_max_freq)
        effective_start = float(frequency_values[0])
        effective_stop = float(frequency_values[-1])
        return FrequencySteppedFrequencyResult(
            frequencies_hz=frequency_values,
            frequency_mode=mode,
            start_freq=effective_start,
            stop_freq=effective_stop,
            effective_start_freq=effective_start,
            effective_stop_freq=effective_stop,
            num_steps=int(frequency_values.size),
            resolution=resolution_value if mode == "octave" else None,
            safe_max_freq=safe_max_freq,
            frequency_clamped=frequency_clamped,
        )

    if mode == "octave":
        frequency_values, clamped_start, clamped_stop, frequency_clamped = _build_octave_frequencies(
            start_freq=start_freq,
            stop_freq=stop_freq,
            sample_rate=sample_rate,
            safe_max_freq=safe_max_freq,
            resolution=resolution_value,
        )
    elif mode == "custom_linear":
        frequency_values, clamped_start, clamped_stop, frequency_clamped = _build_custom_frequencies(
            start_freq=start_freq,
            stop_freq=stop_freq,
            num_steps=num_steps,
            safe_max_freq=safe_max_freq,
            log_space=False,
        )
    elif mode == "custom_log":
        frequency_values, clamped_start, clamped_stop, frequency_clamped = _build_custom_frequencies(
            start_freq=start_freq,
            stop_freq=stop_freq,
            num_steps=num_steps,
            safe_max_freq=safe_max_freq,
            log_space=True,
        )
    else:
        raise ValueError(f"Unsupported frequency_mode: {mode}")

    return FrequencySteppedFrequencyResult(
        frequencies_hz=frequency_values,
        frequency_mode=mode,
        start_freq=float(clamped_start),
        stop_freq=float(clamped_stop),
        effective_start_freq=float(frequency_values[0]),
        effective_stop_freq=float(frequency_values[-1]),
        num_steps=int(frequency_values.size),
        resolution=resolution_value if mode == "octave" else None,
        safe_max_freq=safe_max_freq,
        frequency_clamped=frequency_clamped,
    )


def generate_frequency_stepped(
    *,
    sample_rate,
    repeat_times=1,
    min_duration,
    min_cycles,
    frequency_mode=None,
    stimulus_type=None,
    start_freq=None,
    stop_freq=None,
    num_steps=None,
    resolution=None,
    frequencies=None,
    amplitude=1.0,
    generate_waveform=True,
):
    sample_rate = _positive_int(sample_rate, "sample_rate")
    repeat_times = _positive_int(repeat_times, "repeat_times")
    min_duration = _positive_float(min_duration, "min_duration")
    min_cycles = _positive_float(min_cycles, "min_cycles")
    amplitude = _finite_float(amplitude, "amplitude")

    frequency_result = build_frequency_stepped_frequencies(
        sample_rate=sample_rate,
        frequency_mode=frequency_mode,
        stimulus_type=stimulus_type,
        start_freq=start_freq,
        stop_freq=stop_freq,
        num_steps=num_steps,
        resolution=resolution,
        frequencies=frequencies,
    )

    one_repetition, base_segments, terminal_phase = _build_one_repetition(
        sample_rate=sample_rate,
        frequencies_hz=frequency_result.frequencies_hz,
        min_duration=min_duration,
        min_cycles=min_cycles,
        amplitude=amplitude,
        generate_waveform=generate_waveform,
    )
    per_repetition_sample_count = int(base_segments[-1].boundary_zero_sample + 1)
    segments = _repeat_segments(base_segments, repeat_times, per_repetition_sample_count)
    alignment_sample_count = int(per_repetition_sample_count * repeat_times)
    fadeout_tail_sample_count = int(round(sample_rate * FADEOUT_TAIL_DURATION_S))
    playback_sample_count = int(alignment_sample_count + fadeout_tail_sample_count)

    if generate_waveform:
        body = np.tile(one_repetition, repeat_times)
        tail = _build_fadeout_tail(
            sample_rate=sample_rate,
            sample_count=fadeout_tail_sample_count,
            final_frequency=float(frequency_result.frequencies_hz[-1]),
            terminal_phase=terminal_phase,
            amplitude=amplitude,
        )
        data = np.concatenate([body, tail]).astype(float, copy=False)
    else:
        data = np.array([], dtype=float)

    transition_hz = float(min_cycles / min_duration)
    total_time = playback_sample_count / sample_rate
    step_durations = [
        {
            "step_index": segment.step_index,
            "frequency_hz": segment.frequency_hz,
            "base_duration_s": segment.base_duration_s,
            "actual_duration_s": segment.actual_duration_s,
            "sample_count": segment.sample_count,
        }
        for segment in base_segments
    ]
    frequencies_list = _float_list(frequency_result.frequencies_hz)
    schedule_provenance = {
        "frequency_mode": frequency_result.frequency_mode,
        "frequencies": frequencies_list,
        "min_duration": float(min_duration),
        "min_cycles": float(min_cycles),
        "repeat_times": int(repeat_times),
        "sample_rate": int(sample_rate),
        "schedule_sample_rate": int(sample_rate),
        "schedule_algorithm": SCHEDULE_ALGORITHM,
    }
    if frequency_result.resolution is not None:
        schedule_provenance["resolution"] = frequency_result.resolution

    metadata = {
        "stimulus_method": "frequency_stepped",
        "stimulus_label": "step(sc)",
        "frequency_mode": frequency_result.frequency_mode,
        "stimulus_type": frequency_result.frequency_mode,
        "start_freq": frequency_result.start_freq,
        "stop_freq": frequency_result.stop_freq,
        "num_steps": frequency_result.num_steps,
        "resolution": frequency_result.resolution,
        "effective_start_freq": frequency_result.effective_start_freq,
        "effective_stop_freq": frequency_result.effective_stop_freq,
        "frequencies": frequencies_list,
        "min_duration": float(min_duration),
        "min_cycles": float(min_cycles),
        "amplitude": float(amplitude),
        "repeat_times": int(repeat_times),
        "sample_rate": int(sample_rate),
        "schedule_sample_rate": int(sample_rate),
        "schedule_provenance": schedule_provenance,
        "transition_hz": transition_hz,
        "total_time": total_time,
        "safe_max_freq": frequency_result.safe_max_freq,
        "frequency_clamped": frequency_result.frequency_clamped,
        "per_repetition_sample_count": per_repetition_sample_count,
        "alignment_sample_count": alignment_sample_count,
        "playback_sample_count": playback_sample_count,
        "fadeout_tail_duration_s": FADEOUT_TAIL_DURATION_S,
        "fadeout_tail_exponent": FADEOUT_TAIL_EXPONENT,
        "fadeout_tail_sample_count": fadeout_tail_sample_count,
        "segments": [asdict(segment) for segment in segments],
        "step_durations": step_durations,
        "schedule_algorithm": SCHEDULE_ALGORITHM,
    }

    return FrequencySteppedSignalResult(
        data=data,
        sample_rate=sample_rate,
        frequencies_hz=frequency_result.frequencies_hz,
        segments=segments,
        per_repetition_sample_count=per_repetition_sample_count,
        transition_hz=transition_hz,
        total_time=total_time,
        metadata=metadata,
    )


def resolve_frequency_stepped_schedule(stimulus_metadata: dict, sample_rate: int):
    sample_rate = _positive_int(sample_rate, "sample_rate")
    if not isinstance(stimulus_metadata, dict):
        raise ValueError("stimulus_metadata must be a dictionary")

    repaired_metadata = _repair_resolver_metadata(stimulus_metadata)

    cached = _cached_schedule_result(repaired_metadata, sample_rate)
    if cached is not None:
        return cached

    frequencies = _resolver_retained_frequencies(repaired_metadata)
    resolution = _resolver_resolution_for_generation(repaired_metadata, frequencies)

    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=repaired_metadata.get("repeat_times", 1),
        min_duration=repaired_metadata.get("min_duration"),
        min_cycles=repaired_metadata.get("min_cycles"),
        frequency_mode=repaired_metadata.get("frequency_mode"),
        stimulus_type=repaired_metadata.get("stimulus_type"),
        start_freq=repaired_metadata.get("start_freq"),
        stop_freq=repaired_metadata.get("stop_freq"),
        num_steps=repaired_metadata.get("num_steps"),
        resolution=resolution,
        frequencies=frequencies,
        amplitude=_metadata_amplitude(repaired_metadata),
        generate_waveform=False,
    )
    return _with_preserved_resolver_metadata(repaired_metadata, generated)


def _repair_resolver_metadata(stimulus_metadata):
    repaired = dict(stimulus_metadata)
    mode = _valid_frequency_mode_value(repaired.get("frequency_mode"))
    stimulus_type = _valid_frequency_mode_value(repaired.get("stimulus_type"))
    if mode is None:
        if stimulus_type is None:
            raise ValueError("frequency_mode is required")
        mode = stimulus_type
    repaired["frequency_mode"] = mode
    repaired["stimulus_type"] = mode
    return repaired


def _resolver_retained_frequencies(stimulus_metadata):
    if "frequencies" not in stimulus_metadata:
        return None
    try:
        frequencies = _positive_frequency_array(stimulus_metadata["frequencies"])
        if _valid_frequency_mode_value(stimulus_metadata.get("frequency_mode")) == "octave":
            _validate_octave_explicit_frequencies(frequencies, _resolver_valid_resolution_or_none(stimulus_metadata))
        return _float_list(frequencies)
    except (TypeError, ValueError):
        return None


def _resolver_resolution_for_generation(stimulus_metadata, frequencies):
    if _valid_frequency_mode_value(stimulus_metadata.get("frequency_mode")) != "octave":
        return stimulus_metadata.get("resolution")
    if frequencies is None:
        return stimulus_metadata.get("resolution")
    return _resolver_valid_resolution_or_none(stimulus_metadata)


def _resolver_valid_resolution_or_none(stimulus_metadata):
    resolution = stimulus_metadata.get("resolution")
    if resolution is None:
        return None
    try:
        return _validate_resolution(resolution)
    except (TypeError, ValueError):
        return None


def _build_one_repetition(
    *,
    sample_rate,
    frequencies_hz,
    min_duration,
    min_cycles,
    amplitude,
    generate_waveform,
):
    segments = []
    chunks = []
    start_sample = 0
    start_phase = 0.0
    terminal_phase = 0.0

    for step_index, frequency in enumerate(frequencies_hz):
        frequency = float(frequency)
        base_duration_s = max(min_duration, min_cycles / frequency)
        full_cycles = math.ceil(base_duration_s * frequency)
        boundary_offset = _boundary_index((full_cycles / frequency) * sample_rate)
        end_sample = start_sample + boundary_offset
        sample_count = end_sample - start_sample
        actual_duration_s = sample_count / sample_rate

        if generate_waveform and sample_count > 0:
            offsets = np.arange(sample_count, dtype=float)
            chunks.append(amplitude * np.sin(start_phase + (2 * math.pi * frequency * offsets / sample_rate)))

        terminal_phase = start_phase + 2 * math.pi * frequency * (sample_count / sample_rate)
        segments.append(
            FrequencyStepSegment(
                repetition_index=0,
                step_index=step_index,
                frequency_hz=frequency,
                start_sample=start_sample,
                end_sample=end_sample,
                boundary_zero_sample=end_sample,
                base_duration_s=float(base_duration_s),
                actual_duration_s=float(actual_duration_s),
                sample_count=int(sample_count),
            )
        )
        start_sample = end_sample
        start_phase = terminal_phase

    if generate_waveform:
        terminal_sample = np.array([amplitude * math.sin(terminal_phase)], dtype=float)
        if chunks:
            one_repetition = np.concatenate([*chunks, terminal_sample])
        else:
            one_repetition = terminal_sample
    else:
        one_repetition = np.array([], dtype=float)

    return one_repetition, tuple(segments), terminal_phase


def _repeat_segments(base_segments, repeat_times, per_repetition_sample_count):
    repeated = []
    for repetition_index in range(repeat_times):
        shift = repetition_index * per_repetition_sample_count
        for segment in base_segments:
            repeated.append(
                FrequencyStepSegment(
                    repetition_index=repetition_index,
                    step_index=segment.step_index,
                    frequency_hz=segment.frequency_hz,
                    start_sample=segment.start_sample + shift,
                    end_sample=segment.end_sample + shift,
                    boundary_zero_sample=segment.boundary_zero_sample + shift,
                    base_duration_s=segment.base_duration_s,
                    actual_duration_s=segment.actual_duration_s,
                    sample_count=segment.sample_count,
                )
            )
    return tuple(repeated)


def _build_fadeout_tail(*, sample_rate, sample_count, final_frequency, terminal_phase, amplitude):
    if sample_count <= 0:
        return np.array([], dtype=float)
    n = np.arange(1, sample_count + 1, dtype=float)
    gain = (1 - n / sample_count) ** FADEOUT_TAIL_EXPONENT
    return amplitude * np.sin(terminal_phase + (2 * math.pi * final_frequency * n / sample_rate)) * gain


def _boundary_index(value):
    nearest = round(value)
    if abs(value - nearest) <= 1e-9:
        return int(nearest)
    return int(math.ceil(value))


def _cached_schedule_result(stimulus_metadata, sample_rate):
    try:
        segments_payload = stimulus_metadata["segments"]
        provenance = stimulus_metadata["schedule_provenance"]
    except KeyError:
        return None
    if not isinstance(provenance, Mapping):
        return None
    if stimulus_metadata.get("schedule_sample_rate") != sample_rate:
        return None
    if provenance.get("schedule_algorithm") != SCHEDULE_ALGORITHM:
        return None
    if not _provenance_matches_metadata(provenance, stimulus_metadata, sample_rate):
        return None

    try:
        segments = tuple(_cached_segment_from_payload(segment) for segment in segments_payload)
        frequencies = _positive_frequency_array(stimulus_metadata["frequencies"])
        per_repetition_sample_count = int(stimulus_metadata["per_repetition_sample_count"])
        repeat_times = _positive_int(stimulus_metadata.get("repeat_times", 1), "repeat_times")
    except (KeyError, TypeError, ValueError):
        return None

    try:
        _validate_cached_schedule(
            segments=segments,
            frequencies=frequencies,
            per_repetition_sample_count=per_repetition_sample_count,
            repeat_times=repeat_times,
            step_durations=stimulus_metadata.get("step_durations"),
        )
    except (TypeError, ValueError):
        return None
    authoritative = _authoritative_cached_schedule_result(
        stimulus_metadata=stimulus_metadata,
        sample_rate=sample_rate,
        segments=segments,
        step_durations=stimulus_metadata.get("step_durations"),
        per_repetition_sample_count=per_repetition_sample_count,
    )
    if authoritative is None:
        return None

    metadata = _merged_resolver_metadata(stimulus_metadata, authoritative.metadata)
    metadata["segments"] = [asdict(segment) for segment in segments]
    metadata["step_durations"] = authoritative.metadata["step_durations"]
    return FrequencySteppedSignalResult(
        data=np.array([], dtype=float),
        sample_rate=sample_rate,
        frequencies_hz=authoritative.frequencies_hz,
        segments=segments,
        per_repetition_sample_count=authoritative.per_repetition_sample_count,
        transition_hz=authoritative.transition_hz,
        total_time=authoritative.total_time,
        metadata=metadata,
    )


def _authoritative_cached_schedule_result(
    *,
    stimulus_metadata,
    sample_rate,
    segments,
    step_durations,
    per_repetition_sample_count,
):
    try:
        expected = generate_frequency_stepped(
            sample_rate=sample_rate,
            repeat_times=stimulus_metadata.get("repeat_times", 1),
            min_duration=stimulus_metadata.get("min_duration"),
            min_cycles=stimulus_metadata.get("min_cycles"),
            frequency_mode=stimulus_metadata.get("frequency_mode"),
            stimulus_type=stimulus_metadata.get("stimulus_type"),
            resolution=stimulus_metadata.get("resolution"),
            frequencies=stimulus_metadata["frequencies"],
            amplitude=_metadata_amplitude(stimulus_metadata),
            generate_waveform=False,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if per_repetition_sample_count != expected.per_repetition_sample_count:
        return None
    if not _segments_match_expected(segments, expected.segments):
        return None
    if not _step_durations_match_expected(step_durations, expected.metadata["step_durations"]):
        return None
    return expected


def _metadata_amplitude(stimulus_metadata):
    try:
        return _finite_float(stimulus_metadata.get("amplitude", 1.0), "amplitude")
    except ValueError:
        return 1.0


def _with_preserved_resolver_metadata(stimulus_metadata, generated):
    metadata = _merged_resolver_metadata(stimulus_metadata, generated.metadata)
    return FrequencySteppedSignalResult(
        data=generated.data,
        sample_rate=generated.sample_rate,
        frequencies_hz=generated.frequencies_hz,
        segments=generated.segments,
        per_repetition_sample_count=generated.per_repetition_sample_count,
        transition_hz=generated.transition_hz,
        total_time=generated.total_time,
        metadata=metadata,
    )


def _merged_resolver_metadata(stimulus_metadata, generated_metadata):
    metadata = dict(generated_metadata)
    for key, value in stimulus_metadata.items():
        if key in ("start_freq", "stop_freq"):
            continue
        if key in _CANONICAL_RESOLVER_METADATA_KEYS:
            continue
        if key not in metadata or key == "stimulus_label" or "voltage" in str(key).lower():
            metadata[key] = value
    _apply_resolver_scalar_bounds(stimulus_metadata, metadata)
    return metadata


def _apply_resolver_scalar_bounds(stimulus_metadata, metadata):
    try:
        frequencies = _positive_frequency_array(metadata["frequencies"])
    except (KeyError, TypeError, ValueError):
        return
    effective_start = float(frequencies[0])
    effective_stop = float(frequencies[-1])
    metadata["effective_start_freq"] = effective_start
    metadata["effective_stop_freq"] = effective_stop

    caller_start = _optional_positive_float(stimulus_metadata.get("start_freq"))
    caller_stop = _optional_positive_float(stimulus_metadata.get("stop_freq"))
    mode = _valid_frequency_mode_value(metadata.get("frequency_mode"))
    if mode in {"custom_linear", "custom_log"}:
        if (
            caller_start is not None
            and caller_stop is not None
            and _float_close(caller_start, effective_start)
            and _float_close(caller_stop, effective_stop)
        ):
            metadata["start_freq"] = caller_start
            metadata["stop_freq"] = caller_stop
        else:
            metadata["start_freq"] = effective_start
            metadata["stop_freq"] = effective_stop
        return

    if (
        caller_start is not None
        and caller_stop is not None
        and _scalar_bounds_direction_matches(caller_start, caller_stop, effective_start, effective_stop)
    ):
        metadata["start_freq"] = caller_start
        metadata["stop_freq"] = caller_stop
    else:
        metadata["start_freq"] = effective_start
        metadata["stop_freq"] = effective_stop


def _optional_positive_float(value):
    try:
        return _positive_float(value, "frequency bound")
    except (TypeError, ValueError):
        return None


def _scalar_bounds_direction_matches(start_freq, stop_freq, effective_start_freq, effective_stop_freq):
    effective_delta = effective_stop_freq - effective_start_freq
    if math.isclose(effective_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return True
    caller_delta = stop_freq - start_freq
    if math.isclose(caller_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return False
    return (caller_delta > 0) == (effective_delta > 0)


def _segments_match_expected(segments, expected_segments):
    if len(segments) != len(expected_segments):
        return False
    for segment, expected in zip(segments, expected_segments):
        if segment.repetition_index != expected.repetition_index:
            return False
        if segment.step_index != expected.step_index:
            return False
        if not _float_close(segment.frequency_hz, expected.frequency_hz):
            return False
        if segment.start_sample != expected.start_sample:
            return False
        if segment.end_sample != expected.end_sample:
            return False
        if segment.boundary_zero_sample != expected.boundary_zero_sample:
            return False
        if not _float_close(segment.base_duration_s, expected.base_duration_s):
            return False
        if not _float_close(segment.actual_duration_s, expected.actual_duration_s):
            return False
        if segment.sample_count != expected.sample_count:
            return False
    return True


def _step_durations_match_expected(step_durations, expected_step_durations):
    if not isinstance(step_durations, (list, tuple)):
        return False
    if len(step_durations) != len(expected_step_durations):
        return False
    for duration, expected in zip(step_durations, expected_step_durations):
        if not isinstance(duration, dict):
            return False
        try:
            duration_step_index = int(duration["step_index"])
            frequency_hz = float(duration["frequency_hz"])
            base_duration_s = float(duration["base_duration_s"])
            actual_duration_s = float(duration["actual_duration_s"])
            sample_count = int(duration["sample_count"])
        except (KeyError, TypeError, ValueError):
            return False
        if duration_step_index != expected["step_index"]:
            return False
        if not _float_close(frequency_hz, expected["frequency_hz"]):
            return False
        if not _float_close(base_duration_s, expected["base_duration_s"]):
            return False
        if not _float_close(actual_duration_s, expected["actual_duration_s"]):
            return False
        if sample_count != expected["sample_count"]:
            return False
    return True


def _provenance_matches_metadata(provenance, metadata, sample_rate):
    if not isinstance(provenance, Mapping):
        return False
    comparisons = {
        "frequency_mode": metadata.get("frequency_mode"),
        "min_duration": metadata.get("min_duration"),
        "min_cycles": metadata.get("min_cycles"),
        "repeat_times": metadata.get("repeat_times", 1),
        "sample_rate": sample_rate,
        "schedule_sample_rate": sample_rate,
    }
    if metadata.get("frequency_mode") == "octave" and metadata.get("resolution") is not None:
        comparisons["resolution"] = metadata.get("resolution")
    for key, expected in comparisons.items():
        if key not in provenance:
            return False
        if isinstance(expected, float):
            try:
                provenance_value = float(provenance[key])
            except (TypeError, ValueError):
                return False
            if not math.isclose(provenance_value, expected, rel_tol=0.0, abs_tol=1e-12):
                return False
        elif provenance[key] != expected:
            return False
    if "frequencies" in metadata:
        if "frequencies" not in provenance:
            return False
        try:
            return _float_list(_positive_frequency_array(provenance["frequencies"])) == _float_list(
                _positive_frequency_array(metadata["frequencies"])
            )
        except (TypeError, ValueError):
            return False
    return False


def _cached_segment_from_payload(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Cached segment must be a dictionary")
    try:
        return FrequencyStepSegment(
            repetition_index=_nonnegative_integral_int(payload["repetition_index"], "repetition_index"),
            step_index=_nonnegative_integral_int(payload["step_index"], "step_index"),
            frequency_hz=_positive_float(payload["frequency_hz"], "frequency_hz"),
            start_sample=_nonnegative_integral_int(payload["start_sample"], "start_sample"),
            end_sample=_nonnegative_integral_int(payload["end_sample"], "end_sample"),
            boundary_zero_sample=_nonnegative_integral_int(
                payload["boundary_zero_sample"], "boundary_zero_sample"
            ),
            base_duration_s=_positive_float(payload["base_duration_s"], "base_duration_s"),
            actual_duration_s=_positive_float(payload["actual_duration_s"], "actual_duration_s"),
            sample_count=_nonnegative_integral_int(payload["sample_count"], "sample_count"),
        )
    except KeyError as exc:
        raise ValueError("Cached segment is incomplete") from exc


def _validate_segments(segments):
    previous_start = -1
    for segment in segments:
        if segment.start_sample < previous_start:
            raise ValueError("Cached segments are not ordered")
        if segment.end_sample < segment.start_sample:
            raise ValueError("Cached segment end precedes start")
        if segment.sample_count != segment.end_sample - segment.start_sample:
            raise ValueError("Cached segment sample_count mismatch")
        previous_start = segment.start_sample


def _validate_cached_schedule(
    *,
    segments,
    frequencies,
    per_repetition_sample_count,
    repeat_times,
    step_durations,
):
    _validate_segments(segments)
    if per_repetition_sample_count <= 0:
        raise ValueError("Cached per_repetition_sample_count must be positive")
    steps_per_repetition = len(frequencies)
    expected_segment_count = steps_per_repetition * repeat_times
    if len(segments) != expected_segment_count:
        raise ValueError("Cached segment count does not match frequencies and repetitions")

    previous_end = -1
    base_segments = segments[:steps_per_repetition]
    for index, segment in enumerate(segments):
        repetition_index = index // steps_per_repetition
        step_index = index % steps_per_repetition
        expected_shift = repetition_index * per_repetition_sample_count
        if segment.repetition_index != repetition_index:
            raise ValueError("Cached segment repetition_index mismatch")
        if segment.step_index != step_index:
            raise ValueError("Cached segment step_index mismatch")
        if not _float_close(segment.frequency_hz, frequencies[step_index]):
            raise ValueError("Cached segment frequency mismatch")
        if segment.start_sample < 0 or segment.end_sample < 0 or segment.boundary_zero_sample < 0:
            raise ValueError("Cached segment contains negative sample indices")
        if segment.start_sample < previous_end:
            raise ValueError("Cached segments overlap")

        if repetition_index == 0:
            if step_index == 0 and segment.start_sample != 0:
                raise ValueError("Cached first segment must start at sample zero")
        else:
            base_segment = base_segments[step_index]
            if segment.start_sample != base_segment.start_sample + expected_shift:
                raise ValueError("Cached segment start is not repetition-shifted")
            if segment.end_sample != base_segment.end_sample + expected_shift:
                raise ValueError("Cached segment end is not repetition-shifted")
            if segment.boundary_zero_sample != base_segment.boundary_zero_sample + expected_shift:
                raise ValueError("Cached segment boundary is not repetition-shifted")
            if segment.sample_count != base_segment.sample_count:
                raise ValueError("Cached segment sample_count is not repetition-shifted")
            if not _float_close(segment.base_duration_s, base_segment.base_duration_s):
                raise ValueError("Cached segment base_duration_s is not repetition-shifted")
            if not _float_close(segment.actual_duration_s, base_segment.actual_duration_s):
                raise ValueError("Cached segment actual_duration_s is not repetition-shifted")

        if step_index < steps_per_repetition - 1:
            next_segment = segments[index + 1]
            if segment.boundary_zero_sample != next_segment.start_sample:
                raise ValueError("Cached internal boundary does not match next segment start")
            if segment.end_sample != segment.boundary_zero_sample:
                raise ValueError("Cached internal segment end does not match boundary")
        else:
            expected_final_boundary = per_repetition_sample_count * (repetition_index + 1) - 1
            if segment.boundary_zero_sample != expected_final_boundary:
                raise ValueError("Cached final boundary does not match repetition length")
            if segment.end_sample != segment.boundary_zero_sample:
                raise ValueError("Cached final segment end does not match terminal boundary")
        previous_end = segment.end_sample

    _validate_step_durations(step_durations, base_segments)


def _validate_step_durations(step_durations, base_segments):
    if not isinstance(step_durations, (list, tuple)):
        raise ValueError("Cached step_durations must be a list")
    if len(step_durations) != len(base_segments):
        raise ValueError("Cached step_durations length mismatch")
    for step_index, (duration, segment) in enumerate(zip(step_durations, base_segments)):
        if not isinstance(duration, dict):
            raise ValueError("Cached step_durations entries must be dictionaries")
        try:
            duration_step_index = int(duration["step_index"])
            frequency_hz = float(duration["frequency_hz"])
            base_duration_s = float(duration["base_duration_s"])
            actual_duration_s = float(duration["actual_duration_s"])
            sample_count = int(duration["sample_count"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Cached step_durations entry is incomplete") from None
        if duration_step_index != step_index:
            raise ValueError("Cached step_durations step_index mismatch")
        if not _float_close(frequency_hz, segment.frequency_hz):
            raise ValueError("Cached step_durations frequency mismatch")
        if not _float_close(base_duration_s, segment.base_duration_s):
            raise ValueError("Cached step_durations base_duration_s mismatch")
        if not _float_close(actual_duration_s, segment.actual_duration_s):
            raise ValueError("Cached step_durations actual_duration_s mismatch")
        if sample_count != segment.sample_count:
            raise ValueError("Cached step_durations sample_count mismatch")


def _float_close(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _build_custom_frequencies(*, start_freq, stop_freq, num_steps, safe_max_freq, log_space):
    start = _positive_float(start_freq, "start_freq")
    stop = _positive_float(stop_freq, "stop_freq")
    steps = _positive_int(num_steps, "num_steps")
    if steps == 1 and not math.isclose(start, stop, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("num_steps == 1 requires start_freq == stop_freq")
    clamped_start = min(start, safe_max_freq)
    clamped_stop = min(stop, safe_max_freq)
    if log_space:
        frequencies = np.logspace(math.log10(clamped_start), math.log10(clamped_stop), steps)
    else:
        frequencies = np.linspace(clamped_start, clamped_stop, steps)
    frequencies, generated_clamped = _clamp_frequencies(frequencies, safe_max_freq)
    frequency_clamped = start != clamped_start or stop != clamped_stop or generated_clamped
    return frequencies, clamped_start, clamped_stop, frequency_clamped


def _build_octave_frequencies(*, start_freq, stop_freq, sample_rate, safe_max_freq, resolution):
    resolution = _validate_resolution(resolution)
    start = _positive_float(start_freq, "start_freq")
    stop = _positive_float(stop_freq, "stop_freq")
    clamped_start = min(start, safe_max_freq)
    clamped_stop = min(stop, safe_max_freq)
    preferred = _preferred_frequencies(resolution, safe_max_freq)
    if not preferred:
        raise ValueError("No octave frequencies are available for this sample_rate")

    if math.isclose(clamped_start, clamped_stop, rel_tol=0.0, abs_tol=1e-12):
        snapped = _snap_preferred(preferred, clamped_start, tie="lower")
        frequencies = np.array([snapped], dtype=float)
    else:
        descending = clamped_start > clamped_stop
        lower_bound = min(clamped_start, clamped_stop)
        upper_bound = max(clamped_start, clamped_stop)
        snapped_low = _snap_preferred(preferred, lower_bound, tie="lower")
        snapped_high = _snap_preferred(preferred, upper_bound, tie="upper")
        if snapped_low > snapped_high:
            snapped_low, snapped_high = snapped_high, snapped_low
        selected = [value for value in preferred if snapped_low <= value <= snapped_high]
        if descending:
            selected.reverse()
        frequencies = np.array(selected, dtype=float)

    frequency_clamped = start != clamped_start or stop != clamped_stop
    return frequencies, clamped_start, clamped_stop, frequency_clamped


def preferred_octave_frequencies(resolution, *, sample_rate=None, safe_max_freq=None):
    resolution = _validate_resolution(resolution)
    if safe_max_freq is None:
        if sample_rate is None:
            raise ValueError("sample_rate or safe_max_freq is required")
        sample_rate = _positive_int(sample_rate, "sample_rate")
        safe_max_freq = float(sample_rate) * 0.475
    else:
        safe_max_freq = _positive_float(safe_max_freq, "safe_max_freq")
    return _preferred_frequencies(resolution, safe_max_freq)


def _preferred_frequencies(resolution, safe_max_freq):
    if resolution == "R3":
        r10_bases = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]
        r10_values = []
        for decade in _preferred_decades(safe_max_freq):
            scale = 10**decade
            r10_values.extend(base * scale for base in r10_bases)
        r10_values = sorted(float(value) for value in r10_values)
        anchor_index = r10_values.index(1000.0)
        values = [
            value
            for index, value in enumerate(r10_values)
            if value >= 1.0 and (index - anchor_index) % 3 == 0
        ]
    else:
        bases = {
            "R10": [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80],
            "R20": [10, 11.2, 12.5, 14, 16, 18, 20, 22.4, 25, 28, 31.5, 35.5, 40, 45, 50, 56, 63, 71, 80, 90],
            "R40": [
                10,
                10.6,
                11.2,
                11.8,
                12.5,
                13.2,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21.2,
                22.4,
                23.6,
                25,
                26.5,
                28,
                30,
                31.5,
                33.5,
                35.5,
                37.5,
                40,
                42.5,
                45,
                47.5,
                50,
                53,
                56,
                60,
                63,
                67,
                71,
                75,
                80,
                85,
                90,
                95,
            ],
            "R80": [
                10,
                10.3,
                10.6,
                10.9,
                11.2,
                11.5,
                11.8,
                12.2,
                12.5,
                12.8,
                13.2,
                13.6,
                14,
                14.5,
                15,
                15.5,
                16,
                16.5,
                17,
                17.5,
                18,
                18.5,
                19,
                19.5,
                20,
                20.6,
                21.2,
                21.8,
                22.4,
                23,
                23.6,
                24.3,
                25,
                25.8,
                26.5,
                27.2,
                28,
                29,
                30,
                30.7,
                31.5,
                32.5,
                33.5,
                34.5,
                35.5,
                36.5,
                37.5,
                38.7,
                40,
                41.2,
                42.5,
                43.7,
                45,
                46.2,
                47.5,
                48.7,
                50,
                51.5,
                53,
                54.5,
                56,
                58,
                60,
                61.5,
                63,
                65,
                67,
                69,
                71,
                73,
                75,
                77.5,
                80,
                82,
                85,
                87,
                90,
                92,
                95,
                97,
            ],
        }
        values = []
        for decade in range(-2, 6):
            scale = 10**decade
            values.extend(base * scale for base in bases[resolution])
    return sorted({float(value) for value in values if value > 0 and value <= safe_max_freq})


def _preferred_decades(safe_max_freq):
    highest_decade = max(2, int(math.ceil(math.log10(max(float(safe_max_freq), 1.0)))))
    return range(-2, highest_decade + 1)


def _snap_preferred(preferred, value, tie):
    distances = [abs(candidate - value) for candidate in preferred]
    min_distance = min(distances)
    candidates = [
        candidate
        for candidate, distance in zip(preferred, distances)
        if math.isclose(distance, min_distance, rel_tol=0.0, abs_tol=1e-12)
    ]
    if tie == "upper":
        return float(max(candidates))
    return float(min(candidates))


def _valid_frequency_mode_value(value):
    if value is None:
        return None
    mode = str(value).strip()
    if mode in _VALID_FREQUENCY_MODES:
        return mode
    return None


def _validate_resolution(resolution):
    if not isinstance(resolution, str):
        raise ValueError(f"Unsupported resolution: {resolution}")
    resolution_value = resolution.strip()
    if resolution_value not in _VALID_RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return resolution_value


def _resolve_frequency_mode(frequency_mode, stimulus_type):
    mode = frequency_mode if frequency_mode is not None else stimulus_type
    if mode is None:
        raise ValueError("frequency_mode is required")
    mode = str(mode).strip()
    stimulus_type_value = str(stimulus_type).strip() if stimulus_type is not None else None
    if (
        frequency_mode is not None
        and stimulus_type is not None
        and stimulus_type_value in _VALID_FREQUENCY_MODES
        and mode != stimulus_type_value
    ):
        raise ValueError("frequency_mode and stimulus_type conflict")
    if mode not in _VALID_FREQUENCY_MODES:
        raise ValueError(f"Unsupported frequency_mode: {mode}")
    return mode


def _clamp_frequencies(frequencies, safe_max_freq):
    clamped = np.minimum(frequencies.astype(float, copy=False), safe_max_freq)
    return clamped, bool(np.any(~np.isclose(clamped, frequencies, rtol=0.0, atol=0.0)))


def _positive_frequency_array(frequencies):
    if isinstance(frequencies, (str, bytes, bytearray)):
        raise ValueError("frequencies must be a 1-D numeric sequence")
    _reject_bool_frequency_elements(frequencies)
    try:
        values = np.asarray(frequencies, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("frequencies must be a 1-D numeric sequence") from None
    if values.ndim != 1:
        raise ValueError("frequencies must be a 1-D numeric sequence")
    if values.size == 0:
        raise ValueError("frequencies must not be empty")
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError("frequencies must be finite positive values")
    return values


def _reject_bool_frequency_elements(frequencies):
    if isinstance(frequencies, np.ndarray):
        if frequencies.dtype == np.bool_:
            raise ValueError("frequencies must not contain boolean values")
        if frequencies.dtype == object and any(_is_bool_like(value) for value in frequencies.flat):
            raise ValueError("frequencies must not contain boolean values")
        return
    if isinstance(frequencies, (list, tuple)) and any(_is_bool_like(value) for value in frequencies):
        raise ValueError("frequencies must not contain boolean values")


def _validate_octave_explicit_frequency_order(frequencies):
    if frequencies.size <= 1:
        return
    deltas = np.diff(frequencies.astype(float, copy=False))
    if np.all(deltas > 0) or np.all(deltas < 0):
        return
    raise ValueError("octave frequencies must be strictly monotonic")


def _validate_octave_explicit_frequencies(frequencies, resolution):
    _validate_octave_explicit_frequency_order(frequencies)
    if resolution is None:
        preferred = _all_preferred_frequencies(float(np.max(frequencies)))
    else:
        preferred = _preferred_frequencies(resolution, float(np.max(frequencies)))
    for frequency in frequencies:
        if not any(_float_close(frequency, candidate) for candidate in preferred):
            raise ValueError("octave frequencies must be nominal preferred values")


def _all_preferred_frequencies(max_frequency):
    preferred = set()
    for resolution in _VALID_RESOLUTIONS:
        preferred.update(_preferred_frequencies(resolution, max_frequency))
    return sorted(preferred)


def _positive_int(value, name):
    if _is_bool_like(value):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if result != value and not (isinstance(value, str) and str(result) == value):
        try:
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive integer") from None
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integral_int(value, name):
    if _is_bool_like(value):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative integer") from None
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(numeric_value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value, name):
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value, name):
    if _is_bool_like(value):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _is_bool_like(value):
    return isinstance(value, (bool, np.bool_))


def _float_list(values):
    return [float(value) for value in values]
