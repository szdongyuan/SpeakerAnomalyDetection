FREQUENCY_STEPPED_METHOD = "frequency_stepped"
FREQUENCY_STEPPED_LABEL = "step(sc)"
FREQUENCY_STEPPED_MODES = {"octave", "custom_linear", "custom_log"}
FREQUENCY_STEPPED_RESOLUTIONS = {"R3", "R10", "R20", "R40", "R80"}
FREQUENCY_STEPPED_GENERATOR_KEYS = (
    "sample_rate",
    "repeat_times",
    "min_duration",
    "min_cycles",
    "frequency_mode",
    "stimulus_type",
    "start_freq",
    "stop_freq",
    "num_steps",
    "resolution",
    "frequencies",
    "amplitude",
    "generate_waveform",
)
FREQUENCY_STEPPED_FILENAME_KEYS = (
    "stimulus_method",
    "frequency_mode",
    "start_freq",
    "stop_freq",
    "num_steps",
    "resolution",
    "min_duration",
    "min_cycles",
    "repeat_times",
    "sample_rate",
    "voltage_type",
    "voltage",
)
SUPPORTED_STIMULUS_METHODS = {"chirp", "step", "noise", "frequency_stepped"}
FREQUENCY_STEPPED_SCHEDULE_ALGORITHM = "shared_boundary_sample_v1"
FREQUENCY_STEPPED_FADEOUT_TAIL_DURATION_S = 0.00875
FREQUENCY_STEPPED_FADEOUT_TAIL_EXPONENT = 1.3
FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER = 35
FREQUENCY_STEPPED_MIN_PERCEPTUAL_HARMONIC_ORDER = 1
FREQUENCY_STEPPED_CANONICAL_METADATA_KEYS = {
    "stimulus_method",
    "frequency_mode",
    "stimulus_type",
    "start_freq",
    "stop_freq",
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
OCTAVE_PREFERRED_BASE_FREQUENCIES = {
    "R3": [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80],
    "R10": [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80],
    "R20": [10, 11.2, 12.5, 14, 16, 18, 20, 22.4, 25, 28, 31.5, 35.5, 40, 45, 50, 56, 63, 71, 80, 90],
    "R40": [10, 10.6, 11.2, 11.8, 12.5, 13.2, 14, 15, 16, 17, 18, 19, 20, 21.2, 22.4, 23.6, 25, 26.5, 28, 30, 31.5, 33.5, 35.5, 37.5, 40, 42.5, 45, 47.5, 50, 53, 56, 60, 63, 67, 71, 75, 80, 85, 90, 95],
    "R80": [10, 10.3, 10.6, 10.9, 11.2, 11.5, 11.8, 12.2, 12.5, 12.8, 13.2, 13.6, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.6, 21.2, 21.8, 22.4, 23, 23.6, 24.3, 25, 25.8, 26.5, 27.2, 28, 29, 30, 30.7, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.7, 40, 41.2, 42.5, 43.7, 45, 46.2, 47.5, 48.7, 50, 51.5, 53, 54.5, 56, 58, 60, 61.5, 63, 65, 67, 69, 71, 73, 75, 77.5, 80, 82, 85, 87, 90, 92, 95, 97],
}
