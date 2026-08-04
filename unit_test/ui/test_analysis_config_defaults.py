import ast
import json
from pathlib import Path

from consts.acoustic_analysis.specific_consts import spec_consts


DEFAULT_CONFIG_PATH = Path("ui/ui_config/analysis_default_config.json")
OPERATION_SEQUENCE_PATH = Path("ui/operation_sequence.py")

EXPOSED_ANALYSIS_TYPES = {
    "SPL",
    "SPLF",
    "FFT",
    "Spec",
    "RSC",
    "FR",
    "FBA",
    "HD",
    "RB",
    "PRB",
    "LP",
    "PD",
    "PM",
    "ED",
    "AI",
    "Excel",
}

MANUAL_SEGMENT_ANALYSIS_TYPES = ("SPL", "SPLF", "FR", "HD", "RB", "PRB")
MANUAL_SEGMENT_DEFAULT_KEYS = (
    "limit_mode",
    "manual_upper_enabled",
    "manual_lower_enabled",
    "manual_upper_segments",
    "manual_lower_segments",
)


def load_defaults():
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_every_exposed_analysis_type_has_default_config():
    defaults = load_defaults()

    assert EXPOSED_ANALYSIS_TYPES.issubset(defaults)
    for analysis_type in EXPOSED_ANALYSIS_TYPES:
        assert isinstance(defaults[analysis_type], dict), analysis_type


def test_mel_is_integrated_into_the_spec_analysis_item():
    tree = ast.parse(OPERATION_SEQUENCE_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_analysis_list_layout":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "analysis_items"
                for target in child.targets
            ):
                continue

            analysis_items = ast.literal_eval(child.value)
            assert "频谱分析 (Spec) " in analysis_items
            assert "梅尔频谱图 (MEL) " not in analysis_items
            assert "时频谱图 (Spec) " not in analysis_items
            return

    raise AssertionError("analysis_items was not found")


def test_fft_default_contains_active_fft_configuration_only():
    defaults = load_defaults()

    assert "FFT" in defaults
    assert defaults["FFT"]["n_fft"] == 4096
    assert defaults["FFT"]["limit_checked"] is False
    assert defaults["FFT"]["limit_data"] is None
    assert defaults["FFT"]["analysis_channel"] == 0
    assert not any(key.startswith("dominant_tone") for key in defaults["FFT"])


def test_curve_and_distortion_defaults_keep_legacy_threshold_and_golden_keys():
    defaults = load_defaults()

    for analysis_type in ("SPL", "SPLF", "FR", "FBA", "HD", "RB", "PRB"):
        assert defaults[analysis_type]["limit_checked"] is False
        assert "limit_data" in defaults[analysis_type]

    for analysis_type in ("SPLF", "FR", "HD", "RB", "PRB"):
        assert defaults[analysis_type]["golden_sample_checked"] is False
        assert defaults[analysis_type]["golden_sample_display_modes"] == ["deviation"]


def test_manual_segment_enabled_defaults_have_stable_schema_without_scalar_keys():
    defaults = load_defaults()

    for analysis_type in MANUAL_SEGMENT_ANALYSIS_TYPES:
        config = defaults[analysis_type]
        for key in MANUAL_SEGMENT_DEFAULT_KEYS:
            assert key in config, analysis_type
        assert config["limit_mode"] == "csv"
        assert config["manual_upper_enabled"] is True
        assert config["manual_lower_enabled"] is False
        assert config["manual_upper_segments"] == []
        assert config["manual_lower_segments"] == []
        assert "manual_upper" not in config
        assert "manual_lower" not in config

    assert "limit_mode" not in defaults["FBA"]
    assert "manual_upper_segments" not in defaults["FBA"]
    assert "manual_lower_segments" not in defaults["FBA"]


def test_channel_defaults_are_present_for_channel_aware_dialogs():
    defaults = load_defaults()

    for analysis_type in ("SPL", "SPLF", "FFT", "Spec", "FBA", "LP", "AI"):
        assert defaults[analysis_type]["analysis_channel"] == 0


def test_spec_defaults_include_all_spectrum_modes():
    defaults = load_defaults()
    spec_defaults = defaults["Spec"]

    assert spec_defaults["n_fft"] == spec_consts.DEFAULT_SPEC_N_FFT
    assert spec_defaults["hop_length"] == spec_consts.DEFAULT_SPEC_HOP_LENGTH
    assert spec_defaults["window_func"] == spec_consts.DEFAULT_SPEC_WINDOW
    assert spec_defaults["color_map"] == spec_consts.DEFAULT_SPEC_COLOR_MAP
    assert spec_defaults["freq_scale_type"] == spec_consts.DEFAULT_SPEC_MODE
    assert spec_defaults["mel_n_mels"] == spec_consts.DEFAULT_MEL_BAND_COUNT
    assert spec_defaults["mel_fmin_hz"] == spec_consts.DEFAULT_MEL_FMIN_HZ
    assert spec_defaults["mel_fmax_mode"] == spec_consts.DEFAULT_MEL_FMAX_MODE
    assert "mel_fmax_hz" not in spec_defaults
    assert spec_defaults["top_limit"] == spec_consts.DEFAULT_SPEC_TOP_LIMIT
    assert spec_defaults["bottom_limit"] == spec_consts.DEFAULT_SPEC_BOTTOM_LIMIT
    assert spec_defaults["custom_limit"] is spec_consts.DEFAULT_SPEC_CUSTOM_LIMIT
    assert spec_defaults["analysis_channel"] == spec_consts.DEFAULT_SPEC_ANALYSIS_CHANNEL
    assert "MEL" not in defaults
    assert "v2pa_factor" not in spec_defaults
    assert "weighting" not in spec_defaults


def test_special_dialog_defaults_include_required_legacy_keys():
    defaults = load_defaults()

    spl_defaults = defaults["SPL"]
    for key in (
        "smooth_checked",
        "smooth_enabled",
        "smooth_unit",
        "smooth_time_sec",
        "smooth_points",
        "smooth_algo",
        "spl_window_unit",
        "spl_window_time_sec",
        "spl_window_points",
        "analysis_time_range_enabled",
        "analysis_start_time_sec",
        "analysis_end_time_sec",
        "limit_mode",
        "manual_upper_enabled",
        "manual_lower_enabled",
        "manual_upper_segments",
        "manual_lower_segments",
        "show_overall_spl",
    ):
        assert key in spl_defaults
    assert spl_defaults["show_overall_spl"] is False

    assert defaults["RSC"]["smoothing"] == 0
    assert "octave_smoothing" not in defaults["RSC"]
    assert defaults["RSC"]["enable_threshold_judgment"] is True
    assert defaults["RSC"]["channel_labels"] == {}

    pd_defaults = defaults["PD"]
    for key in (
        "smooth_enabled",
        "smooth_unit",
        "smooth_time_sec",
        "smooth_points",
        "smooth_algo",
        "spl_window_unit",
        "nms_unit",
        "advanced_mode",
        "test_peak_op",
    ):
        assert key in pd_defaults

    ed_defaults = defaults["ED"]
    assert ed_defaults["head"] == {"type": "PD", "config": {}}
    assert ed_defaults["tail"] == {"type": "PM", "config": {}}
    assert ed_defaults["pass_condition"] == {"n1": 1, "n2": 1}

    excel_defaults = defaults["Excel"]
    assert excel_defaults["save_items"] == []
    assert excel_defaults["save_dir"] is None
    assert excel_defaults["file_base"] == "analysis_results"


def test_pattern_match_default_does_not_depend_on_local_template_file():
    defaults = load_defaults()

    assert defaults["PM"]["pattern_list"] == []
    assert "pattern_save_path" not in defaults["PM"]
    assert "pattern_duration_sec" not in defaults["PM"]


def test_get_item_default_config_deep_copies_default_before_adding_type():
    tree = ast.parse(OPERATION_SEQUENCE_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_item_default_config":
            calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "deepcopy"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "copy"
            ]
            assert calls, "get_item_default_config must deep-copy defaults before adding type"
            return

    raise AssertionError("get_item_default_config was not found")
