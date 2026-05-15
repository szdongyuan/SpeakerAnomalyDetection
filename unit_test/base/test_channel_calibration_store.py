import json
import math

import pytest

from base.soundcard_calibration_manager import (
    MicChannelCalibrationResult,
    format_input_channel_label,
    load_mic_channel_v2pa_factors,
    resolve_analysis_v2pa_factor_for_channel,
    resolve_mic_channel_v2pa_factor,
    save_mic_channel_v2pa_factor,
)

def test_save_and_load_channel_v2pa_factor(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"

    save_mic_channel_v2pa_factor(1, 2.5, standard_spl=94, file_path=path)

    assert load_mic_channel_v2pa_factors(file_path=path) == {1: 2.5}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["channels"]["1"]["v2pa_factor"] == 2.5
    assert payload["channels"]["1"]["standard_spl"] == 94
    assert payload["channels"]["1"]["calibrated_at"]


def test_save_preserves_existing_valid_channels(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"

    save_mic_channel_v2pa_factor(3, 3.0, standard_spl=94, file_path=path)
    save_mic_channel_v2pa_factor(1, 1.5, standard_spl=114, file_path=path)

    assert load_mic_channel_v2pa_factors(file_path=path) == {1: 1.5, 3: 3.0}


def test_save_preserves_existing_channel_when_optional_metadata_is_corrupt(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "channels": {
                    "3": {
                        "v2pa_factor": 3.0,
                        "calibrated_at": "2026-05-12",
                        "standard_spl": "corrupt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    save_mic_channel_v2pa_factor(1, 1.5, standard_spl=114, file_path=path)

    assert load_mic_channel_v2pa_factors(file_path=path) == {1: 1.5, 3: 3.0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["channels"]["3"]["v2pa_factor"] == 3.0
    assert payload["channels"]["3"]["calibrated_at"] == "2026-05-12"
    assert "standard_spl" not in payload["channels"]["3"]
    assert payload["channels"]["1"]["standard_spl"] == 114


def test_load_missing_corrupt_or_invalid_json_returns_empty_dict(tmp_path):
    missing_path = tmp_path / "missing.json"
    corrupt_path = tmp_path / "corrupt.json"
    invalid_path = tmp_path / "invalid.json"
    wrong_shape_path = tmp_path / "wrong-shape.json"

    corrupt_path.write_text("{not json", encoding="utf-8")
    invalid_path.write_text(json.dumps({"version": 1, "channels": []}), encoding="utf-8")
    wrong_shape_path.write_text(json.dumps([{"channels": {}}]), encoding="utf-8")

    assert load_mic_channel_v2pa_factors(file_path=missing_path) == {}
    assert load_mic_channel_v2pa_factors(file_path=corrupt_path) == {}
    assert load_mic_channel_v2pa_factors(file_path=invalid_path) == {}
    assert load_mic_channel_v2pa_factors(file_path=wrong_shape_path) == {}


def test_load_ignores_invalid_channel_entries(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "channels": {
                    "0": {"v2pa_factor": 1.0},
                    "-1": {"v2pa_factor": 2.0},
                    "bad": {"v2pa_factor": 3.0},
                    "2": {"v2pa_factor": 0},
                    "3": {"v2pa_factor": math.inf},
                    "4": {"v2pa_factor": "not-number"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_mic_channel_v2pa_factors(file_path=path) == {0: 1.0}


def test_save_rejects_invalid_channel_or_factor(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"

    with pytest.raises(ValueError):
        save_mic_channel_v2pa_factor(-1, 1.0, file_path=path)
    with pytest.raises(ValueError):
        save_mic_channel_v2pa_factor(0, 0, file_path=path)
    with pytest.raises(ValueError):
        save_mic_channel_v2pa_factor(0, math.inf, file_path=path)


def test_save_rejects_non_finite_standard_spl(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"

    with pytest.raises(ValueError, match="standard_spl must be a finite number"):
        save_mic_channel_v2pa_factor(0, 1.0, standard_spl=math.nan, file_path=path)
    with pytest.raises(ValueError, match="standard_spl must be a finite number"):
        save_mic_channel_v2pa_factor(0, 1.0, standard_spl=math.inf, file_path=path)

    assert not path.exists()


def test_resolve_channel_factor_uses_exact_channel(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"
    save_mic_channel_v2pa_factor(1, 2.5, standard_spl=94, file_path=path)

    result = resolve_mic_channel_v2pa_factor(1, file_path=path)

    assert isinstance(result, MicChannelCalibrationResult)
    assert result.factor == 2.5
    assert result.requested_channel == 1
    assert result.source_channel == 1
    assert result.used_fallback is False
    assert result.has_any_calibration is True


def test_resolve_channel_factor_falls_back_to_lowest_calibrated_channel(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"
    save_mic_channel_v2pa_factor(3, 3.0, standard_spl=94, file_path=path)
    save_mic_channel_v2pa_factor(1, 1.5, standard_spl=94, file_path=path)

    result = resolve_mic_channel_v2pa_factor(2, file_path=path)

    assert result.factor == 1.5
    assert result.requested_channel == 2
    assert result.source_channel == 1
    assert result.used_fallback is True
    assert result.has_any_calibration is True


def test_resolve_channel_factor_returns_no_data_when_store_empty(tmp_path):
    result = resolve_mic_channel_v2pa_factor(0, file_path=tmp_path / "missing.json")

    assert result.factor is None
    assert result.requested_channel == 0
    assert result.source_channel is None
    assert result.used_fallback is False
    assert result.has_any_calibration is False


def test_format_input_channel_label_uses_one_based_display():
    assert format_input_channel_label(0) == "In1"
    assert format_input_channel_label("2") == "In3"


def test_resolve_analysis_factor_warns_when_using_fallback(tmp_path):
    path = tmp_path / "mic_channel_calibration.json"
    warnings = []
    save_mic_channel_v2pa_factor(0, 1.25, standard_spl=94, file_path=path)

    factor = resolve_analysis_v2pa_factor_for_channel(2, warn_callback=warnings.append, file_path=path)

    assert factor == 1.25
    assert len(warnings) == 1
    assert "In3" in warnings[0]
    assert "In1" in warnings[0]


def test_resolve_analysis_factor_raises_chinese_prompt_when_no_data(tmp_path):
    with pytest.raises(ValueError, match="未找到输入通道校准数据"):
        resolve_analysis_v2pa_factor_for_channel(0, file_path=tmp_path / "missing.json")
