from base.audio_sample_rate import (
    resolve_device_sample_rate,
    resolve_duplex_sample_rate,
    resolve_input_sample_rate,
    resolve_output_sample_rate,
)


def test_audio_constants_include_bit_depth_and_device_enumeration_base_exceptions():
    from consts.audio_consts import VALID_BIT_DEPTHS, DEVICE_ENUMERATION_BASE_EXCEPTIONS

    assert VALID_BIT_DEPTHS == (32, 64)
    assert DEVICE_ENUMERATION_BASE_EXCEPTIONS == (ValueError, RuntimeError)


def test_audio_sample_rate_no_longer_exports_mismatch_message():
    import base.audio_sample_rate as module

    assert not hasattr(module, "MISMATCH_MESSAGE")


def test_resolver_reads_samplerate_not_default_or_sample_rate():
    device = {"samplerate": "48000", "default_samplerate": 44100, "sample_rate": 44100}
    result = resolve_device_sample_rate(device, "输入设备")
    assert result.ok is True
    assert result.sample_rate == 48000


def test_missing_device_returns_failure():
    result = resolve_input_sample_rate(None)
    assert result.ok is False
    assert result.sample_rate is None
    assert "输入设备" in result.message


def test_missing_samplerate_does_not_fallback_to_default_samplerate():
    result = resolve_output_sample_rate({"default_samplerate": 48000, "sample_rate": 44100})
    assert result.ok is False
    assert "采样率" in result.message


def test_unsupported_samplerate_returns_failure():
    result = resolve_output_sample_rate({"samplerate": 96000})
    assert result.ok is False
    assert "44100" in result.message and "48000" in result.message


def test_non_integral_numeric_samplerate_returns_failure():
    result = resolve_output_sample_rate({"samplerate": 48000.5})
    assert result.ok is False
    assert result.sample_rate is None
    assert "44100" in result.message and "48000" in result.message


def test_non_finite_numeric_samplerate_returns_failure():
    for value in (float("inf"), float("nan")):
        result = resolve_output_sample_rate({"samplerate": value})
        assert result.ok is False
        assert result.sample_rate is None
        assert "44100" in result.message and "48000" in result.message


def test_duplex_requires_matching_samplerates():
    result = resolve_duplex_sample_rate({"samplerate": 44100}, {"samplerate": 48000})
    assert result.ok is False
    assert result.message == "输入设备与输出设备的采样率不一致，请在硬件管理中设置为一致后重新选择。"


def test_duplex_succeeds_when_rates_match():
    result = resolve_duplex_sample_rate({"samplerate": 48000}, {"samplerate": "48000"})
    assert result.ok is True
    assert result.sample_rate == 48000


def test_vkinging_input_uses_hardware_rate_without_relaxing_soundcard_or_output():
    device = {"hostapi_name": "vkinging", "samplerate": 51200}
    for resolver in (resolve_input_sample_rate,
                     lambda d: resolve_device_sample_rate(d, "输入设备", allow_vkinging=True)):
        result = resolver(device)
        assert result.ok is True
        assert result.sample_rate == 51200
    assert resolve_output_sample_rate(device).ok is False
    assert resolve_input_sample_rate({"hostapi_name": "WASAPI", "samplerate": 51200}).ok is False


def test_vkinging_input_range_accepts_only_integer_hardware_rates():
    for value in (4000, 100001, 102400):
        assert resolve_input_sample_rate({"hostapi_name": "vkinging", "samplerate": value}).ok is True
    for value in (True, 51200.0, "51200", 3999, 102401, None):
        result = resolve_input_sample_rate({"hostapi_name": "vkinging", "samplerate": value})
        assert result.ok is False
        assert "采样率" in result.message


def test_duplex_rejects_vkinging_even_at_soundcard_compatible_rate():
    result = resolve_duplex_sample_rate(
        {"hostapi_name": "vkinging", "samplerate": 48000}, {"samplerate": 48000})
    assert result.ok is False
    assert "vkinging" in result.message


def test_vkinging_cannot_be_an_output_even_with_a_soundcard_rate():
    for rate in (44100, 48000):
        output = {"hostapi_name": "vkinging", "samplerate": rate}
        assert resolve_output_sample_rate(output).ok is False
        assert resolve_device_sample_rate(output, "输出设备").ok is False
        assert resolve_duplex_sample_rate({"samplerate": rate}, output).ok is False
