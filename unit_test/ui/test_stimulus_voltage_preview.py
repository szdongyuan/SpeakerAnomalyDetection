import numpy as np
import pytest

from ui.stimulus_voltage_preview import build_stimulus_voltage_preview


@pytest.mark.parametrize('kind,peak', [('Peak', .1), ('rMs', np.sqrt(2) * .1)])
@pytest.mark.parametrize('scale', [1., .037, 1e308, 1e-308])
def test_whole_cycle_voltage_and_source_integrity(kind, peak, scale):
    source = np.array([0., 1., 0., -1.]) * scale
    before = source.copy()
    display, converted = build_stimulus_voltage_preview(source, .1, kind)
    assert converted
    np.testing.assert_allclose(display, [0., peak, 0., -peak])
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(source, display)


def test_noise_rms_uses_whole_source_including_silence():
    source = np.r_[np.random.default_rng(17).normal(size=1000), np.zeros(500)]
    display, converted = build_stimulus_voltage_preview(source, .3, 'RMS')
    assert converted
    assert np.sqrt(np.mean(display ** 2)) == pytest.approx(.3)
    np.testing.assert_allclose(display, source / np.sqrt(np.mean(source ** 2)) * .3)


@pytest.mark.parametrize('source', [[], [0., 0.], [1., -2.]])
@pytest.mark.parametrize('kind', ['Peak', 'RMS'])
def test_zero_voltage_is_safe(source, kind):
    display, converted = build_stimulus_voltage_preview(np.array(source), 0, kind)
    assert converted
    np.testing.assert_array_equal(display, np.zeros(len(source)))


@pytest.mark.parametrize('source', [[], [0., 0.]])
def test_empty_and_zero_source_are_safe(source):
    display, converted = build_stimulus_voltage_preview(np.array(source), .1, 'RMS')
    assert converted
    np.testing.assert_array_equal(display, source)


@pytest.mark.parametrize('voltage,kind', [(None, 'RMS'), ('bad', 'Peak'), (-1, 'RMS'),
    (np.nan, 'RMS'), (np.inf, 'Peak'), (.1, None), (.1, 'unknown')])
def test_invalid_metadata_retains_raw_amplitude(voltage, kind):
    source = np.array([0., .2, -.4])
    display, converted = build_stimulus_voltage_preview(source, voltage, kind)
    assert not converted
    np.testing.assert_array_equal(display, source)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_nonfinite_source_retains_gaps_even_at_zero_voltage(invalid):
    source = np.array([.2, invalid, -.4])
    display, converted = build_stimulus_voltage_preview(source, 0, 'Peak')
    assert not converted
    np.testing.assert_array_equal(display, source)


def test_unrepresentable_rms_target_falls_back():
    source = np.array([0., 1., 0., -1.])
    display, converted = build_stimulus_voltage_preview(source, 1.7e308, 'RMS')
    assert not converted
    np.testing.assert_array_equal(display, source)
