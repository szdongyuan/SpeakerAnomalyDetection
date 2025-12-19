"""
Regression test: Verify PRB 10-35 harmonic analysis unchanged.

Ensures that extending PRB to support 2-35 harmonics does not break
existing 10-35 harmonic analysis functionality.
"""
import numpy as np
import pytest
import os
import sys


def test_prb_10_35_harmonics_still_works():
    """
    Test that PRB analysis with harmonics 10-35 produces same results.

    This is a regression test to ensure backward compatibility.
    """
    from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    fundamental_freq = 100.0
    step_duration = 0.5

    # Create signal with harmonics 10, 15, 20 (original PRB range)
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t)  # Fundamental
    for h in [10, 15, 20]:
        signal += 0.01 * np.sin(2 * np.pi * h * fundamental_freq * t)

    # Metadata
    stimulus_metadata = {
        'fundamental_frequencies': np.array([fundamental_freq]),
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': step_duration,
        'step_duration': step_duration,
        'start_freq': fundamental_freq,
        'stop_freq': fundamental_freq,
        'stimulus_type': 'linear',
        'stimulus_method': 'steps'
    }

    # Analyze
    analyzer = PerceptualStepSignalHD(sample_rate)
    result = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=[10, 15, 20],
        stft_window_type='hann',
        spl_calibration_db=0.0
    )

    loudness = result['perceptual_loudness'][0]

    print(f"PRB loudness (harmonics 10, 15, 20): {loudness:.1f} phons")

    # Should produce positive loudness
    assert loudness > 0, f"PRB should produce positive loudness, got {loudness:.1f}"

    # Reasonable loudness range for these harmonics (3 harmonics at 0.01 amplitude)
    # Expected: ~60-70 phons based on combined energy
    assert 50 < loudness < 80, f"Loudness {loudness:.1f} phons outside expected range"


@pytest.mark.skipif(
    os.environ.get("RUN_UI_TESTS") != "1",
    reason="UI test disabled by default; set RUN_UI_TESTS=1 to enable.",
)
def test_prb_select_all_10_35_backward_compat():
    """
    Test that old config files with 10-35 harmonics still load correctly.

    Validates backward compatibility with existing user configurations.
    """
    from PyQt5.QtWidgets import QApplication
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
    from unittest.mock import MagicMock

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Mock old config with 10-35 harmonics (pre-extension)
    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {
        'PRB': {
            'selected_labels': [10, 11, 12, 15, 20, 25, 30, 35],
            'all_checked': False
        }
    }

    # Create dialog
    dialog = PerceptualRbConfigWindow(mock_config_manager, 'PRB')

    # Verify old config loads without modification
    expected = [10, 11, 12, 15, 20, 25, 30, 35]
    assert dialog.selected_labels == expected, \
        f"Old 10-35 config should load unchanged, got {dialog.selected_labels}"


if __name__ == '__main__':
    print("=" * 80)
    print("Regression Test 1: PRB 10-35 analysis")
    print("=" * 80)
    test_prb_10_35_harmonics_still_works()

    print("\n" + "=" * 80)
    print("Regression Test 2: Backward compatibility with old configs")
    print("=" * 80)
    test_prb_select_all_10_35_backward_compat()

    print("\n✓ All regression tests passed!")
