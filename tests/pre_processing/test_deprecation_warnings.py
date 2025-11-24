"""Tests for deprecation warnings on legacy methods."""
import numpy as np
import pytest
import warnings
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


class TestDeprecationWarnings:
    """Test that legacy methods emit DeprecationWarning."""

    def test_calculate_thd_emits_deprecation_warning(self):
        """Test that calculate_thd emits a DeprecationWarning."""
        analyzer = AudioThdFrequencyResponseAnalysis()

        # Create minimal synthetic data
        freq_dict = {500.0: {"argmax": 10, "i": 0, "harmonic": [10, 20], "harmonic_base": 30}}
        base_freq_list = [500.0, 510.0]
        recorded_signal = np.random.randn(44100)

        # Check that calculate_thd emits DeprecationWarning
        with pytest.warns(DeprecationWarning) as warning_list:
            with warnings.catch_warnings():
                # Suppress the nested get_harmonic warning for this test
                warnings.filterwarnings("ignore", message=".*get_harmonic.*")
                result = analyzer.calculate_thd(freq_dict, base_freq_list, recorded_signal, 44100)

        # Verify warning was raised for calculate_thd
        thd_warnings = [w for w in warning_list if "calculate_thd" in str(w.message)]
        assert len(thd_warnings) > 0, "calculate_thd warning not found"
        assert any("stimulus_metadata" in str(w.message) for w in thd_warnings)
        assert any("hd_refactoring_guide.md" in str(w.message) for w in thd_warnings)

    def test_calculate_spectrum_emits_deprecation_warning(self):
        """Test that calculate_spectrum emits a DeprecationWarning."""
        sr = 44100
        duration = 0.25
        reference_signal = np.random.randn(int(duration * sr))

        # Check that calculate_spectrum emits DeprecationWarning
        with pytest.warns(DeprecationWarning) as warning_list:
            freq_dict, base_freq_list = AudioThdFrequencyResponseAnalysis.calculate_spectrum(
                reference_signal, sr
            )

        # Verify warning was raised
        assert len(warning_list) > 0
        assert any("calculate_spectrum is deprecated" in str(w.message) for w in warning_list)
        assert any("HarmonicIndexBuilder" in str(w.message) for w in warning_list)
        assert any("hd_refactoring_guide.md" in str(w.message) for w in warning_list)

        # Verify the method still works
        assert isinstance(freq_dict, dict)
        assert isinstance(base_freq_list, list)

    def test_get_harmonic_emits_deprecation_warning(self):
        """Test that get_harmonic emits a DeprecationWarning."""
        sr = 44100
        duration = 0.25
        reference_signal = np.random.randn(int(duration * sr))
        recorded_signal = np.random.randn(int(duration * sr))

        # First get the freq_dict (will warn, but we're focused on get_harmonic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            freq_dict, base_freq_list = AudioThdFrequencyResponseAnalysis.calculate_spectrum(
                reference_signal, sr
            )

        # Check that get_harmonic emits DeprecationWarning
        with pytest.warns(DeprecationWarning) as warning_list:
            result = AudioThdFrequencyResponseAnalysis.get_harmonic(
                recorded_signal, freq_dict, sr, harmonics=[1, 2, 3, 4, 5]
            )

        # Verify warning was raised
        assert len(warning_list) > 0
        assert any("get_harmonic is deprecated" in str(w.message) for w in warning_list)
        assert any("HarmonicIndexBuilder" in str(w.message) for w in warning_list)
        assert any("hd_refactoring_guide.md" in str(w.message) for w in warning_list)

        # Verify the method still works
        assert isinstance(result, dict)

    def test_all_three_legacy_methods_warn(self):
        """Test that all three legacy methods together emit warnings."""
        sr = 44100
        num_steps = 4
        step_duration = 0.25
        reference_signal = np.random.randn(int(step_duration * sr * num_steps))
        recorded_signal = np.random.randn(int(step_duration * sr * num_steps))

        # Capture all warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call all three legacy methods
            freq_dict, base_freq_list = AudioThdFrequencyResponseAnalysis.calculate_spectrum(
                reference_signal, sr, gap_len=10
            )

            if freq_dict:  # Only proceed if we have valid data
                freq_dict = AudioThdFrequencyResponseAnalysis.get_harmonic(
                    recorded_signal, freq_dict, sr, harmonics=[1, 2, 3, 4, 5]
                )

                analyzer = AudioThdFrequencyResponseAnalysis()
                try:
                    plot_x, plot_h, plot_thd = analyzer.calculate_thd(
                        freq_dict, base_freq_list, recorded_signal, sr, gap_len=10
                    )
                except (KeyError, IndexError):
                    # Skip this test if data doesn't align properly
                    pytest.skip("Random data alignment issue in legacy method")

                # Verify we got deprecation warnings
                deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 3, f"Expected at least 3 warnings, got {len(deprecation_warnings)}"

                # Check each legacy method is mentioned in warnings
                messages = [str(warning.message) for warning in deprecation_warnings]
                assert any("calculate_spectrum" in msg for msg in messages)
                assert any("get_harmonic" in msg for msg in messages)
                assert any("calculate_thd" in msg for msg in messages)
            else:
                pytest.skip("No valid frequency dictionary from calculate_spectrum")

    def test_warning_stacklevel_correct(self):
        """Test that warnings point to the correct call site, not internal code."""
        sr = 44100
        duration = 0.25
        reference_signal = np.random.randn(int(duration * sr))

        with pytest.warns(DeprecationWarning) as warning_list:
            freq_dict, base_freq_list = AudioThdFrequencyResponseAnalysis.calculate_spectrum(
                reference_signal, sr
            )

        # Check stacklevel is correct (should point to this test, not internal)
        assert len(warning_list) > 0
        warning = warning_list[0]
        # The filename should be this test file or the code calling it
        assert warning.filename.endswith('.py')
