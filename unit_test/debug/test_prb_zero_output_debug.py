"""
Diagnostic test to trace why PRB curve outputs all zeros.

This test instruments each layer to identify where zeros are introduced.
"""
import numpy as np
import unittest
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD


class TestPrbZeroOutputDebug(unittest.TestCase):
    """Debug PRB zero output issue with real-world data simulation."""

    def test_perceptual_thd_with_synthetic_signal(self):
        """Test perceptual calculation with synthetic signal containing harmonics."""

        # Create synthetic signal: 1 kHz fundamental + harmonics
        sample_rate = 48000
        duration = 0.1  # 100ms step
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Fundamental at 1 kHz with amplitude 0.5
        fundamental_freq = 1000.0
        fundamental_amplitude = 0.5
        signal = fundamental_amplitude * np.sin(2 * np.pi * fundamental_freq * t)

        # Add harmonics (10th to 15th order) with decreasing amplitudes
        for h in range(10, 16):
            harmonic_amplitude = 0.05 / h  # Decreasing amplitude
            signal += harmonic_amplitude * np.sin(2 * np.pi * fundamental_freq * h * t)

        print(f"\n=== Layer 1: Input Signal ===")
        print(f"Signal RMS: {np.sqrt(np.mean(signal**2)):.6f}")
        print(f"Signal max: {np.max(np.abs(signal)):.6f}")

        # Create stimulus metadata for single step
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 1000.0,
            'stop_freq': 1000.0,
            'num_steps': 1,
            'total_time': duration,
            'repeat_times': 1,
            'sample_rate': sample_rate
        }

        harmonic_orders = [10, 11, 12, 13, 14, 15]

        # Call the perceptual calculation
        atfra = AudioThdFrequencyResponseAnalysis()
        thd_kwargs = {
            'stimulus_metadata': stimulus_metadata,
            'harmonic_orders': harmonic_orders
        }

        print(f"\n=== Layer 2: Calling _calculate_perceptual_thd_three_phase ===")
        freq_value, harmonic, perceptual_loudness = atfra._calculate_perceptual_thd_three_phase(
            signal, sample_rate, thd_kwargs
        )

        print(f"\n=== Layer 3: Results ===")
        print(f"freq_value: {freq_value}")
        print(f"harmonic: {harmonic}")
        print(f"perceptual_loudness: {perceptual_loudness}")
        print(f"perceptual_loudness shape: {perceptual_loudness.shape}")
        print(f"perceptual_loudness sum: {np.sum(perceptual_loudness):.6f}")
        print(f"perceptual_loudness max: {np.max(perceptual_loudness):.6f}")

        # The issue: Are we getting all zeros?
        if np.all(perceptual_loudness == 0):
            print("\n!!! PROBLEM: All perceptual_loudness values are 0 !!!")

            # Let's manually check intermediate values
            print("\n=== Manual Check: Direct computation ===")
            analyzer = PerceptualStepSignalHD(sample_rate)

            # Compute STFT manually
            from scipy import signal as scipy_signal
            f, t_stft, Zxx = scipy_signal.stft(
                signal,
                fs=sample_rate,
                window='hann',
                nperseg=len(signal),
                noverlap=0
            )
            spectrum = np.abs(Zxx)

            print(f"STFT spectrum shape: {spectrum.shape}")
            print(f"STFT spectrum max: {np.max(spectrum):.6f}")
            print(f"STFT spectrum sum: {np.sum(spectrum):.6f}")

            # Find fundamental bin
            fund_bin = np.argmin(np.abs(f - fundamental_freq))
            print(f"Fundamental bin: {fund_bin}, freq: {f[fund_bin]:.2f} Hz")
            print(f"Fundamental amplitude in spectrum: {spectrum[fund_bin, 0]:.6f}")

            # Find harmonic bins
            for h in harmonic_orders:
                h_freq = fundamental_freq * h
                h_bin = np.argmin(np.abs(f - h_freq))
                print(f"Harmonic {h} (bin {h_bin}, {f[h_bin]:.0f} Hz): amplitude = {spectrum[h_bin, 0]:.6f}")
        else:
            print(f"\n✓ SUCCESS: Got non-zero perceptual loudness values")
            print(f"  Mean: {np.mean(perceptual_loudness):.2f} phons")

        # Assert we get non-zero output
        self.assertGreater(np.sum(perceptual_loudness), 0,
                          "Perceptual loudness should be non-zero for signal with harmonics")


if __name__ == '__main__':
    unittest.main()
