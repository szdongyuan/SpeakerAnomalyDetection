import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import windows

from typing import Optional, Callable, Tuple


class AudioEqualizer:
    """
    Simplified FFT-based audio equalizer for a single frequency band.
    """
    
    @staticmethod
    def _db_to_linear(db_value: float) -> float:
        """Convert dB to linear gain factor."""
        return 10 ** (db_value / 20)
    
    @staticmethod
    def _linear_to_db(linear_value: float) -> float:
        """Convert linear gain factor to dB."""
        if linear_value <= 0:
            return -np.inf
        return 20 * np.log10(linear_value)
    
    @staticmethod
    def _get_transition_function(transition_type: str = 'cosine') -> Callable:
        """
        Get transition function for smooth band edges.
        
        Args:
            transition_type: Type of transition function ('cosine', 'linear', 'sigmoid')
            
        Returns:
            Transition function that maps [0, 1] to [0, 1]
        """
        if transition_type == 'cosine':
            return lambda x: 0.5 * (1 + np.cos(np.pi * (1 - x)))
        elif transition_type == 'linear':
            return lambda x: x
        elif transition_type == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-10 * (x - 0.5)))
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
    
    @staticmethod
    def _apply_window(signal: np.ndarray, window_type: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply window function to signal.
        
        Args:
            signal: Input signal
            window_type: Type of window function (None for no window)
            
        Returns:
            Tuple of (windowed_signal, window_array or None)
        """
        if window_type is not None:
            window = windows.get_window(window_type, len(signal))
            return signal * window, window
        return signal.copy(), None
    
    @staticmethod
    def _remove_window_effect(signal: np.ndarray, window: Optional[np.ndarray]) -> np.ndarray:
        """
        Remove window effect from signal.
        
        Args:
            signal: Windowed signal
            window: Window array used
            
        Returns:
            Signal with window effect removed
        """
        if window is not None:
            safe_window = window.copy()
            safe_window[safe_window < 1e-10] = 1e-10
            return signal / safe_window
        return signal
    
    @staticmethod
    def _calculate_gain_array(
        freqs: np.ndarray,
        start_freq: float,
        end_freq: float,
        linear_gain: float,
        transition_width: float,
        transition_type: str,
        complement_mode: bool
    ) -> np.ndarray:
        """
        Calculate frequency-dependent gain array.
        
        Args:
            freqs: Frequency array
            start_freq: Start frequency of the band
            end_freq: End frequency of the band
            linear_gain: Linear gain value to apply
            transition_width: Transition width as fraction of band width
            transition_type: Type of transition function
            complement_mode: If True, apply gain to frequencies outside the band
            
        Returns:
            Gain array for each frequency
        """
        n_freqs = len(freqs)
        gain_array = np.ones(n_freqs)

        band_width = end_freq - start_freq
        
        if band_width <= 0:
            if complement_mode:
                gain_array[:] = linear_gain 
                if n_freqs > 0 and band_width == 0 :
                     pass
            else:
                pass
            return gain_array

        trans_width_hz = band_width * transition_width
        trans_func = AudioEqualizer._get_transition_function(transition_type)

        if trans_width_hz == 0:     #无需设置过渡带
            if complement_mode:
                condition_outside_band = (freqs < start_freq) | (freqs > end_freq)
                gain_array[condition_outside_band] = linear_gain
            else:
                condition_inside_band = (freqs >= start_freq) & (freqs <= end_freq)
                gain_array[condition_inside_band] = linear_gain
            return gain_array

        lower_trans_start = start_freq - trans_width_hz
        upper_trans_end = end_freq + trans_width_hz

        cond_below_lower_trans = freqs < lower_trans_start
        cond_in_lower_trans = (freqs >= lower_trans_start) & (freqs < start_freq)
        cond_in_band = (freqs >= start_freq) & (freqs <= end_freq)
        cond_in_upper_trans = (freqs > end_freq) & (freqs <= upper_trans_end)
        cond_above_upper_trans = freqs > upper_trans_end

        if complement_mode:
            gain_array[cond_below_lower_trans] = linear_gain
            gain_array[cond_above_upper_trans] = linear_gain
            gain_array[cond_in_band] = 1.0

            if np.any(cond_in_lower_trans):
                t = (freqs[cond_in_lower_trans] - lower_trans_start) / trans_width_hz
                transition_val = trans_func(t)
                gain_array[cond_in_lower_trans] = linear_gain + (1.0 - linear_gain) * transition_val
            
            if np.any(cond_in_upper_trans):
                t = (freqs[cond_in_upper_trans] - end_freq) / trans_width_hz
                transition_val = trans_func(t)
                gain_array[cond_in_upper_trans] = 1.0 + (linear_gain - 1.0) * transition_val
        else:
            gain_array[cond_in_band] = linear_gain

            if np.any(cond_in_lower_trans):
                t = (freqs[cond_in_lower_trans] - lower_trans_start) / trans_width_hz
                transition_val = trans_func(t)
                gain_array[cond_in_lower_trans] = 1.0 + (linear_gain - 1.0) * transition_val

            if np.any(cond_in_upper_trans):
                t = (freqs[cond_in_upper_trans] - end_freq) / trans_width_hz
                transition_val = trans_func(1 - t) 
                gain_array[cond_in_upper_trans] = 1.0 + (linear_gain - 1.0) * transition_val
        
        return gain_array
    
    @staticmethod
    def _apply_gain_to_spectrum(
        spectrum: np.ndarray,
        gain_array: np.ndarray
    ) -> np.ndarray:
        """
        Apply gain array to frequency spectrum.
        
        Args:
            spectrum: FFT spectrum
            gain_array: Gain array for positive frequencies
            
        Returns:
            Modified spectrum
        """
        n_samples = len(spectrum)
        modified_spectrum = spectrum.copy()
        
        modified_spectrum[:n_samples//2] *= gain_array
        modified_spectrum[n_samples//2:] *= gain_array[::-1]
        
        return modified_spectrum
    
    @staticmethod
    def apply_equalizer(
        audio_signal: np.ndarray,
        sample_rate: float,
        start_freq: float,
        end_freq: float,
        gain: float = 0.0,
        gain_mode: str = 'linear',
        window_type: Optional[str] = None,
        transition_width: float = 0.1,
        transition_type: str = 'cosine',
        complement_mode_fre: bool = False
    ) -> np.ndarray:
        """
        Apply equalizer to audio signal for a single frequency band.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            start_freq: Start frequency of the band
            end_freq: End frequency of the band
            gain: Gain value to apply
            gain_mode: 'linear' or 'db'
            transition_width: Transition width as fraction of band width
            transition_type: Type of transition function ('cosine', 'linear', 'sigmoid')
            window_type: Window function to apply (None for no window)
            complement_mode: If True, apply gain to frequencies outside the band
            
        Returns:
            Equalized audio signal
        """
        n_samples = len(audio_signal)
        
        if gain_mode == 'db':
            linear_gain = AudioEqualizer._db_to_linear(gain)
        else:
            linear_gain = gain
        
        windowed_signal, window = AudioEqualizer._apply_window(audio_signal, window_type)
        
        spectrum = fft(windowed_signal)
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
        positive_freqs = freqs[:n_samples//2]
        
        gain_array = AudioEqualizer._calculate_gain_array(
            positive_freqs,
            start_freq,
            end_freq,
            linear_gain,
            transition_width,
            transition_type,
            complement_mode_fre
        )
        
        modified_spectrum = AudioEqualizer._apply_gain_to_spectrum(spectrum, gain_array)
        equalized_signal = np.real(ifft(modified_spectrum))
        equalized_signal = AudioEqualizer._remove_window_effect(equalized_signal, window)
        
        return equalized_signal