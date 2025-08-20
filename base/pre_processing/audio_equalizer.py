from typing import Optional, Callable, Tuple, List, Union

import numpy as np
from scipy.signal import windows

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

        if trans_width_hz == 0:     #don't need set transition width
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
        if start_freq < 0 or end_freq < 0:
            raise ValueError("start_freq and end_freq must be positive")
        if start_freq > end_freq:
            start_freq, end_freq = end_freq, start_freq
        
        n_samples = len(audio_signal)
        
        if gain_mode == 'db':
            linear_gain = AudioEqualizer._db_to_linear(gain)
        else:
            linear_gain = gain
        
        windowed_signal, window = AudioEqualizer._apply_window(audio_signal, window_type)
        
        # Use rFFT to avoid manual positive/negative frequency mirroring and odd/even length pitfalls
        spectrum = np.fft.rfft(windowed_signal, n=n_samples)
        freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)
        
        gain_array = AudioEqualizer._calculate_gain_array(
            freqs,
            start_freq,
            end_freq,
            linear_gain,
            transition_width,
            transition_type,
            complement_mode_fre
        )
        
        modified_spectrum = spectrum * gain_array
        equalized_signal = np.fft.irfft(modified_spectrum, n=n_samples)
        equalized_signal = AudioEqualizer._remove_window_effect(equalized_signal, window)
        
        return equalized_signal
    
    @staticmethod
    def apply_multi_band_equalizer(
        audio_signal: np.ndarray,
        sample_rate: float,
        frequency_ranges: List[Tuple[float, float]],
        gains: Union[float, List[float]] = 0.0,
        gain_mode: str = 'linear',
        window_type: Optional[str] = None,
        transition_width: float = 0.1,
        transition_type: str = 'cosine',
        mode: str = 'cascade',
    ) -> np.ndarray:
        """
        Apply multi-band equalization using existing single-band equalizer.
        
        Two composition modes are supported:
        - 'sum': Extract per-band bandpass components and sum them together. Each
                 component is scaled by its corresponding gain. This emulates a
                 multi-band bandpass when gains are all 1.0.
                 (将通带叠加，一般适合于多通带滤波)
        - 'cascade': Sequentially apply in-place per-band equalization where the in-band
                     region is multiplied by its gain and outside remains 1.0. This
                     emulates a multi-band bandstop when gains are 0.0.
                 (将阻带级联，一般适合于多阻带滤波)
        Args:
            audio_signal: Input time-domain signal.
            sample_rate: Sample rate in Hz.
            frequency_ranges: List of (start_freq, end_freq) pairs in Hz.
            gains: Either a single number applied to all bands or a list of per-band gains.
            gain_mode: 'linear' or 'db' for interpreting gains.
            window_type: Optional window function name.
            transition_width: Transition width as a fraction of band width.
            transition_type: 'cosine', 'linear', or 'sigmoid'.
            mode: 'sum' or 'cascade'.
        
        Returns:
            The processed signal.
        """
        if not frequency_ranges:
            return audio_signal
        
        # Normalize and validate ranges
        normalized_ranges: List[Tuple[float, float]] = []
        for start_freq, end_freq in frequency_ranges:
            if start_freq < 0 or end_freq < 0:
                continue
            if start_freq > end_freq:
                start_freq, end_freq = end_freq, start_freq
            if end_freq <= start_freq:
                continue
            normalized_ranges.append((float(start_freq), float(end_freq)))
        
        if not normalized_ranges:
            return audio_signal
        
        # Prepare gains list
        if isinstance(gains, list):
            if len(gains) != len(normalized_ranges):
                raise ValueError("Length of gains must match length of frequency_ranges")
            gains_list = gains
        else:
            gains_list = [gains] * len(normalized_ranges)
        
        def to_linear(value: float) -> float:
            return AudioEqualizer._db_to_linear(value) if gain_mode == 'db' else float(value)
        linear_gains: List[float] = [to_linear(g) for g in gains_list]
        
        if mode not in ('sum', 'cascade'):
            raise ValueError("mode must be 'sum' or 'cascade'")
        
        if mode == 'sum':
            accumulated = np.zeros_like(audio_signal)
            for (start_freq, end_freq), band_gain in zip(normalized_ranges, linear_gains):
                band_component = AudioEqualizer.apply_equalizer(
                    audio_signal=audio_signal,
                    sample_rate=float(sample_rate),
                    start_freq=float(start_freq),
                    end_freq=float(end_freq),
                    gain=0.0,
                    gain_mode='linear',
                    window_type=window_type,
                    transition_width=transition_width,
                    transition_type=transition_type,
                    complement_mode_fre=True,
                )
                if band_gain != 1.0:
                    band_component = band_component * band_gain
                accumulated += band_component
            return accumulated
        else:
            result = audio_signal.copy()
            for (start_freq, end_freq), band_gain in zip(normalized_ranges, linear_gains):
                result = AudioEqualizer.apply_equalizer(
                    audio_signal=result,
                    sample_rate=float(sample_rate),
                    start_freq=float(start_freq),
                    end_freq=float(end_freq),
                    gain=float(band_gain),
                    gain_mode='linear',
                    window_type=window_type,
                    transition_width=transition_width,
                    transition_type=transition_type,
                    complement_mode_fre=False,
                )
            return result
    