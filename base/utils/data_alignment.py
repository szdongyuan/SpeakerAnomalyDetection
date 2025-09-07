import librosa
import numpy as np
from scipy.signal import correlate, find_peaks

class data_alignment(object):
    def __init__(self, audio_data, input_data, sr=44100):
        self.sr = sr
        self.audio_data = audio_data
        self.input_data = input_data

    def onset_detect(self):
        """
        This algorithm is used for the calculation of long time delay of music signals,
        may not be suitable for loudspeaker signals
        """
        onsets_frames = librosa.onset.onset_detect(y=self.audio_data, sr=self.sr)
        onsets_times = librosa.frames_to_time(onsets_frames, sr=self.sr)
        first_onset_time = onsets_times[0] if len(onsets_times) > 0 else None

        music_delay_frames = int(first_onset_time * self.sr) if first_onset_time is not None else 0

        return music_delay_frames, first_onset_time, onsets_times

    def cross_correlation(self):
        cross_correlation = correlate(self.input_data, self.audio_data, mode="full")
        delay = np.argmax(cross_correlation) - (len(self.audio_data) - 1)
        delay_frames = np.abs(delay)
        delay_seconds = delay_frames / self.sr

        return delay_frames, delay_seconds

    def phase_correlation(self, input_data, audio_data):
        min_len = min(len(input_data), len(audio_data))
        x_trimmed = input_data[:min_len]
        y_trimmed = audio_data[:min_len]

        # fft
        X = np.fft.fft(x_trimmed)
        Y = np.fft.fft(y_trimmed)

        # Phase difference
        R = X * np.conj(Y)  # frequency multiply
        R /= np.abs(R)  # normalization

        # ifft
        r = np.fft.ifft(R).real

        delay_index = np.argmax(r)
        if delay_index > min_len // 2:
            delay_index -= min_len

        delay_time = delay_index

        return delay_time, r


def find_signal_peak(signal_data):
    """
    Find the peak index in a signal
    
    Args:
        signal_data: Input signal data array
        
    Returns:
        int: Index of the peak in the signal
    """
    try:
        signal_array = np.asarray(signal_data).flatten()
        abs_signal = np.abs(signal_array)
        peaks, _ = find_peaks(abs_signal)
        if len(peaks) > 0:
            peak_idx = int(peaks[int(np.argmax(abs_signal[peaks]))])
        else:
            peak_idx = int(np.argmax(abs_signal))
        return peak_idx
    except Exception:
        return 0


def _calc_left_right_from_array(pattern_segment: np.ndarray):
    """
    Calculate the left and right grid points from the array
    This function is copied from PipelinePdPm for consistency
    
    Args:
        pattern_segment: Pattern signal array
        
    Returns:
        tuple: (peak_idx, left_point, right_point)
    """
    pattern_segment = np.asarray(pattern_segment).astype(float)
    
    # Handle multi-dimensional (dual-channel) arrays
    if pattern_segment.ndim > 1:
        # Use first channel for window calculation
        pattern_segment = pattern_segment[0]
    
    peak_idx = find_signal_peak(pattern_segment)
    
    # Calculate left and right points around the peak
    signal_length = len(pattern_segment)
    
    # Use 25% of signal length on each side as default window
    default_window = max(1, signal_length // 4)
    
    left_point = min(peak_idx, default_window)
    right_point = min(signal_length - peak_idx, default_window)
    
    return peak_idx, left_point, right_point


def align_signals_by_peaks(target_data, pattern_data, behavior="min"):
    """
    Align target and pattern signals by their peaks and match lengths
    
    Args:
        target_data: Target signal data array
        pattern_data: Pattern signal data array
        behavior: Alignment behavior mode:
                 - "min": Cut to intersection length if distance not enough
                 - "padding": Zero padding for target, keep pattern same
        
    Returns:
        tuple: (aligned_target_segment, aligned_pattern_segment)
    """
    try:
        # Find peaks for both signals
        target_peak_idx = find_signal_peak(target_data)
        pattern_peak_idx = find_signal_peak(pattern_data)
        
        # Calculate segment length from pattern peak
        _, left_point, right_point = _calc_left_right_from_array(pattern_data)
        segment_length = left_point + right_point
        
        if behavior == "padding":
            # Zero padding mode: pad target if needed, keep pattern unchanged
            target_start = target_peak_idx - left_point
            target_end = target_start + segment_length
            
            # Handle padding for target if indices are out of bounds
            if target_start < 0:
                # Pad at beginning
                padding_left = -target_start
                target_segment = np.pad(target_data, (padding_left, 0), mode='constant', constant_values=0)
                target_start = 0
                target_end = segment_length
            elif target_end > len(target_data):
                # Pad at end
                padding_right = target_end - len(target_data)
                target_segment = np.pad(target_data, (0, padding_right), mode='constant', constant_values=0)
                target_start = target_peak_idx - left_point
                target_end = target_start + segment_length
            else:
                target_segment = target_data
            
            target_segment = target_segment[target_start:target_end]
            
            # Extract pattern segment normally
            pattern_start = max(0, pattern_peak_idx - left_point)
            pattern_end = min(len(pattern_data), pattern_start + segment_length)
            pattern_segment = pattern_data[pattern_start:pattern_end]
            
        else:  # behavior == "min" (default)
            # Min mode: cut to intersection length
            target_start = max(0, target_peak_idx - left_point)
            target_end = min(len(target_data), target_start + segment_length)
            target_segment = target_data[target_start:target_end]
            
            pattern_start = max(0, pattern_peak_idx - left_point) 
            pattern_end = min(len(pattern_data), pattern_start + segment_length)
            pattern_segment = pattern_data[pattern_start:pattern_end]
            
            # Ensure both segments have same length (intersection)
            min_len = min(len(target_segment), len(pattern_segment))
            target_segment = target_segment[:min_len]
            pattern_segment = pattern_segment[:min_len]
        
        return target_segment, pattern_segment
    except Exception:
        # If alignment fails, return original data
        return target_data, pattern_data


def align_signals_with_peak(target_data, pattern_data, external_peak, behavior="min"):
    """
    Align signals using external peak information
    
    Args:
        target_data: Target signal data array
        pattern_data: Pattern signal data array
        external_peak: External peak index to use for target data
        behavior: Alignment behavior mode:
                 - "min": Cut to intersection length if distance not enough
                 - "padding": Zero padding for target, keep pattern same
        
    Returns:
        tuple: (aligned_target_segment, aligned_pattern_segment)
    """
    try:
        # Use external peak for target data
        target_peak_idx = external_peak if external_peak < len(target_data) else find_signal_peak(target_data)
        
        # Calculate segment length from pattern peak
        _, left_point, right_point = _calc_left_right_from_array(pattern_data)
        segment_length = left_point + right_point
        
        if behavior == "padding":
            # Zero padding mode: pad target if needed, keep pattern unchanged
            target_start = target_peak_idx - left_point
            target_end = target_start + segment_length
            
            # Handle padding for target if indices are out of bounds
            if target_start < 0:
                # Pad at beginning
                padding_left = -target_start
                target_segment = np.pad(target_data, (padding_left, 0), mode='constant', constant_values=0)
                target_start = 0
                target_end = segment_length
            elif target_end > len(target_data):
                # Pad at end
                padding_right = target_end - len(target_data)
                target_segment = np.pad(target_data, (0, padding_right), mode='constant', constant_values=0)
                target_start = target_peak_idx - left_point
                target_end = target_start + segment_length
            else:
                target_segment = target_data
            
            target_segment = target_segment[target_start:target_end]
            
            # Extract pattern segment normally
            pattern_start = max(0, find_signal_peak(pattern_data) - left_point)
            pattern_end = min(len(pattern_data), pattern_start + segment_length)
            pattern_segment = pattern_data[pattern_start:pattern_end]
            
        else:  # behavior == "min" (default)
            # Min mode: cut to intersection length
            target_start = max(0, target_peak_idx - left_point)
            target_end = min(len(target_data), target_start + segment_length)
            target_segment = target_data[target_start:target_end]
            
            pattern_start = max(0, find_signal_peak(pattern_data) - left_point)
            pattern_end = min(len(pattern_data), pattern_start + segment_length)
            pattern_segment = pattern_data[pattern_start:pattern_end]
            
            # Ensure both segments have same length (intersection)
            min_len = min(len(target_segment), len(pattern_segment))
            target_segment = target_segment[:min_len]
            pattern_segment = pattern_segment[:min_len]
        
        return target_segment, pattern_segment
    except Exception:
        # If alignment fails, return original data
        return target_data, pattern_data

