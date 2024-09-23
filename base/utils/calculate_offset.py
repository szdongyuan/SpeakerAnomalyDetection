import librosa
import numpy as np

from scipy import signal


def calculate_offset(reference_signal, recorded_signal):
    corr = signal.correlate(recorded_signal, reference_signal)
    delay = np.argmax(abs(corr)) - len(reference_signal) + 1
    return delay
