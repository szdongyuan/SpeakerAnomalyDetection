from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    custom signals, only responsible for signal define
    """
    # Streaming payload for real-time waveform display.
    # Emits either:
    # - dict {"mono": np.ndarray(frames,), "multi": np.ndarray(frames, channels)} (preferred)
    # - legacy np.ndarray(frames,) for backward compatibility
    stream_audio_chunk_signal = pyqtSignal(object)


sign = MySignals()
