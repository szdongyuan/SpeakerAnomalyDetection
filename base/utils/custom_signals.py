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
    # Wake-up only; payload is the source StreamingAudioProcessor, not audio.
    stream_audio_queue_ready_signal = pyqtSignal(object)
    # Automatic target completion; payload is the source processor.
    stream_audio_recording_finished_signal = pyqtSignal(object)


sign = MySignals()
