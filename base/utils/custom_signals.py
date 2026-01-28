from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    custom signals, only responsible for signal define
    """
    # Only keep streaming audio chunks for real-time waveform display.
    stream_audio_chunk_signal = pyqtSignal(object)  # Emits numpy array chunks for streaming audio


sign = MySignals()
