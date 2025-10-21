from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    custom signals, only responsible for signal define
    """
    signal_emitter = pyqtSignal(str)

    run_test_sign = pyqtSignal(str)

    get_result_file_sign = pyqtSignal(int)

    set_result_file_sign = pyqtSignal(int, str, str)

    test_insert_data_into_db_sign = pyqtSignal(str)

    update_mode_display_sign = pyqtSignal(int)

    stream_audio_chunk_signal = pyqtSignal(object)  # Emits numpy array chunks for streaming audio


sign = MySignals()
