from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    custom signals
    """
    run_test_sign = pyqtSignal()

    get_result_file_sign = pyqtSignal(int)

    set_result_file_sign = pyqtSignal(int, str, str)

    test_insert_data_into_db_sign = pyqtSignal(str)

    update_mode_display_sign = pyqtSignal(int)

sign = MySignals()
