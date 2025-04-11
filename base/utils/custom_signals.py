from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    custom signals
    """
    run_test_sign = pyqtSignal()

sign = MySignals()
