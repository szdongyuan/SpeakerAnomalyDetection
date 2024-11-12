import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout


class QmyFigureCanvas(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.resize(300, 100)

        layout = QHBoxLayout()
        label_graph = QLabel("Show graph here...")
        label_graph.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_graph)

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QmyFigureCanvas()
    window.show()
    sys.exit(app.exec_())
