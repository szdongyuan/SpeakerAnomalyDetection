"""A modeless detail view of an existing condition plot's in-memory results."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout


class PlotDetailDialog(QDialog):
    def __init__(self, source):
        super().__init__(source, Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f'{source.station_name or "工位" + str(source.column + 1)} · {source.display.title}')
        self.resize(1000, 650)
        self.source = source
        self.cell = type(source)(source.display, self, row=source.row, column=source.column,
                                      station_name=source.station_name, detail_view=True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.cell)
        source.copy_results_to(self.cell)
        source.results_changed.connect(self.sync_results)

    def sync_results(self):
        self.source.copy_results_to(self.cell)

    def done(self, result):
        self.source.results_changed.disconnect(self.sync_results)
        self.cell.dispose()
        super().done(result)
