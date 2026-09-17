"""One toolbar row with explicit input budgets and font-aware label budgets."""

from dataclasses import dataclass

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtWidgets import QLayout, QLayoutItem, QWidgetItem


@dataclass
class _Entry:
    item: QLayoutItem
    role: str
    minimum: int = 0
    preferred: int = 0


class ToolbarRowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self.setContentsMargins(4, 0, 4, 0)
        self.setSpacing(4)

    def addItem(self, item):
        self._entries.append(_Entry(item, "fixed", item.minimumSize().width(),
                                    item.sizeHint().width()))
        self.invalidate()

    def add_control(self, widget, role, minimum=0, preferred=0):
        self.addChildWidget(widget)
        self._entries.append(_Entry(QWidgetItem(widget), role, minimum, preferred))
        self.invalidate()

    def count(self):
        return len(self._entries)

    def itemAt(self, index):
        return self._entries[index].item if 0 <= index < self.count() else None

    def takeAt(self, index):
        if 0 <= index < self.count():
            item = self._entries.pop(index).item
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return Qt.Horizontal

    def _budgets(self, compact=False):
        entries = [entry for entry in self._entries if not entry.item.isEmpty()]
        minimums, preferred = [], []
        for entry in entries:
            if entry.role == "label":
                widget = entry.item.widget()
                width = widget.caption_width(compact)
                minimums.append(width)
                preferred.append(width)
            elif entry.role == "switch":
                width = entry.item.widget().sizeHint().width()
                minimums.append(width)
                preferred.append(width)
            elif entry.role == "serial":
                width = entry.minimum if compact else entry.preferred
                minimums.append(width)
                preferred.append(width)
            else:
                minimums.append(entry.minimum)
                preferred.append(max(entry.minimum, entry.preferred))
        return entries, minimums, preferred

    def _size(self, minimum):
        entries, minimums, preferred = self._budgets(compact=minimum)
        margins = self.contentsMargins()
        width = sum(minimums if minimum else preferred)
        width += max(0, len(entries) - 1) * self._mode_spacing(minimum)
        return QSize(width + margins.left() + margins.right(),
                     40 + margins.top() + margins.bottom())

    def minimumSize(self):
        return self._size(True)

    def sizeHint(self):
        return self._size(False)

    def _mode_spacing(self, compact):
        return 1 if compact else self.spacing()

    @staticmethod
    def _shrink(widths, minimums, indices, deficit):
        active = [index for index in indices if widths[index] > minimums[index]]
        while deficit > 0 and active:
            share = max(1, deficit // len(active))
            for index in active:
                take = min(widths[index] - minimums[index], share, deficit)
                widths[index] -= take
                deficit -= take
            active = [index for index in active if widths[index] > minimums[index]]
        return deficit

    def setGeometry(self, rect):
        super().setGeometry(rect)
        # Always compute the transition from full captions, never current paint
        # mode/size hints. Font changes therefore cannot produce feedback jitter.
        compact = rect.width() < self.sizeHint().width()
        entries, minimums, widths = self._budgets(compact)
        for entry in entries:
            if entry.role in ("label", "serial"):
                entry.item.widget().set_compact(compact)
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        spacing = self._mode_spacing(compact)
        available = area.width() - max(0, len(entries) - 1) * spacing
        deficit = max(0, sum(widths) - available)
        indices = [index for index, entry in enumerate(entries)
                   if entry.role in ("input", "sn")]
        self._shrink(widths, minimums, indices, deficit)
        # Preferred widths are caps. Unused space follows the final separator.
        x = area.x()
        for entry, width in zip(entries, widths):
            height = min(40, entry.item.maximumSize().height())
            y = area.y() + (area.height() - height) // 2
            entry.item.setGeometry(QRect(x, y, width, height))
            x += width + spacing
