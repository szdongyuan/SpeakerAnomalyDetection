import os

from PyQt5.QtGui import QIcon, QPixmap, QRegion

from consts.running_consts import DEFAULT_DIR


FOLDER_YELLOW_ICON_PATH = os.path.join(
    DEFAULT_DIR,
    "ui",
    "ui_pic",
    "folder",
    "folder-yellow.png",
)


def load_path_selector_folder_icon():
    pixmap = QPixmap(FOLDER_YELLOW_ICON_PATH)
    if pixmap.isNull():
        return QIcon(FOLDER_YELLOW_ICON_PATH)

    visible_rect = QRegion(pixmap.mask()).boundingRect()
    if visible_rect.isValid():
        pixmap = pixmap.copy(visible_rect)
    return QIcon(pixmap)
