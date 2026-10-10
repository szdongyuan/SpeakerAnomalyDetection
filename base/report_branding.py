"""Company artwork for manually exported analysis reports."""

from pathlib import Path

from PyQt5.QtGui import QImage

from base.analysis_report_layout import PDF_RESOLUTION
from consts.running_consts import DEFAULT_DIR


COMPANY_LOGO_PATH = Path(DEFAULT_DIR) / "ui/ui_pic/logo_pic/公司-LOGO.png"
COMPANY_LOGO_WIDTH_MM = 30.0


def load_company_logo():
    """Return the complete original PNG and its proportional report display size."""
    try:
        data = COMPANY_LOGO_PATH.read_bytes()
    except OSError as error:
        raise ValueError("无法读取公司 Logo：公司-LOGO.png") from error
    image = QImage.fromData(data, "PNG")
    if image.isNull():
        raise ValueError("公司 Logo 图片无效：公司-LOGO.png")
    # Share the renderer DPI and preserve the original pixels and aspect ratio.
    target_width = COMPANY_LOGO_WIDTH_MM * PDF_RESOLUTION / 25.4
    scale = min(target_width / image.width(), 1.0)
    width = max(1, round(image.width() * scale))
    height = max(1, round(image.height() * scale))
    return data, width, height
