"""ui.ui_analysis_config package."""

try:
    from .loudness_config_dialog import LoudnessConfigWindow
except ImportError:
    LoudnessConfigWindow = None

try:
    from .roughness_config_dialog import RoughnessConfigWindow, default_roughness_config
except ImportError:
    RoughnessConfigWindow = None
    default_roughness_config = lambda: {}

try:
    from .sharpness_config_dialog import SharpnessConfigWindow, default_sharpness_config
except ImportError:
    SharpnessConfigWindow = None
    default_sharpness_config = lambda: {}
