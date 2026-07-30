"""Shared curve-style constants for acoustic analysis plots."""


MAIN_CURVE_COLOR = "main_curve_color"
UPPER_LIMIT_COLOR = "upper_limit_color"
LOWER_LIMIT_COLOR = "lower_limit_color"

DEFAULT_CURVE_COLORS = {
    MAIN_CURVE_COLOR: "#33C44D",
    UPPER_LIMIT_COLOR: "#800080",
    LOWER_LIMIT_COLOR: "#800080",
}

PLOT_VIEW_KEY = "plot_view"
DEFAULT_PLOT_VIEW_CONFIG = {
    "x_enabled": False,
    "x_min": None,
    "x_max": None,
    "y_enabled": False,
    "y_min": None,
    "y_max": None,
}
PLOT_VIEW_DIALOG_WIDTH = 700
PLOT_VIEW_MIN_VALUE = -1_000_000_000.0
PLOT_VIEW_MAX_VALUE = 1_000_000_000.0
PLOT_VIEW_DECIMALS = 3
PLOT_VIEW_DEFAULT_SINGLE_STEP = 1.0
PLOT_VIEW_TIME_SINGLE_STEP = 0.001

CURVE_COLOR_FIELDS = (
    (MAIN_CURVE_COLOR, "主曲线颜色"),
    (UPPER_LIMIT_COLOR, "上限颜色"),
    (LOWER_LIMIT_COLOR, "下限颜色"),
)

PRESET_CURVE_COLORS = (
    ("墨绿", "#14532D"),
    ("深绿", "#166534"),
    ("森林绿", "#15803D"),
    ("翠绿", "#16A34A"),
    ("绿色", "#33C44D"),
    ("亮绿", "#4ADE80"),
    ("浅绿", "#86EFAC"),
    ("雾绿", "#BBF7D0"),
    ("深青", "#164E63"),
    ("暗青", "#155E75"),
    ("蓝绿", "#0E7490"),
    ("青色", "#0891B2"),
    ("湖青", "#06B6D4"),
    ("亮青", "#22D3EE"),
    ("浅青", "#67E8F9"),
    ("雾青", "#A5F3FC"),
    ("深蓝", "#1E3A8A"),
    ("藏蓝", "#1E40AF"),
    ("浓蓝", "#1D4ED8"),
    ("蓝色", "#2563EB"),
    ("亮蓝", "#3B82F6"),
    ("天蓝", "#60A5FA"),
    ("浅蓝", "#93C5FD"),
    ("雾蓝", "#BFDBFE"),
    ("深靛", "#312E81"),
    ("暗靛", "#3730A3"),
    ("靛蓝", "#4338CA"),
    ("蓝紫", "#4F46E5"),
    ("亮靛", "#6366F1"),
    ("浅靛", "#818CF8"),
    ("淡靛", "#A5B4FC"),
    ("雾靛", "#C7D2FE"),
    ("紫色", "#800080"),
    ("深洋红", "#86198F"),
    ("暗洋红", "#A21CAF"),
    ("洋红", "#C026D3"),
    ("亮洋红", "#D946EF"),
    ("浅洋红", "#E879F9"),
    ("粉紫", "#F0ABFC"),
    ("雾紫", "#F5D0FE"),
    ("深红", "#7F1D1D"),
    ("暗红", "#991B1B"),
    ("浓红", "#B91C1C"),
    ("红色", "#DC2626"),
    ("亮红", "#EF4444"),
    ("珊瑚红", "#F87171"),
    ("浅红", "#FCA5A5"),
    ("雾红", "#FECACA"),
    ("深棕", "#78350F"),
    ("棕色", "#92400E"),
    ("棕黄", "#B45309"),
    ("琥珀", "#D97706"),
    ("橙黄", "#F59E0B"),
    ("金黄", "#FBBF24"),
    ("浅黄", "#FCD34D"),
    ("雾黄", "#FDE68A"),
    ("纯黑", "#000000"),
    ("黑色", "#171717"),
    ("炭灰", "#262626"),
    ("深灰", "#404040"),
    ("石墨灰", "#525252"),
    ("中灰", "#737373"),
    ("浅灰", "#A3A3A3"),
    ("雾灰", "#D4D4D4"),
)
