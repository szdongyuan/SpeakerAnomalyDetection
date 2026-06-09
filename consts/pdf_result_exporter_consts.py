import re


INVALID_FILENAME_CHARS_RE = re.compile(r"[<>:\"/\\\\|?*]")
MAX_RENDERED_VALUE_LENGTH = 160
RESULT_FIELD_LABELS = {
    "frequency_list": "频率点",
    "signal_duration": "时间轴",
    "recorded_signal": "录音信号",
    "signal_spl": "声压级",
    "spl_db": "声压级",
    "spl_db_raw": "原始声压级",
    "freq_value": "频率点",
    "harmonic": "谐波",
    "thd": "失真/响度值",
    "thd_raw": "原始失真/响度值",
    "fr": "频响",
    "fr_raw": "原始频响",
    "band_centers": "频段中心频率",
    "band_levels_db": "各频段声压级",
    "band_levels_weighted_db": "加权各频段声压级",
    "overall_db": "总声压级",
    "overall_weighted_db": "计权总声压级",
    "weighting": "计权方式",
    "exceeded_bands": "超限频段",
}
