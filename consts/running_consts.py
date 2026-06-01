import os
import sys

# DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"
DEFAULT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\", "/") + "/"

# basic consts
KB = 1 << 10
MB = 1 << 20
GB = 1 << 30

# log consts
LOG_DIR = DEFAULT_DIR + "log/"

DEFAULT_LOG_FORMATTER = "[%(asctime)s][%(name)s] - [%(levelname)s] - [%(message)s] [%(filename)s:%(lineno)d]"

SEQUENCE_CONFIG_REGISTRY_PATH = DEFAULT_DIR + "ui/ui_config/sequence_config_registry.json"
AUDIO_DEVICE_CONFIG_PATH = os.path.join(DEFAULT_DIR, "ui", "ui_config", "last_audio_devices.json")
SN_REGEX_RULES_JSON_PATH = os.path.join(DEFAULT_DIR, "ui", "ui_config", "sn_regex_rules.json")

VALID_ACQUISITION_MODES = {
    "PLAY_AND_RECORD",
    "RECORD_ONLY",
}

DEFAULT_PLAY_AND_RECORD_DETAIL = {"use_streaming_recording": False}
DEFAULT_RECORD_ONLY_DETAIL = {
    "total_time": 4.0,
    "sample_rate": 44100,
    "monitor_playback": False,
    "monitor_input_channel": 0,
    "monitor_gain_db": 0.0,
    "use_streaming_recording": False,
}

DEFAULT_LOG = {
    "log_name": LOG_DIR + "main.log",
    "max_size": 2 * MB,
    "backup_count": 9,
    "log_format": DEFAULT_LOG_FORMATTER,
}
AI_LOG = {
    "log_name": LOG_DIR + "ai.log",
    "max_size": 2 * MB,
    "backup_count": 9,
    "log_format": DEFAULT_LOG_FORMATTER,
}
DEBUG_LOG = {
    "log_name": LOG_DIR + "debug.log",
    "max_size": 1 * MB,
    "backup_count": 0,
    "log_format": DEFAULT_LOG_FORMATTER,
}

TEST_LOG = {
    "log_name": LOG_DIR + "test.log",
    "max_size": 100 * KB,
    "backup_count": 0,
    "log_format": DEFAULT_LOG_FORMATTER,
}

LOG_MAPPING = {
    "core": DEFAULT_LOG,
    "train": AI_LOG,
    "evaluate": AI_LOG,
    "predict": AI_LOG,
    "debug": DEBUG_LOG,
    "test": TEST_LOG,
    "db_core": DEFAULT_LOG,
    "soundcard_core": DEFAULT_LOG,
}

tcp_service_recv_bytes = 1024

MODULES_LOAD = [
    ("加载格式", "consts.ui_style_const"),
    ("加载常量", "consts.model_consts"),
    ("加载路径", "consts.running_consts"),
    ("加载日志模块", "base.log_manager"),
    ("加载数据库模块", "base.db_manager"),
    ("加载 AI 模型训练模块", "ui.ai_window"),
    ("加载 校准模块", "ui.calibration_window"),
    ("加载 硬件选择模块", "ui.hardware_window"),
    ("加载 分析流程模块", "ui.sequence.sequence_widget"),
    ("加载 激励信号模块", "ui.stimulus_window"),
    ("加载 模型选择模块", "ui.operation_sequence"),
    ("加载 登录模块", "ui.login_window"),
    ("加载完成", None),
]
