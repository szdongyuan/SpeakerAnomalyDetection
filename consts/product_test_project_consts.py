"""Constants for project-level product test configurations."""


# Registry schema
PROGRAM_REGISTRY_FILE = "program_registry.json"
REGISTRY_ACTIVE_FILE_KEY = "active_file"
REGISTRY_CONFIGS_KEY = "configs"
REGISTRY_FILE_KEY = "file"

# Project schema
PROJECT_NAME_KEY = "project_name"
RESULT_ROOT_DIRECTORY_KEY = "result_root_directory"
TEST_GROUPS_KEY = "test_groups"
GROUP_NAME_KEY = "group_name"
TEST_CONDITIONS_KEY = "test_conditions"
CONDITION_NAME_KEY = "condition_name"
TRIGGER_STATE_KEY = "trigger_state"
TEST_QUEUE_KEY = "test_queue"

# Runtime modes
PRODUCT_TRIGGER_MODE_MANUAL = "manual"
PRODUCT_TRIGGER_MODE_SERIAL = "serial"
PRODUCT_TRIGGER_MODE_MIXED = "mixed"

# Validation rules
INVALID_PROJECT_NAME_CHARS = '<>:"/\\|?*_'
MIXED_ACQUISITION_MODE_ERROR = (
    "同一产品测试配置不能同时包含导入音频和录制音频工况，"
    "请统一各工况测试队列的采集模式后重试"
)
LIMIT_RULE_ANALYSIS_TYPES = {
    "SPL",
    "SPLF",
    "FR",
    "HD",
    "RB",
    "PRB",
    "FFT",
    "FBA",
    "LOUD",
    "LOUDNESS",
}
