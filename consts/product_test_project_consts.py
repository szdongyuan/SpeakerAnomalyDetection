"""Constants for project-level product test configurations."""


# Registry schema
PROGRAM_REGISTRY_FILE = "program_registry.json"
REGISTRY_ACTIVE_FILE_KEY = "active_file"
REGISTRY_CONFIGS_KEY = "configs"
REGISTRY_FILE_KEY = "file"

# Project schema
PROJECT_NAME_KEY = "project_name"
RESULT_ROOT_DIRECTORY_KEY = "result_root_directory"
EXPORT_RAW_AUDIO_CSV_KEY = "export_raw_audio_csv"
TEST_GROUPS_KEY = "test_groups"
GROUP_NAME_KEY = "group_name"
TEST_CONDITIONS_KEY = "test_conditions"
CONDITION_NAME_KEY = "condition_name"
TRIGGER_STATE_KEY = "trigger_state"
TEST_QUEUE_KEY = "test_queue"
OUTPUT_LOAD_KEY = "output_load"

# Runtime modes
PRODUCT_TRIGGER_MODE_MANUAL = "manual"
PRODUCT_TRIGGER_MODE_SERIAL = "serial"
PRODUCT_TRIGGER_MODE_MIXED = "mixed"

# Validation rules
INVALID_PROJECT_NAME_CHARS = '<>:"/\\|?*_'
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
