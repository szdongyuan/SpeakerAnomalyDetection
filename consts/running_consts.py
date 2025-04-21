import os
import sys

DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"
# DEFAULT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\", "/") + "/"

# basic consts
KB = 1 << 10
MB = 1 << 20
GB = 1 << 30

# log consts
LOG_DIR = DEFAULT_DIR + "log/"

DEFAULT_LOG_FORMATTER = '[%(asctime)s][%(name)s] - [%(levelname)s] - [%(message)s] [%(filename)s:%(lineno)d]'

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
