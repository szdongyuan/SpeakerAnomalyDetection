import logging
from logging.handlers import RotatingFileHandler

from consts.running_consts import LOG_DIR, LOG_MAPPING, DEFAULT_LOG


class LogManager(object):

    def __init__(self, thread_holder="core"):
        self.logger = self.set_log_handler(thread_holder)

    @staticmethod
    def set_log_handler(thread_holder):
        logger = logging.getLogger(thread_holder)
        log_info = LOG_MAPPING.get(thread_holder, DEFAULT_LOG)
        handler = RotatingFileHandler(filename=log_info.get("log_name", LOG_DIR + "main.log"),
                                      maxBytes=log_info.get("max_size", 1 << 20),
                                      backupCount=log_info.get("backup_count", 10))
        logger.setLevel(level=logging.INFO)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_info.get("log_format"))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_logger(self):
        return self.logger
