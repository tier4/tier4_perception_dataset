import datetime
import logging
from logging import FileHandler, StreamHandler, getLogger
import os
import uuid

from pythonjsonlogger import jsonlogger

from perception_dataset.configurations import Configurations


class CustomTextFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            message = f"\033[91m{record.getMessage()}\033[0m"  # Red color for error messages
        elif record.levelno == logging.WARNING:
            message = f"\033[93m{record.getMessage()}\033[0m"  # Orange color for warning messages
        else:
            message = record.getMessage()
        record.msg = message
        return super().format(record)

    def __init__(self):
        super().__init__(
            "[%(asctime)s] [%(levelname)s] [file] %(filename)s [func] %(funcName)s [line] %(lineno)d [%(message)s]"
        )


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def parse(self):
        return [
            "timestamp",
            "level",
            "process",
            "processName",
            "thread",
            "threadName",
            "pathname",
            "funcName",
            "lineno",
            "message",
        ]

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_record["timestamp"] = now
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname


class SensitiveWordFilter(logging.Filter):
    def filter(self, record):  # noqa A003
        sensitive_words = [
            "password",
            "auth_token",
            "token",
            "ingest.sentry.io",
            "secret",
        ]
        log_message = record.getMessage()
        for word in sensitive_words:
            if word in log_message:
                return False
        return True


def configure_logger(
    log_file_path=Configurations.log_file_path,
    modname=__name__,
):
    log_directory = os.path.dirname(log_file_path)
    os.makedirs(log_directory, exist_ok=True)

    logger = getLogger(modname)
    logger.addFilter(SensitiveWordFilter())
    logger.setLevel(Configurations.log_level)

    sh = StreamHandler()
    sh.setLevel(Configurations.log_level)

    if Configurations.log_format == "json":
        formatter = CustomJsonFormatter()
    elif Configurations.log_format == "text":
        formatter = CustomTextFormatter()
    else:
        formatter = CustomJsonFormatter()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_file_path)
    fh.setLevel(Configurations.log_level)

    if Configurations.log_format == "json":
        fh_formatter = CustomJsonFormatter()
    elif Configurations.log_format == "text":
        fh_formatter = CustomTextFormatter()
    else:
        fh_formatter = CustomJsonFormatter()
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger


def log_decorator(logger=configure_logger()):
    def _log_decorator(func):
        def wrapper(*args, **kwargs):
            job_id = str(uuid.uuid4())[:8]
            logger.debug(f"START {job_id} func:{func.__name__} args:{args} kwargs:{kwargs}")
            res = func(*args, **kwargs)
            logger.debug(f"RETURN FROM {job_id} return:{res}")
            return res

        return wrapper

    return _log_decorator


class Configurations(object):
    log_format = os.getenv("LOG_FORMAT", "text")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file_path = os.getenv("LOG_FILE_PATH", f"/tmp/log/{datetime.date.today()}.log")
    slack_token = os.getenv("SLACK_TOKEN", None)
