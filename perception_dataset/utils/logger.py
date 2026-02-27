import datetime
import logging
from logging import FileHandler, getLogger
import os
import uuid

from pythonjsonlogger import jsonlogger
from rich.logging import RichHandler

from perception_dataset.configurations import Configurations


class CustomTextFormatter(logging.Formatter):
    def format(self, record):  # noqa: A003
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


# ロギング初期化済みフラグ
_logging_initialized = False


def configure_logger(
    log_file_path=Configurations.log_file_path,
    modname=__name__,
):
    global _logging_initialized

    # 既に初期化済みの場合は、指定されたモジュールのloggerを返すだけ
    if _logging_initialized:
        return logging.getLogger(modname)

    # root loggerを使用して一度だけ初期化
    root = logging.getLogger()
    root.setLevel(Configurations.log_level)

    # --- 既存ハンドラー除去 ---
    root.handlers.clear()

    # --- Filter追加 ---
    root.addFilter(SensitiveWordFilter())

    # --- Formatter ---
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s"
    )

    # --- RichHandler (コンソール) ---
    console_handler = RichHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- FileHandler (ファイル出力) ---
    # ディレクトリがなければ作る
    log_directory = os.path.dirname(log_file_path)
    os.makedirs(log_directory, exist_ok=True)

    file_handler = FileHandler(log_file_path)
    file_handler.setLevel(Configurations.log_level)
    # ファイル出力は常にJSONフォーマット
    file_handler.setFormatter(CustomJsonFormatter())
    root.addHandler(file_handler)

    # 初期化フラグを設定
    _logging_initialized = True

    # 完了ログ（一度だけ）
    root.info(f"Logging initialized. File logs at: {log_file_path}")

    # 指定されたモジュールのloggerを返す
    return logging.getLogger(modname)


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
