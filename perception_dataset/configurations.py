import datetime
import os


class Configurations(object):
    log_format = os.getenv("LOG_FORMAT", "text")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file_path = os.getenv("LOG_FILE_PATH", f"/tmp/log/{datetime.date.today()}.log")
    slack_token = os.getenv("SLACK_TOKEN", None)
