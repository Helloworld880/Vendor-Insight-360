from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import Settings
from utils.request_context import request_id_var, user_id_var


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        record.user_id = user_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        user_id = getattr(record, "user_id", None)
        if user_id is not None:
            payload["user_id"] = user_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self.RESERVED or key in payload:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def _build_handler(handler: logging.Handler) -> logging.Handler:
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())
    return handler


def setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    console_handler = _build_handler(logging.StreamHandler())
    handlers: list[logging.Handler] = [console_handler]

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    if settings.is_production:
        file_handler = RotatingFileHandler(
            filename=Path(settings.log_file_path),
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        handlers.append(_build_handler(file_handler))

    for handler in handlers:
        root.addHandler(handler)
