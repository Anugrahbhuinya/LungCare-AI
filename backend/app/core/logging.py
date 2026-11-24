"""Centralized logging configuration for LungCare AI Backend.

Provides standard structured formatting, severity control, prevents duplicate
handlers, and enforces privacy practices (avoiding logging sensitive payload data).
Attaches request correlation identifiers from request_context to every log record.
"""

import logging
import sys
from typing import Optional

from app.core.request_context import get_request_id

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(request_id)s] | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Install global LogRecord factory hook to guarantee request_id attribute on all LogRecords
_original_log_record_factory = logging.getLogRecordFactory()


def _request_context_log_record_factory(*args, **kwargs) -> logging.LogRecord:
    record = _original_log_record_factory(*args, **kwargs)
    record.request_id = get_request_id()
    return record


logging.setLogRecordFactory(_request_context_log_record_factory)


class RequestContextFilter(logging.Filter):
    """Logging filter that injects the current request correlation ID into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id") or record.request_id is None:
            record.request_id = get_request_id()
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger and framework loggers.
    
    Ensures single-handler attachment to prevent duplicate log records.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplication on reloads/reconfigurations
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    # Attach filter to handler and root logger
    ctx_filter = RequestContextFilter()
    console_handler.addFilter(ctx_filter)
    root_logger.addFilter(ctx_filter)
    root_logger.addHandler(console_handler)

    # Adjust external framework logger levels to prevent noisy logs
    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        framework_logger = logging.getLogger(logger_name)
        framework_logger.handlers.clear()
        framework_logger.addHandler(console_handler)
        framework_logger.addFilter(ctx_filter)
        framework_logger.propagate = False
        framework_logger.setLevel(numeric_level)

    # Silence default uvicorn.access to prevent duplicate access logs and raw query-string leakage
    # (HTTP access events are centrally and securely recorded by lungcare.access middleware).
    # Avoid clearing uvicorn.access handlers directly; set level to WARNING to suppress normal access records.
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.WARNING)
    uvicorn_access.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Obtain a logger instance with the standard configuration."""
    return logging.getLogger(name or "lungcare")
