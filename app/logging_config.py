"""Logging setup for the API.

Two goals here:

1. Logs go to BOTH the console (useful while developing) and a file
   (useful at 3am when nobody is watching the console). The file handler
   rotates, so the log can never quietly eat the disk.
2. Every line carries the request_id of the request that produced it, so
   one request's whole journey can be grepped out of the file later.

The request_id is passed around using a ContextVar rather than being
threaded through every function signature. A ContextVar is a variable
whose value is scoped to the current task/thread, so the middleware can
set it once per request and any log line emitted while handling that
request picks it up automatically.
"""
from __future__ import annotations

import logging
import logging.handlers
from contextvars import ContextVar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "api.log"

# Name of the logger every module in this package hangs off of. Modules
# call logging.getLogger(__name__), which produces "app.main", "app.ml_model",
# etc. — all children of "app", so configuring "app" configures them all.
APP_LOGGER_NAME = "app"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | request_id=%(request_id)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

MAX_BYTES = 1_000_000  # rotate at ~1 MB
BACKUP_COUNT = 5       # keep api.log.1 ... api.log.5

# Default "-" means: a log line emitted outside of any request (startup,
# shutdown) still formats cleanly instead of blowing up on a missing field.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Stamp every record with the current request_id.

    Attached to the *handlers* rather than the logger, so that records
    arriving from other libraries (uvicorn, for example) also get the
    attribute and don't crash the formatter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True  # never actually filters anything out


def kv(**fields: object) -> str:
    """Render fields as logfmt-style `key=value` pairs.

    This is what makes the logs *structured* rather than free prose:
    every line is machine-parseable, so `grep event=prediction_failed`
    or a log shipper's field extraction both work.
    """
    parts = []
    for key, value in fields.items():
        text = "-" if value is None else str(value)
        if text == "" or any(c in text for c in ' "'):
            text = '"' + text.replace('"', '\\"') + '"'
        parts.append(f"{key}={text}")
    return " ".join(parts)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the `app` logger. Safe to call more than once."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(level)

    # Re-running (uvicorn --reload) must not stack up duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    request_id_filter = RequestIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(request_id_filter)

    rotating_file = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    rotating_file.setFormatter(formatter)
    rotating_file.addFilter(request_id_filter)

    logger.addHandler(console)
    logger.addHandler(rotating_file)

    # Don't also hand records to the root logger, or uvicorn's own root
    # handler prints every line a second time.
    logger.propagate = False

    return logger
