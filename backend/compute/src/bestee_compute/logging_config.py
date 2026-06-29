"""Application logging configuration shared by the bestee services.

This is a *helper* the application entrypoints (the Celery worker, the FastAPI
server) call explicitly -- importing this module does not touch logging, so it
respects the convention that libraries don't configure logging on import.

:func:`configure_logging` installs two handlers on the root logger:

* a ``StreamHandler`` to stdout (captured by ``systemd``/``journald`` on a
  droplet), and
* a ``WatchedFileHandler`` writing ``{LOG_DIR}/{service}.log`` -- in-process
  rotation is intentionally *not* used; rotation is delegated to the system
  ``logrotate``, and the handler reopens the file when logrotate moves it.
  Appending plus reopen-on-rotate keeps file logging safe across the worker's
  forked processes.

Environment:
    ``LOG_DIR``     directory for log files (default ``logs``)
    ``LOG_LEVEL``   root level, e.g. ``DEBUG`` / ``INFO`` (default ``INFO``)
    ``LOG_FORMAT``  ``json`` (default, one object per line) or ``text``
"""

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import WatchedFileHandler
from pathlib import Path

# Attributes present on a default LogRecord -- everything else on a record is a
# caller-supplied ``extra=`` field worth surfacing in the JSON output.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_formatter() -> logging.Formatter:
    if os.environ.get("LOG_FORMAT", "json").strip().lower() == "text":
        return logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    return JsonFormatter()


def configure_logging(service: str) -> None:
    """Install stdout + file handlers on the root logger for *service*.

    Idempotent: existing root handlers are removed first, so it is safe to call
    from both Celery's ``setup_logging`` signal and a FastAPI lifespan.
    """
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = _build_formatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = WatchedFileHandler(log_dir / f"{service}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
