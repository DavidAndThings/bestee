"""Tests for the shared logging configuration helper."""

import json
import logging
from pathlib import Path

import pytest

from bestee_compute.logging_config import JsonFormatter, configure_logging


def _record(msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="bestee.test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_json_formatter_emits_parseable_line() -> None:
    data = json.loads(JsonFormatter().format(_record("hello %s", "world")))
    assert data["level"] == "INFO"
    assert data["logger"] == "bestee.test"
    assert data["message"] == "hello world"
    assert "timestamp" in data


def test_json_formatter_surfaces_extra_fields() -> None:
    record = _record("done")
    record.result_id = "clustering_abc"  # a structured ``extra=`` field
    data = json.loads(JsonFormatter().format(record))
    assert data["result_id"] == "clustering_abc"


def test_configure_logging_installs_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    try:
        configure_logging("svc")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2
        logging.getLogger("bestee.test").info("hi")
        log_file = tmp_path / "svc.log"
        assert log_file.exists()
        assert "hi" in log_file.read_text()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
