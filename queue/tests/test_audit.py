"""Offline tests for the per-task audit trail (no broker, no network)."""

import json
from pathlib import Path

import pytest

import audit


def test_record_appends_json_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    audit.record({"task_id": "t1", "state": "SUCCESS"})
    audit.record({"task_id": "t2", "state": "FAILURE"})
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["task_id"] == "t1"
    assert json.loads(lines[1])["state"] == "FAILURE"


def test_postrun_records_ids_email_and_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    class _Task:
        name = "tuning.optimize_clustering"

    audit._on_task_prerun(task_id="task-1")
    audit._on_task_postrun(
        task_id="task-1",
        task=_Task(),
        args=[{"tickers": ["AAA"]}, "user@example.com"],
        retval={"result_id": "clustering_abc"},
        state="SUCCESS",
    )
    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert entry["task_id"] == "task-1"
    assert entry["task_name"] == "tuning.optimize_clustering"
    assert entry["state"] == "SUCCESS"
    assert entry["result_id"] == "clustering_abc"
    assert entry["user_email"] == "user@example.com"
    assert entry["duration_seconds"] is not None


def test_postrun_handles_failure_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    audit._on_task_postrun(
        task_id="task-2",
        task=None,
        args=[{}],
        retval=ValueError("boom"),
        state="FAILURE",
    )
    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert entry["state"] == "FAILURE"
    assert entry["result_id"] is None
    assert entry["user_email"] is None
    assert entry["duration_seconds"] is None
