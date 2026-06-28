"""Per-task audit trail.

Writes one JSON line per task to ``{LOG_DIR}/audit.jsonl`` (append-only; a single
small line write is atomic across the worker's forked processes).  Wired entirely
through Celery signals, so task bodies stay untouched and *every* task -- including
failures -- is recorded.

Each record carries: the completion timestamp, task id, task name, final state,
the ``result_id`` the task produced (when any), the requesting ``user_email``
forwarded by the API, and the wall-clock duration in seconds.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celery.signals import task_postrun, task_prerun

# Task start times keyed by task id; a task runs start-to-finish in one process,
# so a per-process dict is sufficient to compute durations.
_starts: dict[str, datetime] = {}


def _audit_path() -> Path:
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "audit.jsonl"


def record(entry: dict[str, Any]) -> None:
    """Append *entry* as a single JSON line to the audit log."""
    with _audit_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


@task_prerun.connect
def _on_task_prerun(task_id: str | None = None, **_: Any) -> None:
    if task_id is not None:
        _starts[task_id] = datetime.now(UTC)


@task_postrun.connect
def _on_task_postrun(
    task_id: str | None = None,
    task: Any = None,
    args: Any = None,
    retval: Any = None,
    state: str | None = None,
    **_: Any,
) -> None:
    finished = datetime.now(UTC)
    started = _starts.pop(task_id, None) if task_id else None
    result_id = retval.get("result_id") if isinstance(retval, dict) else None
    # The API dispatches each task with ``args=[payload, user_email]``.
    user_email = args[1] if isinstance(args, list | tuple) and len(args) > 1 else None
    record(
        {
            "timestamp": finished.isoformat(),
            "task_id": task_id,
            "task_name": getattr(task, "name", None),
            "state": state,
            "result_id": result_id,
            "user_email": user_email,
            "duration_seconds": (
                (finished - started).total_seconds() if started else None
            ),
        }
    )
