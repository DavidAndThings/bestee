"""Redis client for direct job-listing queries.

Celery's ``AsyncResult`` API requires a known task id.  To list *all* jobs we
scan the Redis result-backend directly: Celery writes task metadata as JSON
under the key ``celery-task-meta-{task_id}``.  We use a single ``MGET`` for
each page so listing never issues one round-trip per task.
"""

import json
import os
from typing import Any, cast

import redis
from dotenv import load_dotenv

_KEY_PREFIX = "celery-task-meta-"
_START_KEY_PREFIX = "celery-task-started-"
# Our own per-user job index (submitted jobs are recorded here at dispatch time,
# so queued jobs are listable immediately and the history survives across
# browsers / devices -- unlike the Celery meta, which only appears once a worker
# picks the task up).
_JOB_PREFIX = "bestee:job:"
_USER_JOBS_PREFIX = "bestee:user-jobs:"
_MAX_JOBS_PER_USER = 1000


def _build_redis_client() -> redis.Redis:
    """Return a Redis client backed by ``DO_REDIS_CONNECTION``.

    ``redis.from_url`` is lazy -- no socket is opened until the first command --
    so this is safe to call at module import time even when Redis is down.
    """
    load_dotenv()
    url = os.environ.get("DO_REDIS_CONNECTION", "redis://localhost:6379/0")
    return redis.from_url(url, decode_responses=True)


_redis_client = _build_redis_client()


def list_all_task_ids() -> list[str]:
    """Scan the result backend and return every known task id, sorted.

    Sorting gives callers a stable order so offset/limit pagination is
    consistent across requests.
    """
    keys: list[str] = list(_redis_client.scan_iter(f"{_KEY_PREFIX}*"))
    return sorted(k[len(_KEY_PREFIX) :] for k in keys)


def get_task_start_times(task_ids: list[str]) -> dict[str, str]:
    """Fetch start timestamps recorded by the ``task_prerun`` worker signal.

    Returns a mapping of task id → ISO UTC string for tasks that have started;
    tasks that are still pending (or whose key has expired) are omitted.
    """
    if not task_ids:
        return {}
    keys = [f"{_START_KEY_PREFIX}{tid}" for tid in task_ids]
    values: list[Any] = cast(list[Any], _redis_client.mget(keys))
    return {tid: raw for tid, raw in zip(task_ids, values) if raw is not None}


def record_job(
    user_id: str,
    task_id: str,
    analysis: str,
    payload: dict[str, Any],
    result_id: str,
    created_at: str,
) -> None:
    """Persist a submitted job and prepend it to the user's job list.

    The record holds everything the listing needs that the Celery result
    backend can't reliably give us up front: the analysis, the request payload
    (for redo), the deterministic result id, and the submit time.
    """
    record = json.dumps(
        {
            "task_id": task_id,
            "analysis": analysis,
            "payload": payload,
            "result_id": result_id,
            "created_at": created_at,
        }
    )
    user_key = f"{_USER_JOBS_PREFIX}{user_id}"
    pipe = _redis_client.pipeline()
    pipe.set(f"{_JOB_PREFIX}{task_id}", record)
    pipe.lpush(user_key, task_id)
    pipe.ltrim(user_key, 0, _MAX_JOBS_PER_USER - 1)
    pipe.execute()


def get_job_record(task_id: str) -> dict[str, Any] | None:
    """The stored record for *task_id*, or ``None`` if it was never recorded."""
    raw = cast("str | None", _redis_client.get(f"{_JOB_PREFIX}{task_id}"))
    return json.loads(raw) if raw is not None else None


def list_user_jobs(
    user_id: str, offset: int, limit: int
) -> tuple[int, list[dict[str, Any]]]:
    """A ``(total, records)`` page of the user's jobs, newest first.

    ``records`` are the dicts written by :func:`record_job`; a missing record
    (e.g. expired) is skipped, so the page may be shorter than *limit*.
    """
    user_key = f"{_USER_JOBS_PREFIX}{user_id}"
    total = int(cast(int, _redis_client.llen(user_key)))
    task_ids: list[str] = cast(
        "list[str]", _redis_client.lrange(user_key, offset, offset + limit - 1)
    )
    if not task_ids:
        return total, []
    keys = [f"{_JOB_PREFIX}{tid}" for tid in task_ids]
    raws: list[Any] = cast(list[Any], _redis_client.mget(keys))
    records = [json.loads(raw) for raw in raws if raw is not None]
    return total, records


def get_task_metas(task_ids: list[str]) -> list[dict[str, Any]]:
    """Bulk-fetch the raw Celery metadata for *task_ids* via a single MGET.

    Keys absent from Redis (expired or never written) are returned as a
    synthetic ``PENDING`` entry.
    """
    if not task_ids:
        return []
    keys = [f"{_KEY_PREFIX}{tid}" for tid in task_ids]
    values: list[Any] = cast(list[Any], _redis_client.mget(keys))
    metas: list[dict[str, Any]] = []
    for tid, raw in zip(task_ids, values):
        if raw is None:
            metas.append({"task_id": tid, "status": "PENDING"})
        else:
            meta: dict[str, Any] = json.loads(raw)
            meta.setdefault("task_id", tid)
            metas.append(meta)
    return metas
