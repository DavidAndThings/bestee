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
