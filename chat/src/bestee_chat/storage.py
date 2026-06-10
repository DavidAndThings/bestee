"""Append-only JSON storage that spreads records across size-bounded files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from bestee_chat.config import MAX_FILE_BYTES


def store_mapping(
    mapping: Mapping[str, object],
    cache_dir: Path,
    max_bytes: int = MAX_FILE_BYTES,
) -> Path:
    """Store ``mapping`` as JSON in ``cache_dir`` and return the file used.

    Records are kept in JSON-array files. Storage works as follows:

    1. The mapping is appended to the *smallest* existing JSON file in
       ``cache_dir`` (a new file is created if the directory is empty).
    2. If that write makes the file exceed ``max_bytes``, the mapping spills
       over into a fresh JSON file instead, so no file grows too large.

    A single record larger than ``max_bytes`` cannot be split and is kept as
    is. The directory is created if it does not exist.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    target = _smallest_json_file(cache_dir) or _new_json_file(cache_dir)
    records = _load_records(target)
    records.append(dict(mapping))
    _write_records(target, records)

    # If appending overflowed the file (and it holds more than just this new
    # record), move the new record into a fresh file to keep this one bounded.
    if target.stat().st_size > max_bytes and len(records) > 1:
        records.pop()
        _write_records(target, records)
        target = _new_json_file(cache_dir)
        _write_records(target, [dict(mapping)])

    return target


def _smallest_json_file(cache_dir: Path) -> Path | None:
    """Return the smallest ``*.json`` file in ``cache_dir``, or None if empty."""
    files = list(cache_dir.glob("*.json"))
    return min(files, key=lambda p: p.stat().st_size) if files else None


def _new_json_file(cache_dir: Path) -> Path:
    """Return a path for a new, uniquely-numbered JSON file in ``cache_dir``."""
    indices = [int(p.stem) for p in cache_dir.glob("*.json") if p.stem.isdigit()]
    next_index = (max(indices) + 1) if indices else 1
    return cache_dir / f"{next_index:04d}.json"


def _load_records(path: Path) -> list[object]:
    """Load the JSON array from ``path`` (empty list if missing or invalid)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _write_records(path: Path, records: list[object]) -> None:
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
