"""Local cache layout and manifest for downloaded dumps and built indexes.

Everything the ``wiki`` subpackage writes to disk lives under a single cache
root, namespaced by source (``<root>/<dbname>/<date>/``). A small JSON manifest
records what has been downloaded/built so later runs can decide what is fresh.

Resolution order for the cache root:

1. an explicit ``cache_dir`` argument,
2. the ``BESTEE_WIKI_CACHE`` environment variable,
3. ``~/.cache/bestee-chat/wiki`` (XDG-style default).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from bestee_chat.wiki.sources import WikiSource

_CACHE_ENV = "BESTEE_WIKI_CACHE"
_DEFAULT_CACHE = "~/.cache/bestee-chat/wiki"
_MANIFEST_NAME = "manifest.json"


def resolve_cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache root directory, honouring the resolution order above.

    The directory is *not* created here; callers create the specific
    sub-paths they need via :class:`CacheLayout`.
    """
    raw = (
        str(cache_dir)
        if cache_dir is not None
        else os.environ.get(_CACHE_ENV) or _DEFAULT_CACHE
    )
    return Path(raw).expanduser()


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Resolved on-disk paths for a single :class:`WikiSource`."""

    root: Path
    source: WikiSource

    @property
    def source_dir(self) -> Path:
        """Directory holding all artifacts for this source."""
        return self.root / self.source.dbname / self.source.date_segment

    @property
    def dump_path(self) -> Path:
        """Where the downloaded articles dump is stored."""
        return self.source_dir / self.source.data_filename

    @property
    def index_dump_path(self) -> Path:
        """Where the multistream offset index is stored (may be unused)."""
        name = self.source.index_filename or "multistream-index.txt.bz2"
        return self.source_dir / name

    @property
    def manifest_path(self) -> Path:
        """Path of the JSON manifest for this source."""
        return self.source_dir / _MANIFEST_NAME

    def index_dir(self, profile: str) -> Path:
        """Directory for a built index at the given quality ``profile``."""
        return self.source_dir / f"index.{profile}"

    def sqlite_path(self, profile: str) -> Path:
        """Path of the SQLite (FTS5 + metadata) index for a ``profile``."""
        return self.index_dir(profile) / "index.sqlite"

    def vectors_path(self, profile: str) -> Path:
        """Path of the stored embedding matrix for a ``profile``."""
        return self.index_dir(profile) / "vectors.npz"

    def ensure_source_dir(self) -> Path:
        """Create and return :attr:`source_dir`."""
        self.source_dir.mkdir(parents=True, exist_ok=True)
        return self.source_dir

    def ensure_index_dir(self, profile: str) -> Path:
        """Create and return :meth:`index_dir` for a ``profile``."""
        path = self.index_dir(profile)
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_layout(
    source: WikiSource,
    cache_dir: str | os.PathLike[str] | None = None,
) -> CacheLayout:
    """Build a :class:`CacheLayout` for ``source`` under the resolved root."""
    return CacheLayout(root=resolve_cache_root(cache_dir), source=source)


@dataclass(slots=True)
class Manifest:
    """Mutable record of what has been cached for a source.

    Persisted as ``manifest.json`` next to the dump. Unknown keys read from
    disk are preserved in :attr:`extra` so forward-compatible fields written
    by a newer version are not silently dropped.
    """

    dbname: str = ""
    date: str = ""
    dump_sha1: str | None = None
    dump_bytes: int | None = None
    downloaded_at: str | None = None
    indexes: dict[str, dict[str, object]] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load a manifest from ``path``, or return an empty one if missing."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__slots__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            dbname=str(data.get("dbname", "")),
            date=str(data.get("date", "")),
            dump_sha1=data.get("dump_sha1"),
            dump_bytes=data.get("dump_bytes"),
            downloaded_at=data.get("downloaded_at"),
            indexes=dict(data.get("indexes", {})),
            extra=extra,
        )

    def save(self, path: Path) -> None:
        """Write this manifest to ``path`` as pretty JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "dbname": self.dbname,
            "date": self.date,
            "dump_sha1": self.dump_sha1,
            "dump_bytes": self.dump_bytes,
            "downloaded_at": self.downloaded_at,
            "indexes": self.indexes,
            **self.extra,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
