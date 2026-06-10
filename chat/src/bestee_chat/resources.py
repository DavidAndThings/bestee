"""Unified management of the external resources bestee-chat depends on.

Two very different things -- the sentence-embedding **model** (downloaded from
the Hugging Face Hub) and a Wikipedia **dump** (downloaded from a Wikimedia
mirror) -- are both "fetch once, verify, cache locally, use offline". This
module gives them one shape (:class:`Resource`) and one front door
(:class:`ResourceManager`) so they can be inspected, downloaded, and cleaned
together.

Network access for *both* is governed by the single ``BESTEE_OFFLINE`` switch
(see :func:`bestee_chat.config.apply_offline`).
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bestee_chat import config
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.download import download_dump
from bestee_chat.wiki.sources import WikiSource

if TYPE_CHECKING:
    from bestee_chat.wiki._util import ProgressCallback
    from bestee_chat.wiki.builder import BuildResult


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    """A point-in-time snapshot of one resource's local state."""

    name: str
    kind: str
    present: bool
    location: str
    size_bytes: int | None
    detail: str


class Resource(ABC):
    """Something fetched from outside and cached locally.

    Implementations decide *how* to fetch/verify/locate their bytes; the
    :class:`ResourceManager` and CLI treat them uniformly.
    """

    name: str
    kind: str

    @abstractmethod
    def status(self) -> ResourceStatus:
        """Return the resource's current local state (no network)."""

    @abstractmethod
    def ensure(
        self, *, force: bool = False, progress: ProgressCallback | None = None
    ) -> None:
        """Download the resource if missing (or ``force``); verify on the way."""

    @abstractmethod
    def clean(self) -> None:
        """Delete the locally cached copy of the resource."""


@dataclass(frozen=True, slots=True)
class _HFRepo:
    """The bits of a Hugging Face cache entry we care about."""

    repo_path: Path
    size_on_disk: int


def _hf_cached_repo(repo_id: str) -> _HFRepo | None:
    """Return the Hugging Face cache entry for ``repo_id``, or None."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return None
    try:
        info = scan_cache_dir()
    except Exception:
        return None
    for repo in info.repos:
        if repo.repo_id == repo_id:
            return _HFRepo(Path(repo.repo_path), int(repo.size_on_disk))
    return None


class ModelResource(Resource):
    """The sentence-embedding model, cached in the Hugging Face hub cache."""

    kind = "model"

    def __init__(self, model_name: str | None = None) -> None:
        if model_name is None:
            from bestee_chat.embeddings import SemanticEncoder

            model_name = SemanticEncoder().model_name
        self.model_name = model_name
        self.name = f"model:{model_name}"

    def status(self) -> ResourceStatus:
        repo = _hf_cached_repo(self.model_name)
        if repo is None:
            return ResourceStatus(
                self.name, self.kind, False, "(huggingface cache)", None, "not cached"
            )
        return ResourceStatus(
            name=self.name,
            kind=self.kind,
            present=True,
            location=str(repo.repo_path),
            size_bytes=repo.size_on_disk,
            detail="cached",
        )

    def ensure(
        self, *, force: bool = False, progress: ProgressCallback | None = None
    ) -> None:
        # progress is unused: Hugging Face manages its own download reporting.
        if not force and self.status().present:
            return
        if config.is_offline():
            raise RuntimeError(f"{self.name} is not cached and BESTEE_OFFLINE is set")
        from bestee_chat.embeddings import SemanticEncoder

        SemanticEncoder(self.model_name).warm_up()

    def clean(self) -> None:
        repo = _hf_cached_repo(self.model_name)
        if repo is not None:
            shutil.rmtree(repo.repo_path, ignore_errors=True)


class DumpResource(Resource):
    """A Wikipedia dump (and any indexes built from it) under the wiki cache."""

    kind = "dump"

    def __init__(
        self, source: WikiSource | None = None, cache_dir: str | None = None
    ) -> None:
        self.source = source or WikiSource()
        self.cache_dir = cache_dir
        self.name = f"dump:{self.source.slug}"

    def status(self) -> ResourceStatus:
        layout = get_layout(self.source, self.cache_dir)
        path = layout.dump_path
        if not path.exists():
            return ResourceStatus(
                self.name, self.kind, False, str(path), None, "not downloaded"
            )
        manifest = Manifest.load(layout.manifest_path)
        indexes = ", ".join(sorted(manifest.indexes)) or "no indexes"
        return ResourceStatus(
            name=self.name,
            kind=self.kind,
            present=True,
            location=str(path),
            size_bytes=path.stat().st_size,
            detail=f"indexes: {indexes}",
        )

    def ensure(
        self, *, force: bool = False, progress: ProgressCallback | None = None
    ) -> None:
        if not force and self.status().present:
            return
        if config.is_offline():
            raise RuntimeError(
                f"{self.name} is not downloaded and BESTEE_OFFLINE is set"
            )
        download_dump(
            self.source, cache_dir=self.cache_dir, refresh=force, progress=progress
        )

    def build_index(
        self,
        profile: str = "auto",
        *,
        lexical_progress: ProgressCallback | None = None,
        vector_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Download (if needed) and build the search index for this dump."""
        self.ensure()
        from bestee_chat.wiki.builder import build_index as _build

        return _build(
            self.source,
            profile,
            cache_dir=self.cache_dir,
            download_if_missing=False,
            lexical_progress=lexical_progress,
            vector_progress=vector_progress,
        )

    def clean(self) -> None:
        layout = get_layout(self.source, self.cache_dir)
        shutil.rmtree(layout.source_dir, ignore_errors=True)

    def clean_indexes(self) -> None:
        """Remove the built indexes for this dump, keeping the downloaded dump."""
        layout = get_layout(self.source, self.cache_dir)
        for path in layout.source_dir.glob("index.*"):
            shutil.rmtree(path, ignore_errors=True)


def _count_records(files: Iterable[Path]) -> int:
    """Count records across JSON-array files, ignoring unreadable ones."""
    total = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        if isinstance(data, list):
            total += len(data)
    return total


class KnowledgeCacheResource(Resource):
    """Learned records of one ``kind`` (people, articles, ...) under the cache.

    Unlike the model and dumps, this resource is *populated* by
    ``bestee-chat learn`` rather than downloaded, so :meth:`ensure` is a no-op;
    it is managed here for unified status and cleanup.
    """

    def __init__(self, kind: str, directory: str | Path | None = None) -> None:
        self.kind = kind
        self.name = kind
        self._directory = directory

    def _dir(self) -> Path:
        return config.knowledge_dir(self.kind, self._directory)

    def status(self) -> ResourceStatus:
        directory = self._dir()
        files = sorted(directory.glob("*.json")) if directory.exists() else []
        if not files:
            return ResourceStatus(
                self.name, self.kind, False, str(directory), None, f"no {self.kind}"
            )
        total = sum(path.stat().st_size for path in files)
        records = _count_records(files)
        return ResourceStatus(
            name=self.name,
            kind=self.kind,
            present=True,
            location=str(directory),
            size_bytes=total,
            detail=f"{records} records in {len(files)} file(s)",
        )

    def ensure(
        self, *, force: bool = False, progress: ProgressCallback | None = None
    ) -> None:
        # Nothing to fetch: records are added by ``bestee-chat learn``.
        return

    def clean(self) -> None:
        shutil.rmtree(self._dir(), ignore_errors=True)


class ResourceManager:
    """A small registry that operates over a set of :class:`Resource` objects."""

    def __init__(self, resources: Iterable[Resource]) -> None:
        self._resources = list(resources)

    @property
    def resources(self) -> list[Resource]:
        """The managed resources, in registration order."""
        return list(self._resources)

    def status(self) -> list[ResourceStatus]:
        """Return the status of every managed resource."""
        return [resource.status() for resource in self._resources]

    def ensure(self, *, force: bool = False) -> None:
        """Ensure every managed resource is present locally."""
        for resource in self._resources:
            resource.ensure(force=force)

    def clean(self) -> None:
        """Delete every managed resource's local copy."""
        for resource in self._resources:
            resource.clean()


def default_manager(
    *,
    model: str | None = None,
    source: WikiSource | None = None,
    cache_dir: str | None = None,
) -> ResourceManager:
    """Build a manager over the model, a Wikipedia dump, and learned knowledge."""
    return ResourceManager(
        [
            ModelResource(model),
            DumpResource(source, cache_dir),
            KnowledgeCacheResource("persons"),
            KnowledgeCacheResource("articles"),
        ]
    )
