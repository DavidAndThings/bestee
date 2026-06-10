"""Central configuration: cache locations, offline mode, and tunable constants.

This is the single registry for the ``BESTEE_*`` environment variables so every
knob is documented in one place:

* ``BESTEE_CACHE_DIR``      -- root for all downloaded/derived resources.
* ``BESTEE_OFFLINE``        -- forbid all network access (also sets HF offline).
* ``BESTEE_WIKI_CACHE``     -- override just the wiki cache (see ``wiki.cache``).
* ``BESTEE_<KIND>_DIR``     -- override a learned-knowledge dir, e.g.
  ``BESTEE_PERSONS_DIR`` or ``BESTEE_ARTICLES_DIR``.
* ``BESTEE_EMBEDDING_MODEL`` / ``BESTEE_DEVICE`` -- embedding model + device.
"""

from __future__ import annotations

import os
from pathlib import Path

_CACHE_DIR_ENV = "BESTEE_CACHE_DIR"
_DEFAULT_CACHE_DIR = "~/.cache/bestee-chat"
_OFFLINE_ENV = "BESTEE_OFFLINE"

#: Default maximum size of a single JSON file before records spill into a new one.
MAX_FILE_BYTES = 64 * 1024

#: Similarity below which the Brain rejects a candidate answer.
CANDIDATE_REJECTION_THRESHOLD = 0.5

_TRUTHY = {"1", "true", "yes", "on"}


def cache_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the root directory for all bestee-chat caches.

    Resolution order: explicit ``override`` -> ``BESTEE_CACHE_DIR`` ->
    ``~/.cache/bestee-chat``. The directory is not created here; callers create
    the specific sub-paths they need.
    """
    raw = (
        str(override)
        if override is not None
        else os.environ.get(_CACHE_DIR_ENV) or _DEFAULT_CACHE_DIR
    )
    return Path(raw).expanduser()


def knowledge_dir(kind: str, override: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache directory for learned records of a given ``kind``.

    Resolution order: explicit ``override`` -> ``BESTEE_<KIND>_DIR`` (e.g.
    ``BESTEE_PERSONS_DIR``) -> ``<cache_root>/<kind>``. Learned knowledge is a
    managed cache, so it lives under the shared cache root by default rather
    than inside the project.
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(f"BESTEE_{kind.upper()}_DIR")
    if env:
        return Path(env).expanduser()
    return cache_root() / kind


def persons_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Directory for learned people (alias for ``knowledge_dir('persons')``)."""
    return knowledge_dir("persons", override)


def is_offline() -> bool:
    """Whether ``BESTEE_OFFLINE`` requests fully offline operation."""
    return os.environ.get(_OFFLINE_ENV, "").strip().lower() in _TRUTHY


def apply_offline() -> bool:
    """Propagate ``BESTEE_OFFLINE`` to ``HF_HUB_OFFLINE`` and return offline state.

    Calling this once at CLI/start-up makes a single switch govern network
    access for both the embedding model (Hugging Face) and dump downloads.
    """
    offline = is_offline()
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return offline
