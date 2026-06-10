"""Quality profiles and hardware-driven ``auto`` selection.

Relevance is a continuum across several independent axes (granularity, model
size, whether to run a semantic rerank, how deep to recall). A *profile* is a
named point on that frontier. ``resolve_profile("auto", hw)`` inspects a
:class:`~bestee_chat.wiki.hardware.HardwareProfile` and chooses the richest
preset the machine can comfortably build and serve.

Selection is pure arithmetic over the hardware record, so it is fully
unit-testable with synthetic profiles and needs no real hardware or ML deps.
"""

from __future__ import annotations

from dataclasses import dataclass

from bestee_chat.embeddings import DEFAULT_MODEL
from bestee_chat.wiki.hardware import HardwareProfile

#: Granularity of the unit that gets embedded/indexed.
GRANULARITY_LEAD = "lead"
GRANULARITY_ARTICLE = "article"


@dataclass(frozen=True, slots=True)
class Profile:
    """A concrete set of build/query settings.

    Attributes
    ----------
    name:
        The profile's identifier (e.g. ``"economy"``).
    semantic:
        Whether the vector/rerank layer is used at all. When ``False`` the
        profile is lexical-only (BM25) and needs no ML dependencies.
    model:
        The sentence-transformers model to embed with (ignored when
        ``semantic`` is ``False``).
    granularity:
        Which slice of each page is embedded -- the lead paragraph or the
        full article body.
    recall_k:
        How many BM25 candidates to retrieve before an optional rerank.
    rerank:
        Whether to semantically re-rank the recalled candidates.
    """

    name: str
    semantic: bool
    model: str
    granularity: str
    recall_k: int
    rerank: bool


# Ordered from cheapest to richest. ``auto`` walks this from the top down and
# picks the first preset the hardware can support.
_LEXICAL = Profile(
    name="lexical",
    semantic=False,
    model="",
    granularity=GRANULARITY_LEAD,
    recall_k=50,
    rerank=False,
)
_ECONOMY = Profile(
    name="economy",
    semantic=True,
    model=DEFAULT_MODEL,
    granularity=GRANULARITY_LEAD,
    recall_k=80,
    rerank=True,
)
_STANDARD = Profile(
    name="standard",
    semantic=True,
    model="sentence-transformers/all-mpnet-base-v2",
    granularity=GRANULARITY_ARTICLE,
    recall_k=150,
    rerank=True,
)
_QUALITY = Profile(
    name="quality",
    semantic=True,
    model="BAAI/bge-large-en-v1.5",
    granularity=GRANULARITY_ARTICLE,
    recall_k=250,
    rerank=True,
)
_MAX = Profile(
    name="max",
    semantic=True,
    model="BAAI/bge-large-en-v1.5",
    granularity=GRANULARITY_ARTICLE,
    recall_k=400,
    rerank=True,
)

PROFILES: dict[str, Profile] = {
    p.name: p for p in (_LEXICAL, _ECONOMY, _STANDARD, _QUALITY, _MAX)
}

#: Minimum total RAM (GiB) a preset wants before ``auto`` will choose it.
_RAM_FLOOR = {"economy": 0.0, "standard": 24.0, "quality": 56.0, "max": 96.0}


def list_profiles() -> list[str]:
    """Return the known profile names, cheapest to richest."""
    return list(PROFILES)


def get_profile(name: str) -> Profile:
    """Look up a profile by name, raising ``KeyError`` if unknown."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(PROFILES)
        raise KeyError(f"unknown profile {name!r}; choose one of: {known}") from exc


def auto_profile(hw: HardwareProfile) -> Profile:
    """Pick the richest profile ``hw`` can support, top-down with floors.

    The two gates are kept separate, matching the design:

    * a **capability** gate -- no ``torch`` means lexical-only;
    * a **budget** gate -- RAM floors (and requiring a real accelerator for the
      heaviest tiers, since a large-model build is impractically slow on CPU).
    """
    if not hw.torch_available:
        return _LEXICAL

    has_accel = hw.accel in ("cuda", "mps")
    if hw.ram_gb >= _RAM_FLOOR["max"] and hw.accel == "cuda":
        return _MAX
    if hw.ram_gb >= _RAM_FLOOR["quality"] and has_accel:
        return _QUALITY
    if hw.ram_gb >= _RAM_FLOOR["standard"] and has_accel:
        return _STANDARD
    return _ECONOMY


def resolve_profile(name: str, hw: HardwareProfile) -> Profile:
    """Resolve a requested profile ``name`` against the hardware ``hw``.

    ``"auto"`` delegates to :func:`auto_profile`. Any other name is looked up
    directly, but a semantic profile is **downgraded to lexical** when ``torch``
    is unavailable, so an explicit request can never crash on a bare install.
    """
    if name == "auto":
        return auto_profile(hw)
    profile = get_profile(name)
    if profile.semantic and not hw.torch_available:
        return _LEXICAL
    return profile
