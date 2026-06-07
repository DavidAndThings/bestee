"""Expose wiki dump search as a :class:`bestee_chat.engine.Knowledge` source.

This adapts the standalone search API to the chatbot's knowledge interface, so
a downloaded Wikipedia dump can be added to a :class:`bestee_chat.engine.Brain`
alongside other knowledge sources and queried uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence

from bestee_chat.embeddings import Encoder
from bestee_chat.engine import Candidate, Knowledge
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource


def _normalise(values: list[float]) -> list[float]:
    """Min-max scale ``values`` into ``[0, 1]`` (all 0.5 when degenerate)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class WikiKnowledge(Knowledge):
    """Knowledge backed by a pre-built Wikipedia dump index.

    The raw retrieval score (BM25, or a BM25/semantic blend) is min-max scaled
    into ``[0, 1]`` per query so the :class:`~bestee_chat.engine.Brain` has a
    bounded relevance value to rank and threshold against. The candidate
    ``answer`` is the page's snippet and ``source`` is its title.
    """

    def __init__(
        self,
        source: WikiSource | None = None,
        *,
        profile: str = "auto",
        cache_dir: str | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        super().__init__(encoder)
        self._source = source or WikiSource()
        self._profile = profile
        self._cache_dir = cache_dir

    def search(self, query: Sequence[str], k: int = 5) -> list[Candidate]:
        """Return up to ``k`` candidates for ``query`` from the dump index."""
        text = " ".join(query).strip()
        if not text:
            return []

        hits = search_wiki(
            text,
            source=self._source,
            profile=self._profile,
            limit=k,
            cache_dir=self._cache_dir,
            encoder=self._encoder,
        )
        scores = _normalise([hit.score for hit in hits])
        return [
            Candidate(answer=hit.snippet, score=score, source=hit.title)
            for hit, score in zip(hits, scores, strict=True)
        ]
