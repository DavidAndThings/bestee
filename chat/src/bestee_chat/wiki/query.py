"""Query a built index: BM25 recall, with optional semantic reranking.

The retrieval path is hybrid by design. The always-present lexical layer
recalls candidates by BM25; when the active profile is semantic and a vector
store is present, those candidates are re-scored against the query embedding
and the two signals are blended. With no vectors (or no ``torch``) the same
function transparently returns pure BM25 results.
"""

from __future__ import annotations

import logging

from bestee_chat.embeddings import Encoder, cosine_scores
from bestee_chat.wiki._util import normalise, optional_encoder
from bestee_chat.wiki.cache import CacheLayout, get_layout
from bestee_chat.wiki.hardware import probe_hardware
from bestee_chat.wiki.index import LexicalIndex
from bestee_chat.wiki.models import SearchHit
from bestee_chat.wiki.profiles import Profile, resolve_profile
from bestee_chat.wiki.sources import WikiSource
from bestee_chat.wiki.vectors import VectorStore

logger = logging.getLogger(__name__)

#: Weight given to the semantic signal when blending with normalised BM25.
SEMANTIC_WEIGHT = 0.7


def search_wiki(
    query: str,
    *,
    source: WikiSource | None = None,
    profile: str = "auto",
    limit: int = 10,
    cache_dir: str | None = None,
    encoder: Encoder | None = None,
) -> list[SearchHit]:
    """Return up to ``limit`` ranked :class:`SearchHit` results for ``query``.

    Requires an index to already exist for the resolved profile (build one with
    :func:`bestee_chat.wiki.builder.build_index`). Raises ``FileNotFoundError``
    if none is present.
    """
    source = source or WikiSource()
    layout = get_layout(source, cache_dir)
    resolved = resolve_profile(profile, probe_hardware(cache_dir))

    sqlite_path = layout.sqlite_path(resolved.name)
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"no index for profile {resolved.name!r} at {sqlite_path}; "
            "build one with build_index() first"
        )

    use_semantic = resolved.semantic and layout.vectors_path(resolved.name).exists()
    recall_k = resolved.recall_k if use_semantic else limit

    with LexicalIndex(sqlite_path) as lexical:
        hits = lexical.search(query, limit=recall_k)

    if not use_semantic or not hits:
        return _renumber(hits[:limit])

    reranked = _rerank(query, hits, layout, resolved, encoder)
    return _renumber(reranked[:limit])


def _rerank(
    query: str,
    hits: list[SearchHit],
    layout: CacheLayout,
    profile: Profile,
    encoder: Encoder | None,
) -> list[SearchHit]:
    """Blend BM25 and semantic similarity over the recalled ``hits``."""
    enc = encoder or optional_encoder()
    if enc is None:
        return hits

    store = VectorStore.load(layout.vectors_path(profile.name))
    sub, present = store.matrix_for([h.page_id for h in hits])
    if not present:
        return hits

    query_vec = enc.encode(query)
    sims = cosine_scores(sub, query_vec)
    sim_by_pid = {pid: float(sims[i]) for i, pid in enumerate(present)}

    bm25_norm = normalise([h.score for h in hits])
    blended: list[SearchHit] = []
    for hit, lex_norm in zip(hits, bm25_norm, strict=True):
        sim = sim_by_pid.get(hit.page_id)
        if sim is None:
            score = (1.0 - SEMANTIC_WEIGHT) * lex_norm
        else:
            score = SEMANTIC_WEIGHT * sim + (1.0 - SEMANTIC_WEIGHT) * lex_norm
        blended.append(_with_score(hit, score))

    blended.sort(key=lambda h: h.score, reverse=True)
    return blended


def _with_score(hit: SearchHit, score: float) -> SearchHit:
    return SearchHit(
        rank=hit.rank,
        score=score,
        title=hit.title,
        snippet=hit.snippet,
        url=hit.url,
        page_id=hit.page_id,
    )


def _renumber(hits: list[SearchHit]) -> list[SearchHit]:
    """Reassign 1-based ranks after slicing/reordering."""
    return [_with_rank(hit, i) for i, hit in enumerate(hits, start=1)]


def _with_rank(hit: SearchHit, rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=hit.score,
        title=hit.title,
        snippet=hit.snippet,
        url=hit.url,
        page_id=hit.page_id,
    )
