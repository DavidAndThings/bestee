"""Optional semantic layer: precomputed page embeddings for reranking.

This reuses the project's existing on-device embedder
(:class:`bestee_chat.embeddings.SemanticEncoder`) rather than introducing a
new model stack. Each indexed page is embedded once and the matrix is stored
as a compressed ``.npz`` next to the lexical index. At query time the stored
vectors for the BM25-recalled candidates are scored against the query with
:func:`bestee_chat.embeddings.cosine_scores`.

Brute-force cosine over a candidate set is more than fast enough at the scale
this layer targets (Simple English, or article-level English Wikipedia). A
dedicated ANN backend (FAISS / hnswlib) is a documented future upgrade behind
this same interface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bestee_chat.embeddings import Encoder
from bestee_chat.wiki._util import ProgressCallback
from bestee_chat.wiki.profiles import GRANULARITY_LEAD

_MAX_BODY_CHARS = 2000


def _encode_texts(encoder: Encoder, texts: Sequence[str]) -> np.ndarray:
    """Embed ``texts`` using ``encoder``'s batch path when available."""
    batch = getattr(encoder, "encode_batch", None)
    if callable(batch):
        return np.asarray(batch(list(texts)), dtype=np.float32)
    return np.vstack([np.asarray(encoder.encode(t), dtype=np.float32) for t in texts])


def _unit_text(title: str, summary: str, body: str, granularity: str) -> str:
    """Compose the text to embed for one page, per ``granularity``."""
    if granularity == GRANULARITY_LEAD:
        return f"{title}. {summary}".strip()
    return f"{title}. {body[:_MAX_BODY_CHARS]}".strip()


def build_vectors(
    sqlite_path: str | Path,
    out_path: str | Path,
    encoder: Encoder,
    *,
    granularity: str,
    batch_size: int = 256,
    progress: ProgressCallback | None = None,
) -> int:
    """Embed every page in a lexical index and save a vector store.

    Returns the number of vectors written. Pages are read in ``rowid`` order
    and embedded in batches to bound memory.
    """
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT p.page_id AS page_id, f.title AS title, "
            "f.summary AS summary, f.body AS body "
            "FROM pages p JOIN pages_fts f ON f.rowid = p.rowid "
            "ORDER BY p.rowid"
        ).fetchall()
    finally:
        conn.close()

    page_ids = np.array([int(r["page_id"]) for r in rows], dtype=np.int64)
    chunks: list[np.ndarray] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [
            _unit_text(r["title"], r["summary"] or "", r["body"] or "", granularity)
            for r in batch_rows
        ]
        chunks.append(_encode_texts(encoder, texts))
        if progress is not None:
            progress(min(start + batch_size, total), total)

    if chunks:
        matrix = np.vstack(chunks).astype(np.float32)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, vectors=matrix, page_ids=page_ids)
    return total


class VectorStore:
    """Loaded page embeddings, addressable by ``page_id``."""

    def __init__(self, vectors: np.ndarray, page_ids: np.ndarray) -> None:
        self.vectors = vectors
        self.page_ids = page_ids
        self._index = {int(pid): i for i, pid in enumerate(page_ids)}

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Load a vector store previously written by :func:`build_vectors`."""
        with np.load(path) as data:
            return cls(data["vectors"], data["page_ids"])

    def matrix_for(self, page_ids: Sequence[int]) -> tuple[np.ndarray, list[int]]:
        """Return ``(submatrix, present_ids)`` for the given ``page_ids``.

        ``submatrix`` rows align with ``present_ids`` (input ids missing from
        the store are dropped), so a caller can map scores back by position.
        """
        idx: list[int] = []
        present: list[int] = []
        for pid in page_ids:
            i = self._index.get(int(pid))
            if i is not None:
                idx.append(i)
                present.append(int(pid))
        if not idx:
            return np.zeros((0, 0), dtype=np.float32), []
        return self.vectors[idx], present
