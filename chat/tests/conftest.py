"""Shared fixtures for the wiki subpackage tests.

A deterministic, dependency-free :class:`FakeEncoder` stands in for the real
sentence-transformers model so the semantic path can be tested without
downloading weights or importing torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DUMP = Path(__file__).parent / "fixtures" / "sample-pages-articles.xml"

# A tiny vocabulary; embeddings are normalised term-presence counts over it.
_VOCAB = (
    "king",
    "monarch",
    "queen",
    "england",
    "english",
    "kingdom",
    "car",
    "automobile",
    "vehicle",
    "transport",
)


class FakeEncoder:
    """Deterministic bag-of-words encoder implementing the Encoder protocol."""

    def encode(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vec = np.array(
            [float(lowered.count(term)) for term in _VOCAB], dtype=np.float32
        )
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.encode(t) for t in texts]).astype(np.float32)


@pytest.fixture
def sample_dump() -> Path:
    """Path to the committed sample XML dump fixture."""
    return FIXTURE_DUMP


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    """A deterministic encoder usable in place of the real model."""
    return FakeEncoder()
