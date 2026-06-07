"""Local, hardware-adaptive sentence embeddings for semantic similarity.

The encoder wraps a ``sentence-transformers`` model that runs entirely
on-device. Model weights are downloaded once from the Hugging Face Hub and
cached locally; afterwards no network access is required. Set the environment
variable ``HF_HUB_OFFLINE=1`` to forbid network access entirely.

Hardware adaptivity
-------------------
The model is placed on the best available device, preferring a CUDA GPU, then
an Apple Silicon (MPS) GPU, and finally the CPU. Override the defaults with:

* ``BESTEE_DEVICE`` (or the ``device`` argument) -- for example ``cpu``,
  ``cuda``, ``cuda:1``, or ``mps``.
* ``BESTEE_EMBEDDING_MODEL`` (or the ``model_name`` argument) -- swap in a
  larger, higher-quality model (for example ``all-mpnet-base-v2``) when
  running on more capable hardware.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: Small, fast, general-purpose default model (~90 MB).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_DEVICE_ENV = "BESTEE_DEVICE"
_MODEL_ENV = "BESTEE_EMBEDDING_MODEL"


class Encoder(Protocol):
    """Minimal structural interface for objects that embed text into vectors."""

    def encode(self, text: str) -> np.ndarray: ...


def detect_device() -> str:
    """Return the best available torch device: ``cuda``, ``mps``, or ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors, or ``0.0`` if either is zero."""
    vec_a = np.asarray(a, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / norm)


class SemanticEncoder:
    """Lazily-loaded, hardware-adaptive text embedder.

    Importing this module and constructing an encoder are cheap and free of
    side effects: ``torch`` and the model weights are only imported and loaded
    on first use.
    """

    def __init__(
        self, model_name: str | None = None, device: str | None = None
    ) -> None:
        self.model_name = model_name or os.environ.get(_MODEL_ENV, DEFAULT_MODEL)
        self._requested_device = device or os.environ.get(_DEVICE_ENV) or None
        self._model: SentenceTransformer | None = None

    @property
    def device(self) -> str:
        """The resolved torch device (computed without loading the model)."""
        if self._model is not None:
            return str(self._model.device)
        return self._requested_device or detect_device()

    def warm_up(self) -> None:
        """Download (if needed) and load the model so later calls are instant."""
        self._load()

    def encode(self, text: str) -> np.ndarray:
        """Embed a single string into a unit-normalized vector."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings into unit-normalized vectors."""
        model = self._load()
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self._requested_device or detect_device()
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model


_default_encoder: SemanticEncoder | None = None


def default_encoder() -> SemanticEncoder:
    """Return a lazily-created, process-wide encoder shared by all callers."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = SemanticEncoder()
    return _default_encoder


def download() -> None:
    """Download and cache the default model for offline use (CLI entry point)."""
    encoder = default_encoder()
    print(f"Loading model {encoder.model_name!r} on device {encoder.device!r}...")
    encoder.warm_up()
    print("Model cached locally and ready for offline use.")
