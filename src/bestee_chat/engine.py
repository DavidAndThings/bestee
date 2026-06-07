from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from bestee_chat.embeddings import cosine_similarity, default_encoder

if TYPE_CHECKING:
    import numpy as np

    from bestee_chat.embeddings import Encoder


class Exchange:
    def __init__(
        self,
        user_utterance: Sequence[str],
        bot_utterance: Sequence[str],
        encoder: Encoder | None = None,
    ) -> None:
        self.user_utterance = user_utterance
        self.bot_utterance = bot_utterance
        self._encoder = encoder
        self._user_embedding: np.ndarray | None = None

    @property
    def encoder(self) -> Encoder:
        """The encoder used for embeddings (defaults to the shared encoder)."""
        encoder = self._encoder
        if encoder is None:
            encoder = default_encoder()
            self._encoder = encoder
        return encoder

    def similarity(self, query: Sequence[str]) -> float:
        """Return the semantic similarity between ``query`` and ``user_utterance``.

        Both token sequences are embedded with a local sentence-transformers
        model and compared by cosine similarity. The score ranges from roughly
        ``0.0`` (unrelated) to ``1.0`` (equivalent meaning); empty input on
        either side yields ``0.0``. The ``user_utterance`` embedding is computed
        once and cached.
        """
        query_text = " ".join(query).strip()
        user_text = " ".join(self.user_utterance).strip()
        if not query_text or not user_text:
            return 0.0

        user_embedding = self._user_embedding
        if user_embedding is None:
            user_embedding = self.encoder.encode(user_text)
            self._user_embedding = user_embedding

        query_embedding = self.encoder.encode(query_text)
        return cosine_similarity(query_embedding, user_embedding)

    def __hash__(self) -> int:
        return hash(self.bot_utterance) + hash(self.user_utterance)

    def __eq__(self, other):
        return (
            self.user_utterance == other.user_utterance
            and self.bot_utterance == other.bot_utterance
        )

    def __repr__(self):
        return (
            f"Exchange(user_utterance={self.user_utterance}, "
            f"bot_utterance={self.bot_utterance})"
        )

    def __str__(self):
        return f"User: {self.user_utterance}\nBot: {self.bot_utterance}"


class Knowledge(ABC):
    def __init__(self, encoder: Encoder | None = None) -> None:
        self._encoder = encoder

    @abstractmethod
    def generate_exchanges(
        self,
    ) -> set[Exchange]:
        raise NotImplementedError


class Brain:
    def __init__(self) -> None:
        self._knowledge_base: set[Knowledge] = set()

    def add_knowledge(self, knowledge: Knowledge) -> None:
        self._knowledge_base.add(knowledge)

    def pick_highest_ranked_exchange(
        self, query: Sequence[str]
    ) -> tuple[Exchange, float]:

        highest_ranked_exchange = None
        highest_ranked_score = 0.0

        for knowledge in self._knowledge_base:
            for exchange in knowledge.generate_exchanges():
                score = exchange.similarity(query)
                if score > highest_ranked_score:
                    highest_ranked_score = score
                    highest_ranked_exchange = exchange

        if highest_ranked_exchange is None:
            raise ValueError("No exchanges found")

        return highest_ranked_exchange, highest_ranked_score
