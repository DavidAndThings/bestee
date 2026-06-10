from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bestee_chat.config import CANDIDATE_REJECTION_THRESHOLD

if TYPE_CHECKING:
    from bestee_chat.embeddings import Encoder


@dataclass(frozen=True)
class Candidate:
    """A possible answer to a query, produced by a piece of :class:`Knowledge`.

    ``score`` is a relevance value (ideally a cosine in ``[0, 1]``) that the
    :class:`Brain` uses to rank and accept/reject candidates across sources;
    ``source`` records where the answer came from.
    """

    answer: str
    score: float
    source: str


class Knowledge(ABC):
    """A source of answers that can be searched with a query.

    Subclasses retrieve however suits their data -- precomputed embeddings,
    SQL, an API, on-the-fly computation -- and need not enumerate everything up
    front. This is what lets a knowledge source be genuinely anything.
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        self._encoder = encoder

    @abstractmethod
    def search(self, query: Sequence[str], k: int = 5) -> list[Candidate]:
        """Return up to ``k`` candidate answers for ``query``, best first.

        Return an empty list when nothing is relevant. Scores should be
        comparable across knowledge sources so the :class:`Brain` can rank them
        together (a cosine / relevance value in ``[0, 1]`` is recommended).
        """
        raise NotImplementedError


class NoRelevantCandidateError(ValueError):
    """Raised when no knowledge source can answer a query well enough.

    This happens when the brain holds no knowledge, when no source returns any
    candidate, or when every candidate scores below the rejection threshold.
    """


class Brain:
    """Routes a query to its knowledge sources and returns the best candidate."""

    def __init__(self) -> None:
        self._knowledge_base: set[Knowledge] = set()

    def add_knowledge(self, knowledge: Knowledge) -> None:
        self._knowledge_base.add(knowledge)

    def best_candidate(
        self,
        query: Sequence[str],
        threshold: float = CANDIDATE_REJECTION_THRESHOLD,
        k: int = 5,
    ) -> Candidate:
        """Return the highest-scoring candidate for ``query`` across all sources.

        Each knowledge source is asked to ``search`` the query; the pooled
        candidates are ranked by score. A candidate is *rejected* when its score
        is below ``threshold``. If every candidate is rejected -- i.e. the query
        is unrelated to everything the brain knows -- a
        :class:`NoRelevantCandidateError` is raised.
        """
        candidates = [
            candidate
            for knowledge in self._knowledge_base
            for candidate in knowledge.search(query, k)
        ]
        if not candidates:
            raise NoRelevantCandidateError("no knowledge source returned a candidate")

        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score < threshold:
            raise NoRelevantCandidateError(
                f"all candidates rejected: best score {best.score:.3f} is below "
                f"the rejection threshold {threshold}"
            )
        return best
