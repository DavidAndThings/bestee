from collections.abc import Sequence

import pytest

from bestee_chat.engine import (
    Brain,
    Candidate,
    Knowledge,
    NoRelevantCandidateError,
)


class _FakeKnowledge(Knowledge):
    """Knowledge that returns a fixed list of candidates, ignoring the query."""

    def __init__(self, candidates: list[Candidate]) -> None:
        super().__init__()
        self._candidates = candidates

    def search(self, query: Sequence[str], k: int = 5) -> list[Candidate]:
        return self._candidates[:k]


def _brain(*knowledge: Knowledge) -> Brain:
    brain = Brain()
    for piece in knowledge:
        brain.add_knowledge(piece)
    return brain


def test_returns_highest_scoring_candidate() -> None:
    brain = _brain(
        _FakeKnowledge([Candidate("low", 0.6, "a"), Candidate("high", 0.9, "a")])
    )
    best = brain.best_candidate(["q"])
    assert best.answer == "high"
    assert best.score == pytest.approx(0.9)


def test_pools_candidates_across_sources() -> None:
    brain = _brain(
        _FakeKnowledge([Candidate("from-a", 0.70, "a")]),
        _FakeKnowledge([Candidate("from-b", 0.95, "b")]),
    )
    best = brain.best_candidate(["q"])
    assert best.answer == "from-b"
    assert best.source == "b"


def test_rejects_when_all_candidates_below_threshold() -> None:
    brain = _brain(_FakeKnowledge([Candidate("meh", 0.3, "a")]))
    with pytest.raises(NoRelevantCandidateError):
        brain.best_candidate(["q"])


def test_accepts_at_threshold_boundary() -> None:
    brain = _brain(_FakeKnowledge([Candidate("ok", 0.5, "a")]))
    assert brain.best_candidate(["q"], threshold=0.5).answer == "ok"


def test_empty_brain_raises() -> None:
    with pytest.raises(NoRelevantCandidateError):
        Brain().best_candidate(["anything"])


def test_source_returning_no_candidates_raises() -> None:
    brain = _brain(_FakeKnowledge([]))
    with pytest.raises(NoRelevantCandidateError):
        brain.best_candidate(["q"])


def test_threshold_argument_controls_rejection() -> None:
    brain = _brain(_FakeKnowledge([Candidate("x", 0.7, "a")]))
    assert brain.best_candidate(["q"], threshold=0.5).answer == "x"  # accepted
    with pytest.raises(NoRelevantCandidateError):
        brain.best_candidate(["q"], threshold=0.8)  # rejected
