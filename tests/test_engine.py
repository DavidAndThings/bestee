import os

import numpy as np
import pytest

from bestee_chat.engine import (
    Brain,
    Exchange,
    Knowledge,
    NoRelevantExchangeError,
)


class FakeEncoder:
    """Deterministic encoder for fast, offline unit tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.calls: list[str] = []
        self._vectors = {
            text: np.asarray(vec, dtype=np.float64) for text, vec in vectors.items()
        }

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return self._vectors[text]


def test_identical_meaning_scores_one() -> None:
    encoder = FakeEncoder({"how are you": [1.0, 0.0, 0.0]})
    exchange = Exchange("how are you".split(), ["hi"], encoder=encoder)
    assert exchange.similarity("how are you".split()) == pytest.approx(1.0)


def test_unrelated_meaning_scores_zero() -> None:
    encoder = FakeEncoder(
        {"how are you": [1.0, 0.0, 0.0], "stock prices fell": [0.0, 1.0, 0.0]}
    )
    exchange = Exchange("how are you".split(), ["hi"], encoder=encoder)
    assert exchange.similarity("stock prices fell".split()) == pytest.approx(0.0)


def test_partial_similarity_between_zero_and_one() -> None:
    encoder = FakeEncoder(
        {"how are you": [1.0, 0.0, 0.0], "how is everything": [1.0, 1.0, 0.0]}
    )
    exchange = Exchange("how are you".split(), ["hi"], encoder=encoder)
    score = exchange.similarity("how is everything".split())
    assert 0.0 < score < 1.0


def test_empty_query_returns_zero_without_encoding() -> None:
    encoder = FakeEncoder({})
    exchange = Exchange("anything at all".split(), ["hi"], encoder=encoder)
    assert exchange.similarity([]) == 0.0
    assert encoder.calls == []


def test_empty_user_utterance_returns_zero() -> None:
    encoder = FakeEncoder({})
    exchange = Exchange([], ["hi"], encoder=encoder)
    assert exchange.similarity("hello".split()) == 0.0
    assert encoder.calls == []


def test_user_embedding_is_cached() -> None:
    encoder = FakeEncoder(
        {
            "hello world": [1.0, 0.0, 0.0],
            "hi": [1.0, 0.0, 0.0],
            "hey": [1.0, 0.0, 0.0],
        }
    )
    exchange = Exchange("hello world".split(), ["x"], encoder=encoder)
    exchange.similarity("hi".split())
    exchange.similarity("hey".split())
    assert encoder.calls.count("hello world") == 1


# ---------------------------------------------------------------------------
# Brain rejection behavior (deterministic, no model)
# ---------------------------------------------------------------------------


class _FixedKnowledge(Knowledge):
    """Knowledge that yields a fixed set of exchanges."""

    def __init__(self, exchanges: set[Exchange]) -> None:
        super().__init__()
        self._exchanges = exchanges

    def generate_exchanges(self) -> set[Exchange]:
        return self._exchanges


def _brain_with(exchange: Exchange) -> Brain:
    brain = Brain()
    brain.add_knowledge(_FixedKnowledge({exchange}))
    return brain


def test_accepts_exchange_scoring_at_or_above_threshold() -> None:
    # query and the exchange's question map to the same vector -> cosine 1.0
    encoder = FakeEncoder({"when born": [1.0, 0.0], "born when": [1.0, 0.0]})
    exchange = Exchange(("when", "born"), ("In 1815.",), encoder=encoder)
    brain = _brain_with(exchange)

    result, score = brain.pick_highest_ranked_exchange(["born", "when"])

    assert result is exchange
    assert score == pytest.approx(1.0)


def test_rejects_when_every_exchange_is_below_threshold() -> None:
    # query is orthogonal to the only exchange -> cosine 0.0 < 0.5
    encoder = FakeEncoder({"when born": [1.0, 0.0], "cake recipe": [0.0, 1.0]})
    exchange = Exchange(("when", "born"), ("In 1815.",), encoder=encoder)
    brain = _brain_with(exchange)

    with pytest.raises(NoRelevantExchangeError):
        brain.pick_highest_ranked_exchange(["cake", "recipe"])


def test_empty_brain_raises() -> None:
    with pytest.raises(NoRelevantExchangeError):
        Brain().pick_highest_ranked_exchange(["anything"])


def test_threshold_argument_controls_rejection() -> None:
    # cosine([1, 1], [1, 0]) = 1 / sqrt(2) ~= 0.707
    encoder = FakeEncoder({"a b": [1.0, 1.0], "q": [1.0, 0.0]})
    exchange = Exchange(("a", "b"), ("answer",), encoder=encoder)
    brain = _brain_with(exchange)

    _, score = brain.pick_highest_ranked_exchange(["q"])  # default 0.5 -> accepted
    assert score == pytest.approx(0.70710, abs=1e-4)

    with pytest.raises(NoRelevantExchangeError):
        brain.pick_highest_ranked_exchange(["q"], threshold=0.8)


@pytest.mark.skipif(
    not os.environ.get("BESTEE_RUN_INTEGRATION"),
    reason="set BESTEE_RUN_INTEGRATION=1 to run model-backed tests",
)
def test_semantic_similarity_with_real_model() -> None:
    pytest.importorskip("sentence_transformers")
    from bestee_chat.embeddings import SemanticEncoder

    try:
        encoder = SemanticEncoder()
        encoder.warm_up()
    except Exception as exc:  # pragma: no cover - depends on network/hardware
        pytest.skip(f"model unavailable: {exc}")

    exchange = Exchange(
        "how do I reset my password".split(),
        ["Visit the settings page."],
        encoder=encoder,
    )
    paraphrase = exchange.similarity("I forgot my login credentials".split())
    unrelated = exchange.similarity("what time does the match start".split())
    assert paraphrase > unrelated
