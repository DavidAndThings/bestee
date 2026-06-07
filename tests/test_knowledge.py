import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from bestee_chat.engine import Brain, Candidate
from bestee_chat.knowledge import AboutPerson


class FakeEncoder:
    """Deterministic bag-of-words encoder for fast, offline tests.

    Tokens are hashed into a fixed-size vector, so texts that share words have
    higher cosine similarity -- enough to exercise ranking without a model.
    """

    _DIM = 256

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._DIM, dtype=np.float64)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).hexdigest()
            vector[int(digest, 16) % self._DIM] += 1.0
        return vector


# ---------------------------------------------------------------------------
# AboutPerson.search (no model)
# ---------------------------------------------------------------------------


def test_search_returns_candidates_from_this_person() -> None:
    person = AboutPerson(
        {"name": "Jane Doe", "Born": "1 January 1990", "Spouse": "John Doe"},
        encoder=FakeEncoder(),
    )
    candidates = person.search("tell me something".split())
    assert candidates
    assert all(isinstance(c, Candidate) for c in candidates)
    assert all(c.source == "Jane Doe" for c in candidates)


def test_search_ranks_the_relevant_field_first() -> None:
    person = AboutPerson(
        {
            "name": "Jane Doe",
            "Born": "1 January 1990",
            "Spouse": "John Doe",
            "Education": "Oxford",
        },
        encoder=FakeEncoder(),
    )
    best = person.search("when was Jane Doe born".split())[0]
    assert best.answer == "Jane Doe was born 1 January 1990."


def test_search_excludes_metadata_and_name_fields() -> None:
    person = AboutPerson(
        {"__id__": "abc-123", "name": "Jane Doe", "Born": "1990"},
        encoder=FakeEncoder(),
    )
    answers = " ".join(c.answer for c in person.search("id".split()))
    assert "abc-123" not in answers
    assert "born" in answers.lower()  # only the Born field produced a candidate


def test_search_returns_empty_without_a_name() -> None:
    person = AboutPerson({"Born": "1990"}, encoder=FakeEncoder())
    assert person.search("when born".split()) == []


def test_search_returns_empty_for_blank_query() -> None:
    person = AboutPerson({"name": "Jane Doe", "Born": "1990"}, encoder=FakeEncoder())
    assert person.search([]) == []


def test_search_respects_k() -> None:
    person = AboutPerson(
        {
            "name": "Jane Doe",
            "Born": "1990",
            "Spouse": "John",
            "Children": "Kid",
            "Party": "Independent",
            "Awards": "A prize",
        },
        encoder=FakeEncoder(),
    )
    assert len(person.search("anything".split(), k=2)) == 2


# ---------------------------------------------------------------------------
# Field/metadata helpers and cache loading
# ---------------------------------------------------------------------------


def test_field_name_normalises_labels() -> None:
    assert AboutPerson._field_name("Resting place") == "resting place"
    assert (
        AboutPerson._field_name("Other political, affiliations")
        == "other political affiliations"
    )


def test_is_metadata_only_matches_dunder_keys() -> None:
    assert AboutPerson._is_metadata("__id__")
    assert AboutPerson._is_metadata("__type__")
    assert not AboutPerson._is_metadata("name")
    assert not AboutPerson._is_metadata("Born")
    assert not AboutPerson._is_metadata("_id")  # single underscore is not metadata


def test_build_from_cache_loads_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persons_dir = tmp_path / "persons"
    persons_dir.mkdir(parents=True)
    # A single cache file may hold several person records.
    (persons_dir / "0001.json").write_text(
        json.dumps(
            [
                {"__type__": "person", "name": "Ada Lovelace", "Born": "1815"},
                {"__type__": "person", "name": "Alan Turing", "Born": "1912"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BESTEE_PERSONS_DIR", str(persons_dir))

    knowledge = AboutPerson.build_from_cache()

    assert len(knowledge) == 2
    assert {k.name for k in knowledge} == {"Ada Lovelace", "Alan Turing"}


# ---------------------------------------------------------------------------
# Integration test (real model, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("BESTEE_RUN_INTEGRATION"),
    reason="set BESTEE_RUN_INTEGRATION=1 to run model-backed tests",
)
def test_brain_answers_birth_question_with_real_model() -> None:
    person = AboutPerson(
        {
            "name": "Ada Lovelace",
            "Born": "10 December 1815, London, England",
            "Spouse": "William King",
            "Known for": "the first computer program",
        }
    )
    brain = Brain()
    brain.add_knowledge(person)

    best = brain.best_candidate("when was Ada Lovelace born?".split())

    assert best.source == "Ada Lovelace"
    assert "1815" in best.answer or "born" in best.answer.lower()
    assert best.score > 0.0
