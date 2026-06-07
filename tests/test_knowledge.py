import json
import os
from pathlib import Path

import pytest

from bestee_chat.engine import Brain, Exchange
from bestee_chat.knowledge import AboutPerson

_HILLARY_JSON = (
    Path(__file__).resolve().parent.parent / "resources" / "hillary_clinton.json"
)


# ---------------------------------------------------------------------------
# Unit tests (no model / no network)
# ---------------------------------------------------------------------------


def test_init_from_json_object() -> None:
    person = AboutPerson(
        {"name": "Jane Doe", "Born": "1 January 1990", "Spouse": "John Doe"}
    )
    exchanges = person.generate_exchanges()
    assert len(exchanges) == 2  # "name" is excluded
    assert all(isinstance(ex, Exchange) for ex in exchanges)


def test_init_from_keyword_fields() -> None:
    person = AboutPerson(name="Jane Doe", Born="1 January 1990", Spouse="John Doe")
    assert len(person.generate_exchanges()) == 2


def test_name_is_woven_into_questions_not_its_own_field() -> None:
    person = AboutPerson(name="Jane Doe", Born="1 January 1990")
    (exchange,) = person.generate_exchanges()
    question = " ".join(exchange.user_utterance).lower()
    assert "jane doe" in question
    assert "born" in question


def test_answer_contains_name_and_value() -> None:
    person = AboutPerson(name="Jane Doe", Born="1 January 1990")
    (exchange,) = person.generate_exchanges()
    answer = " ".join(exchange.bot_utterance)
    assert "Jane Doe" in answer
    assert "1 January 1990" in answer


def test_empty_values_are_skipped() -> None:
    person = AboutPerson(name="Jane Doe", Born="", Spouse="John Doe")
    assert len(person.generate_exchanges()) == 1


def test_missing_name_returns_empty_set() -> None:
    person = AboutPerson(Born="1 January 1990")
    assert person.generate_exchanges() == set()


def test_field_name_normalises_labels() -> None:
    assert AboutPerson._field_name("Resting place") == "resting place"
    assert (
        AboutPerson._field_name("Other political, affiliations")
        == "other political affiliations"
    )


def test_dunder_metadata_fields_are_skipped() -> None:
    person = AboutPerson(
        {
            "__type__": "__person__",
            "__id__": "abc-123",
            "name": "Jane Doe",
            "Born": "1990",
        }
    )
    exchanges = person.generate_exchanges()
    assert len(exchanges) == 1  # only "Born"; name + dunder metadata excluded
    text = " ".join(
        " ".join(ex.user_utterance) + " " + " ".join(ex.bot_utterance)
        for ex in exchanges
    )
    assert "abc-123" not in text
    assert "__id__" not in text and "__type__" not in text


def test_is_metadata_only_matches_dunder_keys() -> None:
    assert AboutPerson._is_metadata("__id__")
    assert AboutPerson._is_metadata("__type__")
    assert not AboutPerson._is_metadata("name")
    assert not AboutPerson._is_metadata("Born")
    assert not AboutPerson._is_metadata("_id")  # single underscore is not metadata


def test_labels_with_spaces_become_questions() -> None:
    person = AboutPerson({"name": "Jane Doe", "Resting place": "Old Cemetery"})
    (exchange,) = person.generate_exchanges()
    question = " ".join(exchange.user_utterance).lower()
    assert "resting place" in question


def test_utterances_are_hashable_tuples() -> None:
    # generate_exchanges returns a set, which requires hashable Exchanges,
    # which in turn requires tuple (not list) utterances.
    person = AboutPerson(name="Jane Doe", Born="1 January 1990")
    (exchange,) = person.generate_exchanges()
    assert isinstance(exchange.user_utterance, tuple)
    assert isinstance(exchange.bot_utterance, tuple)
    assert hash(exchange) == hash(exchange)


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
    questions = " ".join(
        " ".join(ex.user_utterance) for k in knowledge for ex in k.generate_exchanges()
    )
    assert "Ada Lovelace" in questions
    assert "Alan Turing" in questions


# ---------------------------------------------------------------------------
# Integration test (real model + scraped resource, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("BESTEE_RUN_INTEGRATION"),
    reason="set BESTEE_RUN_INTEGRATION=1 to run model-backed tests",
)
@pytest.mark.skipif(
    not _HILLARY_JSON.exists(),
    reason="resources/hillary_clinton.json is not present",
)
def test_brain_answers_birth_question_from_infobox() -> None:
    data = json.loads(_HILLARY_JSON.read_text(encoding="utf-8"))
    brain = Brain()
    brain.add_knowledge(AboutPerson(data))

    exchange, score = brain.pick_highest_ranked_exchange(
        "When was Hillary Clinton born?".split()
    )
    answer = " ".join(exchange.bot_utterance)

    assert score > 0.0
    assert "1947" in answer or "born" in answer.lower()
