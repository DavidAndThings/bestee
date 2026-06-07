from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dotenv import load_dotenv

from bestee_chat.config import _DEFAULT_PERSONS_DIR, _PERSONS_DIR_ENV
from bestee_chat.embeddings import cosine_scores, default_encoder
from bestee_chat.engine import Candidate, Knowledge

if TYPE_CHECKING:
    from bestee_chat.embeddings import Encoder

# Natural-language phrasings for common infobox fields. Anything not listed
# here falls back to the generic templates below.
_QUESTION_TEMPLATES = {
    "born": "When and where was {name} born?",
    "died": "When and where did {name} die?",
    "spouse": "Who is {name}'s spouse?",
    "spouses": "Who has {name} been married to?",
    "children": "Who are {name}'s children?",
    "parents": "Who are {name}'s parents?",
    "education": "Where was {name} educated?",
    "awards": "What awards has {name} received?",
    "website": "What is {name}'s website?",
}

_ANSWER_TEMPLATES = {
    "born": "{name} was born {value}.",
    "died": "{name} died {value}.",
}

_DEFAULT_QUESTION = "What is {name}'s {field}?"
_DEFAULT_ANSWER = "{name}'s {field} is {value}."


class AboutPerson(Knowledge):
    """Knowledge derived from a person's Wikipedia infobox.

    Each infobox field becomes a templated (question, answer) pair. The question
    embeddings are precomputed once; :meth:`search` encodes the query a single
    time and ranks all questions against it at once.

    Initialise it with an infobox mapping, e.g. the JSON produced by
    :func:`bestee_chat.wikipedia.scrape_infobox`::

        import json

        with open("resources/persons/0001.json") as f:
            people = [AboutPerson(record) for record in json.load(f)]

    Individual fields may also be passed as keyword arguments
    (``AboutPerson(name="Ada Lovelace", born="1815")``).
    """

    def __init__(
        self,
        data: Mapping[str, object] | None = None,
        encoder: Encoder | None = None,
        **fields: object,
    ) -> None:
        super().__init__(encoder)
        merged: dict[str, object] = dict(data) if data else {}
        merged.update(fields)
        self._data = merged
        self._name = str(merged.get("name", "")).strip()
        self._answers: list[str] = []
        self._question_embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        """The subject's name (empty string if the infobox has none)."""
        return self._name

    @staticmethod
    def build_from_cache() -> set[AboutPerson]:
        """Build an :class:`AboutPerson` for every cached person record.

        Reads the JSON-array files written by
        :func:`bestee_chat.wikipedia.learn_about_a_person` from the persons
        directory (``resources/persons`` by default, or ``$BESTEE_PERSONS_DIR``).
        Each file may hold several records, so every record becomes its own
        knowledge object.
        """
        load_dotenv()
        persons_dir = Path(os.getenv(_PERSONS_DIR_ENV) or _DEFAULT_PERSONS_DIR)

        persons: set[AboutPerson] = set()
        for path in persons_dir.glob("*.json"):
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            persons.update(AboutPerson(record) for record in records)
        return persons

    def search(self, query: Sequence[str], k: int = 5) -> list[Candidate]:
        """Return the answers whose templated questions best match ``query``.

        The query is encoded once and compared against the precomputed question
        embeddings; the ``k`` best answers are returned as :class:`Candidate`
        objects scored by cosine similarity.
        """
        embeddings = self._ensure_built()
        query_text = " ".join(query).strip()
        if embeddings.size == 0 or not query_text:
            return []

        encoder = self._encoder or default_encoder()
        query_embedding = encoder.encode(query_text)
        scores = cosine_scores(embeddings, query_embedding)

        ranked = np.argsort(scores)[::-1][:k]
        return [
            Candidate(
                answer=self._answers[int(i)],
                score=float(scores[int(i)]),
                source=self._name,
            )
            for i in ranked
        ]

    def _ensure_built(self) -> np.ndarray:
        """Build the (question, answer) pairs and precompute question embeddings.

        The ``name`` field identifies the subject and is woven into every
        question and answer rather than becoming a pair of its own. Metadata
        fields (keys wrapped in double underscores, e.g. ``__id__``) and empty
        fields are skipped.
        """
        if self._question_embeddings is not None:
            return self._question_embeddings

        questions: list[str] = []
        answers: list[str] = []
        if self._name:
            for label, raw_value in self._data.items():
                if label == "name" or self._is_metadata(label):
                    continue
                value = str(raw_value).strip()
                if not value:
                    continue

                field = self._field_name(label)
                question = _QUESTION_TEMPLATES.get(field, _DEFAULT_QUESTION).format(
                    name=self._name, field=field
                )
                answer = _ANSWER_TEMPLATES.get(field, _DEFAULT_ANSWER).format(
                    name=self._name, field=field, value=value
                )
                questions.append(question)
                answers.append(answer)

        if questions:
            encoder = self._encoder or default_encoder()
            embeddings = np.vstack([encoder.encode(question) for question in questions])
        else:
            embeddings = np.zeros((0, 0))

        self._answers = answers
        self._question_embeddings = embeddings
        return embeddings

    @staticmethod
    def _is_metadata(label: str) -> bool:
        """Return True for serialization metadata keys.

        Metadata keys are wrapped in double underscores, e.g. ``__type__`` or
        ``__id__``. Note that ``name`` is a regular field, not metadata.
        """
        return label.startswith("__") and label.endswith("__")

    @staticmethod
    def _field_name(label: str) -> str:
        """Normalise an infobox label into a lowercase field name."""
        cleaned = label.replace(",", " ").replace("\n", " ")
        return " ".join(cleaned.split()).lower()
