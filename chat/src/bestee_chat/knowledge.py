from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from dotenv import load_dotenv

from bestee_chat import config
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


def _load_cached_records(kind: str) -> list[dict[str, object]]:
    """Load all cached records of ``kind`` from its knowledge directory.

    Reads the JSON-array files written by ``bestee-chat learn`` from
    ``config.knowledge_dir(kind)``; each file may hold several records.
    """
    load_dotenv()
    directory = config.knowledge_dir(kind)
    records: list[dict[str, object]] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        if isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict))
    return records


class AboutPerson(Knowledge):
    """Knowledge derived from a person's Wikipedia infobox.

    Each infobox field becomes a templated (question, answer) pair. The question
    embeddings are precomputed once; :meth:`search` encodes the query a single
    time and ranks all questions against it at once.

    Initialise it with an infobox mapping (e.g. the dict produced by
    :func:`bestee_chat.wikipedia.scrape_infobox`) or pass fields as keyword
    arguments (``AboutPerson(name="Ada Lovelace", born="1815")``). To load every
    person learned via ``bestee-chat learn person``, use
    :meth:`build_from_cache`.
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
        """Build an :class:`AboutPerson` for every cached ``person`` record."""
        return {AboutPerson(record) for record in _load_cached_records("persons")}

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


class AboutArticle(Knowledge):
    """Knowledge from a learned Wikipedia article (title + cached summary).

    Built by ``bestee-chat learn article``. Each article answers queries that
    are semantically close to its title with its stored summary.
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
        self._title = str(merged.get("title", "")).strip()
        self._summary = str(merged.get("summary", "")).strip()
        self._embedding: np.ndarray | None = None

    @property
    def title(self) -> str:
        """The article's title (used as the candidate source)."""
        return self._title

    @staticmethod
    def build_from_cache() -> set[AboutArticle]:
        """Build an :class:`AboutArticle` for every cached ``article`` record."""
        return {AboutArticle(record) for record in _load_cached_records("articles")}

    def search(self, query: Sequence[str], k: int = 5) -> list[Candidate]:
        """Return the article's summary, scored by title-to-query similarity."""
        query_text = " ".join(query).strip()
        if not self._title or not self._summary or not query_text:
            return []

        encoder = self._encoder or default_encoder()
        embedding = self._embedding
        if embedding is None:
            embedding = encoder.encode(self._title)
            self._embedding = embedding

        query_vec = encoder.encode(query_text)
        scores = cosine_scores(embedding[np.newaxis, :], query_vec)
        return [
            Candidate(answer=self._summary, score=float(scores[0]), source=self._title)
        ]
