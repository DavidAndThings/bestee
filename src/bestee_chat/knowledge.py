from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bestee_chat.engine import Exchange, Knowledge

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

    Initialise it with an infobox mapping, e.g. the JSON produced by
    :func:`bestee_chat.wikipedia.scrape_infobox`::

        import json

        with open("resources/hillary_clinton.json") as f:
            person = AboutPerson(json.load(f))

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

    def generate_exchanges(self) -> set[Exchange]:
        """Turn each infobox field into a question/answer :class:`Exchange`.

        The ``name`` field identifies the subject and is woven into every
        question and answer rather than becoming an exchange of its own.
        Metadata fields (keys wrapped in double underscores, e.g. ``__id__``)
        and empty fields are skipped. Returns an empty set when no name is
        available.
        """
        name = str(self._data.get("name", "")).strip()
        if not name:
            return set()

        exchanges: set[Exchange] = set()
        for label, raw_value in self._data.items():
            if label == "name" or self._is_metadata(label):
                continue
            value = str(raw_value).strip()
            if not value:
                continue

            field = self._field_name(label)
            question = _QUESTION_TEMPLATES.get(field, _DEFAULT_QUESTION).format(
                name=name, field=field
            )
            answer = _ANSWER_TEMPLATES.get(field, _DEFAULT_ANSWER).format(
                name=name, field=field, value=value
            )
            exchanges.add(
                Exchange(
                    user_utterance=tuple(question.split()),
                    bot_utterance=tuple(answer.split()),
                    encoder=self._encoder,
                )
            )
        return exchanges

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
