import json
from pathlib import Path

import pytest

from bestee_chat import learn
from bestee_chat.wiki.models import SearchHit


def test_learn_article_stores_the_top_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hit = SearchHit(
        rank=1,
        score=2.0,
        title="Ada Lovelace",
        snippet="An English mathematician.",
        url="https://simple.wikipedia.org/wiki/Ada_Lovelace",
        page_id=1,
    )
    monkeypatch.setattr(learn, "search_wiki", lambda *args, **kwargs: [hit])
    monkeypatch.setenv("BESTEE_ARTICLES_DIR", str(tmp_path / "articles"))

    path = learn.learn_article("Ada Lovelace")

    (record,) = json.loads(path.read_text(encoding="utf-8"))
    assert record["__type__"] == "article"
    assert record["title"] == "Ada Lovelace"
    assert record["summary"] == "An English mathematician."
    assert record["__url__"].endswith("Ada_Lovelace")


def test_learn_article_without_results_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(learn, "search_wiki", lambda *args, **kwargs: [])
    with pytest.raises(LookupError):
        learn.learn_article("nothing relevant")
