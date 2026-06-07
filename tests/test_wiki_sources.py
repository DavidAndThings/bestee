"""URL/slug building for WikiSource (pure, no network)."""

from __future__ import annotations

from bestee_chat.wiki.sources import WikiSource


def test_latest_multistream_urls() -> None:
    src = WikiSource(lang="simple", multistream=True)
    assert src.dbname == "simplewiki"
    assert src.date_segment == "latest"
    assert src.data_filename == ("simplewiki-latest-pages-articles-multistream.xml.bz2")
    assert src.index_filename == (
        "simplewiki-latest-pages-articles-multistream-index.txt.bz2"
    )
    assert src.data_url.startswith("https://dumps.wikimedia.org/simplewiki/latest/")
    assert src.checksums_url.endswith("simplewiki-latest-sha1sums.txt")


def test_single_stream_has_no_index() -> None:
    src = WikiSource(lang="en", multistream=False)
    assert src.data_filename == "enwiki-latest-pages-articles.xml.bz2"
    assert src.index_filename is None
    assert src.index_url is None


def test_dated_pin() -> None:
    src = WikiSource(lang="simple", dated="20240601")
    assert src.date_segment == "20240601"
    assert "20240601" in src.data_filename
    assert src.slug == "simplewiki/20240601"


def test_page_url() -> None:
    src = WikiSource(lang="simple")
    url = src.page_url("List of English monarchs")
    assert url == "https://simple.wikipedia.org/wiki/List_of_English_monarchs"
