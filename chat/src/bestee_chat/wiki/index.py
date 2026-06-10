"""SQLite FTS5 lexical index: build from pages, query with BM25.

This is the always-present retrieval layer. It uses SQLite's built-in FTS5
module (so no third-party search engine) and BM25 ranking with per-column
weights -- the title is weighted most heavily, which is what makes title-like
queries such as "List of English kings" land on the right page.

Wikipedia *redirects* are folded in as searchable aliases on their target
page, so a query phrased differently from the canonical title (kings vs
monarchs) still matches via the redirect that actually exists in the dump.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bestee_chat.wiki.clean import clean_page
from bestee_chat.wiki.models import RawPage, SearchHit
from bestee_chat.wiki.sources import WikiSource

# Column weights for bm25(): title, aliases, summary, body.
_BM25_WEIGHTS = (10.0, 6.0, 3.0, 1.0)
_TERM = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class BuildStats:
    """Summary of an index build."""

    articles: int
    redirects: int
    aliases_linked: int


def _check_fts5(conn: sqlite3.Connection) -> None:
    """Raise a clear error if this SQLite build lacks the FTS5 module."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "this Python's SQLite was built without the FTS5 extension, "
            "which the wiki lexical index requires"
        ) from exc


def _match_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression (OR of terms).

    Using OR maximises recall; BM25 then ranks, so common words like "of"
    contribute little while distinctive words dominate. Returns ``None`` when
    the query has no usable terms.
    """
    terms = _TERM.findall(query)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


class LexicalIndex:
    """A BM25 full-text index backed by a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> LexicalIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def build(self, pages: Iterable[RawPage], source: WikiSource) -> BuildStats:
        """Build the index from an iterable of :class:`RawPage` objects.

        Articles are indexed immediately; redirects are buffered and, once all
        articles are in, attached to their targets as searchable aliases.
        """
        conn = self._conn
        _check_fts5(conn)
        conn.executescript(
            """
            DROP TABLE IF EXISTS meta;
            DROP TABLE IF EXISTS pages;
            DROP TABLE IF EXISTS pages_fts;
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE pages(
                rowid INTEGER PRIMARY KEY,
                page_id INTEGER,
                title TEXT,
                url TEXT
            );
            CREATE VIRTUAL TABLE pages_fts USING fts5(
                title, aliases, summary, body,
                tokenize='porter unicode61'
            );
            """
        )

        title_to_rowid: dict[str, int] = {}
        redirects: dict[str, list[str]] = {}
        articles = 0
        redirect_count = 0

        with conn:
            for page in pages:
                if page.is_redirect and page.namespace == 0:
                    redirect_count += 1
                    if page.redirect_target:
                        redirects.setdefault(page.redirect_target, []).append(
                            page.title
                        )
                    continue
                if page.namespace != 0:
                    continue

                summary, body = clean_page(page.text)
                cur = conn.execute(
                    "INSERT INTO pages(page_id, title, url) VALUES (?, ?, ?)",
                    (page.page_id, page.title, source.page_url(page.title)),
                )
                rowid = cur.lastrowid
                conn.execute(
                    "INSERT INTO pages_fts(rowid, title, aliases, summary, body) "
                    "VALUES (?, ?, '', ?, ?)",
                    (rowid, page.title, summary, body),
                )
                title_to_rowid[page.title] = int(rowid or 0)
                articles += 1

            aliases_linked = self._attach_aliases(redirects, title_to_rowid)
            self._write_meta(source, articles)

        return BuildStats(
            articles=articles,
            redirects=redirect_count,
            aliases_linked=aliases_linked,
        )

    def _attach_aliases(
        self,
        redirects: dict[str, list[str]],
        title_to_rowid: dict[str, int],
    ) -> int:
        """Fold redirect titles into their target pages' ``aliases`` column."""
        linked = 0
        for target, alias_titles in redirects.items():
            rowid = title_to_rowid.get(target)
            if rowid is None:
                continue
            self._conn.execute(
                "UPDATE pages_fts SET aliases = ? WHERE rowid = ?",
                (" ; ".join(alias_titles), rowid),
            )
            linked += len(alias_titles)
        return linked

    def _write_meta(self, source: WikiSource, articles: int) -> None:
        rows = {
            "dbname": source.dbname,
            "date": source.date_segment,
            "articles": str(articles),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            list(rows.items()),
        )

    def article_count(self) -> int:
        """Return the number of indexed articles."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM pages").fetchone()
        return int(row["n"]) if row else 0

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Return up to ``limit`` BM25-ranked hits for ``query``."""
        match = _match_query(query)
        if match is None:
            return []
        sql = (
            "SELECT p.page_id AS page_id, p.title AS title, p.url AS url, "
            "snippet(pages_fts, 2, '', '', ' ... ', 16) AS snippet, "
            "bm25(pages_fts, ?, ?, ?, ?) AS bm25 "
            "FROM pages_fts JOIN pages p ON p.rowid = pages_fts.rowid "
            "WHERE pages_fts MATCH ? "
            "ORDER BY bm25 LIMIT ?"
        )
        params = (*_BM25_WEIGHTS, match, limit)
        rows = self._conn.execute(sql, params).fetchall()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            snippet = (row["snippet"] or "").strip() or row["title"]
            hits.append(
                SearchHit(
                    rank=rank,
                    score=-float(row["bm25"]),  # bm25 is negative; flip so up=better
                    title=row["title"],
                    snippet=snippet,
                    url=row["url"],
                    page_id=int(row["page_id"]),
                )
            )
        return hits
