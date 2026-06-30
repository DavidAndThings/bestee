"""Search over securities (by symbol or name) and SIC industry titles.

``get_search_catalog`` builds the searchable catalog -- every ticker (matchable
by its symbol or company/ETF name) plus every SIC industry title -- and caches
it (the first call fetches the full ticker universe and the SIC list, so the
first request warms the cache and later ones are instant). ``GET /search`` ranks
case-insensitive matches against it: exact matches first, then prefixes, then
other substrings.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Literal

from bestee_compute.stocks.sic import get_sic_codes_df
from bestee_compute.stocks.tickers import get_all_tickers_df
from fastapi import APIRouter, Query

from schemas import SearchResult, SearchResults

router = APIRouter(prefix="/search", tags=["search"])


@dataclass(frozen=True)
class _CatalogEntry:
    """A searchable catalog entry and the casefolded fields to match against.

    A security is matched by its symbol *or* its name; a SIC industry only by
    its title (``kind`` lets the UI tell the two apart, since an industry term
    later expands to many tickers).
    """

    value: str
    label: str
    kind: Literal["ticker", "sic"]
    ticker: str | None
    name: str | None
    haystacks: tuple[str, ...]


@cache
def get_search_catalog() -> Sequence[_CatalogEntry]:
    """The searchable catalog: every security (by symbol or name) and every SIC
    industry title. Cached after the first (network-bound) call.
    """
    entries: list[_CatalogEntry] = []
    # SIC industries -- a single term that expands to many tickers; the user
    # picks the title, so it stays a one-line entry matched by that title.
    for title in get_sic_codes_df().get_column("Industry Title").to_list():
        if title:
            entries.append(
                _CatalogEntry(
                    value=title,
                    label=title,
                    kind="sic",
                    ticker=None,
                    name=None,
                    haystacks=(title.casefold(),),
                )
            )
    # Securities (stocks and ETFs alike) -- matched by symbol or company name,
    # shown as "Name (SYMBOL)" and stored/resolved by symbol.
    tickers = get_all_tickers_df()
    symbols = tickers.get_column("Ticker").to_list()
    names = tickers.get_column("Name").to_list()
    for symbol, name in zip(symbols, names):
        if not symbol:
            continue
        label = f"{name} ({symbol})" if name else symbol
        haystacks = (
            (symbol.casefold(), name.casefold()) if name else (symbol.casefold(),)
        )
        entries.append(
            _CatalogEntry(
                value=symbol,
                label=label,
                kind="ticker",
                ticker=symbol,
                name=name,
                haystacks=haystacks,
            )
        )
    return entries


def _match_tier(entry: _CatalogEntry, needle: str) -> int | None:
    """Best match tier of *needle* against *entry* (lower is better), or ``None``.

    ``0`` is an exact field match (e.g. typing a whole ticker), ``1`` a prefix
    match, ``2`` any other substring -- taken over the entry's symbol and name
    (or its industry title).
    """
    best: int | None = None
    for field in entry.haystacks:
        if needle not in field:
            continue
        tier = 0 if field == needle else 1 if field.startswith(needle) else 2
        best = tier if best is None else min(best, tier)
    return best


def _search(
    query: str, catalog: Sequence[_CatalogEntry], limit: int
) -> list[_CatalogEntry]:
    """Case-insensitive search; exact, then prefix, then substring matches.

    Within a tier, entries order by label. Entries sharing a ``value`` are
    de-duplicated (keeping the best-ranked), and the result is capped at *limit*.
    """
    needle = query.casefold()
    ranked: list[tuple[int, str, _CatalogEntry]] = []
    for entry in catalog:
        tier = _match_tier(entry, needle)
        if tier is not None:
            ranked.append((tier, entry.label.casefold(), entry))
    ranked.sort(key=lambda item: (item[0], item[1]))
    out: list[_CatalogEntry] = []
    seen: set[str] = set()
    for _, _, entry in ranked:
        if entry.value in seen:
            continue
        seen.add(entry.value)
        out.append(entry)
        if len(out) >= limit:
            break
    return out


@router.get("", response_model=SearchResults)
def search_terms(
    q: Annotated[str, Query(min_length=1, description="Substring to search for.")],
    limit: Annotated[int, Query(ge=1, le=100, description="Max results.")] = 20,
) -> SearchResults:
    """Search securities (by symbol or name) and SIC industries for *q*.

    Exact matches rank above prefixes, which rank above other substring matches;
    results are capped at *limit*.
    """
    matches = _search(q, get_search_catalog(), limit)
    results = [
        SearchResult(
            value=entry.value,
            label=entry.label,
            kind=entry.kind,
            ticker=entry.ticker,
            name=entry.name,
        )
        for entry in matches
    ]
    return SearchResults(query=q, count=len(results), results=results)
