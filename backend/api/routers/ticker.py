"""Search over the catalog of SIC industry titles and company names.

``get_all_search_terms`` builds the searchable catalog -- every SIC industry
title plus every company name -- and caches it (the first call fetches the full
ticker universe and the SIC list, so the first request warms the cache and later
ones are instant).  ``GET /search`` ranks case-insensitive matches against it.
"""

from collections.abc import Sequence
from functools import cache
from typing import Annotated

from bestee_compute.stocks.sic import get_sic_codes_df
from bestee_compute.stocks.tickers import get_all_tickers_df
from fastapi import APIRouter, Query

from schemas import SearchResults

router = APIRouter(prefix="/search", tags=["search"])


@cache
def get_all_search_terms() -> Sequence[str]:
    """The searchable catalog: SIC industry titles plus every company name.

    Cached after the first (network-bound) call.
    """

    sic_industry_titles = get_sic_codes_df().get_column("Industry Title").to_list()
    company_names = get_all_tickers_df().get_column("Name").to_list()
    return sic_industry_titles + company_names


def _search(query: str, terms: Sequence[str], limit: int) -> list[str]:
    """Case-insensitive substring search; prefix matches ranked first.

    De-duplicates terms that fold to the same string, returns prefix matches
    (sorted) ahead of other substring matches (sorted), capped at *limit*.
    """
    needle = query.casefold()
    prefix: list[str] = []
    other: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:  # the ticker "Name" column is nullable
            continue
        folded = term.casefold()
        if needle not in folded or folded in seen:
            continue
        seen.add(folded)
        (prefix if folded.startswith(needle) else other).append(term)
    return (sorted(prefix) + sorted(other))[:limit]


@router.get("", response_model=SearchResults)
def search_terms(
    q: Annotated[str, Query(min_length=1, description="Substring to search for.")],
    limit: Annotated[int, Query(ge=1, le=100, description="Max results.")] = 20,
) -> SearchResults:
    """Search SIC industry titles and company names for *q* (case-insensitive).

    Prefix matches rank above other substring matches; results are capped at
    *limit*.
    """
    matches = _search(q, get_all_search_terms(), limit)
    return SearchResults(query=q, count=len(matches), results=matches)
