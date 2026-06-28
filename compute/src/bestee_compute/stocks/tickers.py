"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import importlib.resources
import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import platformdirs
import polars as pl
from great_tables import GT
from massive import RESTClient
from massive.rest.models import RelatedCompany, Ticker, TickerDetails

import bestee_compute.resources
from bestee_compute.client import get_client
from bestee_compute.stocks import columns as cols
from bestee_compute.stocks.sic import get_sic_codes_df

logger = logging.getLogger(__name__)

# Fields to extract from TickerDetails, in display order.  Each tuple is
# (attribute_name_on_SDK_model, column_label_in_table).  The SDK attribute
# names match the Stock dataclass fields, so the decorator pipeline reuses
# this to rebuild a Stock from a details row (label -> attribute).
DETAIL_FIELDS: list[tuple[str, str]] = [
    ("ticker", cols.TICKER),
    ("name", cols.NAME),
    ("description", cols.DESCRIPTION),
    ("type", cols.TYPE),
    ("market", cols.MARKET),
    ("locale", cols.LOCALE),
    ("primary_exchange", cols.PRIMARY_EXCHANGE),
    ("currency_name", cols.CURRENCY),
    ("cik", cols.CIK),
    ("composite_figi", cols.COMPOSITE_FIGI),
    ("share_class_figi", cols.SHARE_CLASS_FIGI),
    ("sic_code", cols.SIC_CODE),
    ("sic_description", cols.SIC_DESCRIPTION),
    ("market_cap", cols.MARKET_CAP),
    ("share_class_shares_outstanding", cols.SHARES_OUTSTANDING),
    ("weighted_shares_outstanding", cols.WEIGHTED_SHARES_OUTSTANDING),
    ("total_employees", cols.TOTAL_EMPLOYEES),
    ("list_date", cols.LIST_DATE),
    ("homepage_url", cols.HOMEPAGE_URL),
    ("phone_number", cols.PHONE_NUMBER),
    ("ticker_root", cols.TICKER_ROOT),
]


def _fetch_tickers(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> list[Ticker]:
    """Fetch every ticker from the Massive API and return raw objects.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        market: Optional market filter (e.g. ``"stocks"``, ``"crypto"``,
            ``"fx"``, ``"otc"``).
        ticker_type: Optional ticker type filter (e.g. ``"CS"`` for common
            stock, ``"ETF"``, ``"ADRC"``).  See the Ticker Types API for
            the full list.
        active: If *True* only actively-traded tickers are returned.
            *None* returns both active and delisted tickers.
        limit: Page size per request (max 1000).  A higher value means
            fewer HTTP round-trips.

    Returns:
        A list of :class:`massive.rest.models.Ticker` objects.

    Raises:
        RuntimeError: If no API key is available.
    """
    client = get_client(api_key)
    logger.info(
        "Fetching tickers (market=%s, type=%s, active=%s)", market, ticker_type, active
    )

    tickers: list[Ticker] = []
    for ticker in client.list_tickers(
        market=market,
        type=ticker_type,
        active=active,
        limit=limit,
    ):
        if isinstance(ticker, Ticker):
            tickers.append(ticker)

    logger.info("Fetched %d tickers", len(tickers))
    return tickers


def get_all_tickers_df(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> pl.DataFrame:
    """Return all tickers as a Polars DataFrame (unstyled).

    Used by the decorator pipeline; see :func:`get_all_tickers` for the
    Great Tables-wrapped public variant and full argument documentation.
    """
    logger.info(
        "Building all-tickers DataFrame (market=%s, type=%s, active=%s)",
        market,
        ticker_type,
        active,
    )
    tickers = _fetch_tickers(
        api_key=api_key,
        market=market,
        ticker_type=ticker_type,
        active=active,
        limit=limit,
    )
    return pl.DataFrame(
        [
            {
                cols.TICKER: t.ticker,
                cols.NAME: t.name,
                cols.MARKET: t.market,
                cols.TYPE: t.type,
                cols.CURRENCY: t.currency_name,
                cols.EXCHANGE: t.primary_exchange,
                cols.ACTIVE: t.active,
            }
            for t in tickers
        ]
    )


def get_all_tickers(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> GT:
    """Return every ticker offered by the Massive API as a Great Tables object.

    Fetches all tickers (auto-paginating through every page) and returns
    the result as a :class:`great_tables.GT` display table.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        market: Optional market filter (e.g. ``"stocks"``, ``"crypto"``,
            ``"fx"``, ``"otc"``).
        ticker_type: Optional ticker type filter (e.g. ``"CS"`` for common
            stock, ``"ETF"``, ``"ADRC"``).  See the Ticker Types API for
            the full list.
        active: If *True* only actively-traded tickers are returned.
            *None* returns both active and delisted tickers.
        limit: Page size per request (max 1000).  A higher value means
            fewer HTTP round-trips.

    Returns:
        A :class:`great_tables.GT` display table.

    Raises:
        RuntimeError: If no API key is available.
    """
    df = get_all_tickers_df(
        api_key=api_key,
        market=market,
        ticker_type=ticker_type,
        active=active,
        limit=limit,
    )
    logger.info("Built GT table with %d tickers", df.height)
    return (
        GT(df)
        .tab_header(
            title="All Tickers",
            subtitle=f"{df.height} tickers from the Massive API",
        )
        .cols_align(align="center", columns=[cols.MARKET, cols.TYPE, cols.ACTIVE])
        .cols_align(
            align="left",
            columns=[cols.TICKER, cols.NAME, cols.CURRENCY, cols.EXCHANGE],
        )
    )


def _fetch_one_ticker_detail(
    client: RESTClient,
    symbol: str,
) -> TickerDetails | None:
    """Fetch details for a single ticker, returning *None* on failure."""
    try:
        result = client.get_ticker_details(symbol)
    except Exception:
        logger.warning("Failed to fetch details for %s", symbol, exc_info=True)
        return None
    if isinstance(result, TickerDetails):
        logger.debug("Fetched details for %s", symbol)
        return result
    return None


def get_ticker_detail(
    symbol: str,
    *,
    api_key: str | None = None,
) -> TickerDetails | None:
    """Return raw :class:`TickerDetails` for a single ticker, or *None*.

    Useful when callers only need a small piece of the detail payload
    (e.g. the long-form ``name``) and don't want the DataFrame wrapping
    that :func:`get_ticker_details_df` provides.  Network/SDK errors are
    caught and reported via the logger.

    Args:
        symbol: The ticker symbol to look up.
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY`` when
            *None*.

    Returns:
        The :class:`TickerDetails` model on success, or *None* if the
        request failed or the SDK returned an unexpected type.

    Raises:
        RuntimeError: If no API key is available.
    """
    client = get_client(api_key)
    return _fetch_one_ticker_detail(client, symbol)


def get_ticker_details_df(
    tickers: list[str],
    *,
    api_key: str | None = None,
    max_workers: int = 10,
) -> pl.DataFrame:
    """Return ticker-details as a Polars DataFrame (unstyled).

    Used by the decorator pipeline; see :func:`get_ticker_details` for
    the Great Tables-wrapped public variant and full argument
    documentation.
    """
    logger.info(
        "Fetching details for %d tickers (max_workers=%d)",
        len(tickers),
        max_workers,
    )
    client = get_client(api_key)

    # Preserve input order by mapping in a thread pool.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda t: _fetch_one_ticker_detail(client, t), tickers))

    details: list[TickerDetails] = [r for r in results if r is not None]
    logger.info("Retrieved details for %d/%d tickers", len(details), len(tickers))

    # One row per ticker, one column per detail field.
    rows: list[dict[str, str | None]] = []
    for det in details:
        row: dict[str, str | None] = {}
        for attr, label in DETAIL_FIELDS:
            value = getattr(det, attr, None)
            if isinstance(value, float):
                row[label] = f"{value:,.0f}"
            elif isinstance(value, int):
                row[label] = f"{value:,}"
            elif value is not None:
                row[label] = str(value)
            else:
                row[label] = None
        rows.append(row)
    # Declare the schema explicitly: every cell is str-or-None.  Without
    # this, polars infers from the first few rows — if those rows are
    # null for a column it guesses the wrong dtype and chokes when a
    # later row finally has a string value.
    schema = {label: pl.Utf8 for _, label in DETAIL_FIELDS}
    return pl.DataFrame(rows, schema=schema)


def get_ticker_details(
    tickers: list[str],
    *,
    api_key: str | None = None,
    max_workers: int = 10,
) -> GT:
    """Return detailed company information for each ticker.

    The Massive ``get_ticker_details`` endpoint is per-ticker, so this
    function dispatches the requests **concurrently** using a thread
    pool.  The Massive SDK is synchronous but each call is I/O-bound,
    so threading gives a near-linear speedup until the API rate limit
    or the network is saturated.

    Args:
        tickers: Ticker symbols to look up (e.g. ``["AAPL", "MSFT"]``).
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        max_workers: Number of concurrent HTTP requests.  Defaults to
            ``10``.  Tune up for higher API tiers, down to respect rate
            limits.

    Returns:
        A :class:`great_tables.GT` display table with tickers as rows
        and detail fields as columns.  Tickers for which the API
        returned no data are silently dropped.

    Raises:
        RuntimeError: If no API key is available.
    """
    df = get_ticker_details_df(tickers, api_key=api_key, max_workers=max_workers)
    detail_cols = [label for _, label in DETAIL_FIELDS]

    logger.info("Built ticker details GT table")
    return (
        GT(df)
        .tab_header(
            title="Ticker Details",
            subtitle=f"{df.height} tickers from the Massive API",
        )
        .cols_align(align="left", columns=detail_cols)
        .sub_missing(missing_text="\u2014")
    )


# ── Ticker → SIC code index (bundled cache) ──────────────────────────

_TICKER_SIC_RESOURCE = "ticker_sic_codes.json"
_USER_CACHE_DIR = Path(platformdirs.user_cache_dir("bestee"))
_ticker_sic_index: dict[str, str] | None = None


def _sic_key(value: object) -> str:
    """Normalize a SIC code to a comparable string.

    Trims whitespace and drops leading zeros for numeric codes so equivalent
    forms match (``7372`` == ``"7372"``, and ``100`` == ``"0100"``).
    """
    text = str(value if value is not None else "").strip()
    return str(int(text)) if text.isdigit() else text


def _user_ticker_sic_cache() -> Path:
    """Path to the writable cache of ticker->SIC pairs discovered at runtime."""
    return _USER_CACHE_DIR / _TICKER_SIC_RESOURCE


def _load_ticker_sic_index() -> dict[str, str]:
    """Return the ``ticker -> SIC code`` map, cached in memory after first read.

    The bundled snapshot shipped with the package is overlaid with any entries
    discovered at runtime and saved to the writable user cache, so codes learned
    from the related-companies endpoint persist across runs.
    """
    global _ticker_sic_index
    if _ticker_sic_index is None:
        ref = importlib.resources.files(bestee_compute.resources).joinpath(
            _TICKER_SIC_RESOURCE
        )
        index = cast(dict[str, str], json.loads(ref.read_text(encoding="utf-8")))
        user_cache = _user_ticker_sic_cache()
        if user_cache.is_file():
            try:
                index.update(json.loads(user_cache.read_text(encoding="utf-8")))
            except OSError, json.JSONDecodeError:
                logger.warning("Ignoring unreadable SIC cache at %s", user_cache)
        _ticker_sic_index = index
    return _ticker_sic_index


def _remember_sic_codes(new_entries: Mapping[str, str]) -> None:
    """Persist runtime-discovered ``ticker -> SIC`` pairs to the user cache.

    Updates the in-memory index and merges *new_entries* into the writable
    user-cache file (written atomically via a temp file + rename).  Failures are
    logged and swallowed so caching never breaks a lookup.
    """
    if not new_entries:
        return
    _load_ticker_sic_index().update(new_entries)
    path = _user_ticker_sic_cache()
    merged: dict[str, str] = {}
    if path.is_file():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            logger.warning("Ignoring unreadable SIC cache at %s", path)
    merged.update(new_entries)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(merged, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        logger.info("Cached %d new ticker->SIC entries to %s", len(new_entries), path)
    except OSError:
        logger.warning("Failed to write SIC cache at %s", path)


def build_ticker_sic_index(
    *,
    api_key: str | None = None,
    market: str | None = "stocks",
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
    max_workers: int = 20,
) -> dict[str, str]:
    """Fetch every ticker in the universe and map it to its SIC code.

    Returns ``{ticker: sic_code}`` (normalized strings) for every ticker that
    carries a SIC code; those without one (most ETFs and funds) are omitted.
    This makes one details request per ticker -- the universe is several
    thousand symbols -- so it exists to (re)build the bundled cache that
    :func:`get_tickers_by_sic_code` reads, not to run on every call.

    Args:
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY``.
        market: Market filter (default ``"stocks"``).
        ticker_type: Ticker-type filter (default *None*, every type).
        active: Restrict to actively-traded tickers (default *True*).
        limit: Page size when listing the universe (max 1000).
        max_workers: Number of concurrent details requests.

    Returns:
        A ``{ticker: sic_code}`` mapping.

    Raises:
        RuntimeError: If no API key is available.
    """
    universe = [
        t.ticker
        for t in _fetch_tickers(
            api_key=api_key,
            market=market,
            ticker_type=ticker_type,
            active=active,
            limit=limit,
        )
    ]
    client = get_client(api_key)
    logger.info("Building ticker->SIC index for %d tickers", len(universe))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        details = pool.map(lambda s: _fetch_one_ticker_detail(client, s), universe)
    index: dict[str, str] = {}
    for detail in details:
        if detail is None:
            continue
        ticker = detail.ticker
        sic = _sic_key(getattr(detail, "sic_code", None))
        if ticker and sic:
            index[ticker] = sic
    logger.info("Built ticker->SIC index with %d entries", len(index))
    return index


def get_tickers_by_sic_category_name(
    name: str,
    *,
    index: Mapping[str, str] | None = None,
    refresh: bool = False,
    api_key: str | None = None,
) -> list[str]:
    """Return every ticker whose SIC industry title contains *name*.

    *name* is matched case-insensitively as a literal substring against the SEC
    SIC industry titles (which are upper-case), and tickers from **all** matching
    SIC codes are returned -- a keyword such as ``"software"`` maps to several
    codes.  Returns an empty list when no industry title matches.

    Args:
        name: Substring to search for in the SIC industry titles.
        index: A ``{ticker: sic_code}`` map to filter.  Defaults to the bundled
            cache (or a fresh build when *refresh* is *True*).
        refresh: Rebuild the index from the Massive API instead of the cache.
        api_key: Massive API key (only used when *refresh* is *True*).

    Returns:
        A sorted list of the matching ticker symbols.
    """
    titles = get_sic_codes_df()
    matched = titles.filter(
        pl.col("Industry Title")
        .str.to_lowercase()
        .str.contains(name.lower(), literal=True)
    )
    target_codes = {_sic_key(code) for code in matched.get_column("SIC Code").to_list()}
    if not target_codes:
        return []
    if index is None:
        index = (
            build_ticker_sic_index(api_key=api_key)
            if refresh
            else _load_ticker_sic_index()
        )
    return sorted(ticker for ticker, sic in index.items() if sic in target_codes)


def get_tickers_by_sic_code(
    sic_code: str | int,
    *,
    index: Mapping[str, str] | None = None,
    refresh: bool = False,
    api_key: str | None = None,
) -> list[str]:
    """Return every ticker whose company SIC code equals *sic_code*.

    Reads the bundled ``ticker -> SIC code`` cache by default (offline and
    instant).  Pass *index* to filter a custom map, or *refresh=True* to rebuild
    the map live from the Massive API via :func:`build_ticker_sic_index` (slow).

    Args:
        sic_code: The SIC code to match (``"7372"`` or ``7372``).
        index: A ``{ticker: sic_code}`` map to filter.  Defaults to the bundled
            cache (or a fresh build when *refresh* is *True*).
        refresh: Rebuild the index from the Massive API instead of the cache.
        api_key: Massive API key (only used when *refresh* is *True*).

    Returns:
        A sorted list of the matching ticker symbols.
    """
    target = _sic_key(sic_code)
    if index is None:
        index = (
            build_ticker_sic_index(api_key=api_key)
            if refresh
            else _load_ticker_sic_index()
        )
    return sorted(ticker for ticker, sic in index.items() if sic == target)


# ── Related companies ────────────────────────────────────────────────


def get_related_tickers(ticker: str, *, api_key: str | None = None) -> list[str]:
    """Return the tickers Massive reports as related to *ticker*.

    Wraps the ``/related-companies`` endpoint.  Network/SDK errors are caught
    and reported via the logger, returning an empty list.
    """
    client = get_client(api_key)
    try:
        related = cast(list[RelatedCompany], client.get_related_companies(ticker))
    except Exception:
        logger.warning(
            "Failed to fetch related companies for %s", ticker, exc_info=True
        )
        return []
    return [rc.ticker for rc in related if rc.ticker]


def _sic_lookup(
    ticker: str,
    index: Mapping[str, str],
    client: RESTClient,
    discovered: dict[str, str] | None = None,
) -> str:
    """SIC code for *ticker* from *index*, falling back to a live details call.

    A SIC code fetched live (one absent from *index*) is recorded in
    *discovered* when given, so the caller can persist newly-seen tickers.
    """
    if ticker in index:
        return index[ticker]
    detail = _fetch_one_ticker_detail(client, ticker)
    sic = _sic_key(getattr(detail, "sic_code", None)) if detail else ""
    if sic and discovered is not None:
        discovered[ticker] = sic
    return sic


def related_tickers_sharing_sic(
    ticker: str,
    *,
    index: Mapping[str, str] | None = None,
    api_key: str | None = None,
) -> list[str]:
    """Related companies of *ticker* that share its SIC code.

    Reads the seed's and each related company's SIC code from the bundled
    ``ticker -> SIC code`` cache (falling back to a live :class:`TickerDetails`
    lookup for symbols missing from it), then keeps the related tickers whose
    SIC code matches the seed's.

    Args:
        ticker: The seed ticker whose industry peers we compare against.
        index: A ``{ticker: sic_code}`` map; defaults to the bundled cache.
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY``.

    Returns:
        A sorted list of related tickers with the same SIC code as *ticker*
        (empty when the seed's SIC code is unknown).

    Side effects:
        When the default cache is used (no *index* given), SIC codes fetched
        live for tickers absent from the cache -- typically newly-seen symbols
        from the related-companies endpoint -- are written back to the user
        cache so subsequent lookups are offline.
    """
    use_cache = index is None
    idx = _load_ticker_sic_index() if use_cache else index
    client = get_client(api_key)
    discovered: dict[str, str] = {}
    seed_sic = _sic_lookup(ticker, idx, client, discovered)
    if not seed_sic:
        logger.info("No SIC code known for %s; cannot compare peers", ticker)
        return []
    related = get_related_tickers(ticker, api_key=api_key)
    matches = sorted(
        r for r in related if _sic_lookup(r, idx, client, discovered) == seed_sic
    )
    if use_cache:
        _remember_sic_codes(discovered)
    return matches
