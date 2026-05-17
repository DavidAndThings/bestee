"""Scrape SIC codes and descriptions from the SEC website."""

import html
import importlib.resources
import json
import logging
import re
from pathlib import Path

import httpx
import platformdirs
import polars as pl
from great_tables import GT

import bestee.resources

logger = logging.getLogger(__name__)

_SEC_SIC_URL = (
    "https://www.sec.gov/search-filings/"
    "standard-industrial-classification-sic-code-list"
)

# The SEC requires a descriptive User-Agent for automated access.
# See https://www.sec.gov/os/webmaster-faq#code-support
_USER_AGENT = "bestee/0.1.0 (support@bestee.dev)"

# Regex to match a table row with 3 cells: SIC Code, Office, Industry Title.
_ROW_RE = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>\s*(\d+)\s*</td>\s*"  # SIC Code (digits)
    r"<td[^>]*>\s*(.*?)\s*</td>\s*"  # Office
    r"<td[^>]*>\s*(.*?)\s*</td>\s*"  # Industry Title
    r"</tr>",
    re.DOTALL,
)

# User-level cache directory following OS conventions
# (e.g. ~/Library/Caches/bestee on macOS, ~/.cache/bestee on Linux).
_USER_CACHE_DIR = Path(platformdirs.user_cache_dir("bestee"))
_USER_CACHE = _USER_CACHE_DIR / "sic_codes.json"


# ── Internal helpers ─────────────────────────────────────────────────


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _scrape_sic_codes() -> list[dict[str, str]]:
    """Fetch and parse SIC codes from the SEC website.

    Returns:
        A list of dicts with keys ``sic_code``, ``industry_title``, and
        ``office``.

    Raises:
        Exception: On any network or parsing failure.
    """
    logger.info("Fetching SIC codes from %s", _SEC_SIC_URL)

    response = httpx.get(
        _SEC_SIC_URL,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status()

    matches = _ROW_RE.findall(response.text)

    if not matches:
        msg = "No SIC codes found in SEC page — HTML structure may have changed"
        raise ValueError(msg)

    rows: list[dict[str, str]] = []
    for sic_code, office, title in matches:
        rows.append(
            {
                "sic_code": sic_code.strip(),
                "industry_title": _strip_html(title).strip(),
                "office": _strip_html(office).strip(),
            }
        )

    logger.info("Parsed %d SIC codes from SEC page", len(rows))
    return rows


def _save_cache(rows: list[dict[str, str]]) -> None:
    """Persist scraped data to the user cache directory."""
    try:
        _USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _USER_CACHE.write_text(json.dumps(rows, indent=2))
        logger.debug("Saved %d SIC codes to %s", len(rows), _USER_CACHE)
    except OSError:
        logger.warning("Failed to write SIC cache to %s", _USER_CACHE)


def _load_bundled_cache() -> list[dict[str, str]] | None:
    """Load the SIC snapshot bundled with the package via importlib.resources."""
    try:
        ref = importlib.resources.files(bestee.resources).joinpath("sic_codes.json")
        data: list[dict[str, str]] = json.loads(ref.read_text(encoding="utf-8"))
        logger.info("Loaded %d SIC codes from bundled cache", len(data))
        return data
    except OSError, json.JSONDecodeError:
        logger.warning("Failed to read bundled SIC cache")
        return None


def _load_cache() -> list[dict[str, str]] | None:
    """Load SIC data from the user cache, falling back to the bundled cache.

    Returns:
        The cached rows, or *None* if no cache is available.
    """
    # 1. Try user cache (from a previous successful scrape).
    if _USER_CACHE.is_file():
        try:
            data: list[dict[str, str]] = json.loads(
                _USER_CACHE.read_text(encoding="utf-8")
            )
            logger.info(
                "Loaded %d SIC codes from user cache (%s)", len(data), _USER_CACHE
            )
            return data
        except OSError, json.JSONDecodeError:
            logger.warning("Failed to read user cache at %s", _USER_CACHE)

    # 2. Fall back to the bundled snapshot.
    return _load_bundled_cache()


# ── Public API ───────────────────────────────────────────────────────


def get_sic_codes() -> GT:
    """Return the complete SIC code list as a GT table.

    On the first call the data is scraped from the SEC website and
    cached locally.  Subsequent calls use the cache.  If scraping fails
    (network error, rate limit, HTML change, etc.) the function falls
    back to the local cache — either a previous successful scrape or the
    bundled snapshot shipped with the package.

    Returns:
        A :class:`great_tables.GT` table with columns for SIC Code,
        Industry Title, and Office.

    Raises:
        RuntimeError: If scraping fails **and** no cache is available.
    """
    rows: list[dict[str, str]] | None = None

    # 1. Try scraping from the SEC website.
    try:
        rows = _scrape_sic_codes()
        _save_cache(rows)
    except Exception:
        logger.warning(
            "Failed to scrape SIC codes from SEC — falling back to cache",
            exc_info=True,
        )

    # 2. Fall back to cache if scraping failed.
    if rows is None:
        rows = _load_cache()

    # 3. No data at all — raise.
    if rows is None:
        msg = (
            "Could not obtain SIC codes: scraping failed and no local "
            "cache is available"
        )
        raise RuntimeError(msg)

    df = pl.DataFrame(
        [
            {
                "SIC Code": r["sic_code"],
                "Industry Title": r["industry_title"],
                "Office": r["office"],
            }
            for r in rows
        ]
    )

    logger.info("Built SIC codes GT table with %d rows", len(rows))

    return (
        GT(df)
        .tab_header(
            title="SIC Codes",
            subtitle=f"{len(rows)} Standard Industrial Classification codes",
        )
        .cols_align(align="left", columns=["SIC Code", "Industry Title", "Office"])
    )
