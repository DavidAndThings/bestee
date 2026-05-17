"""Scrape SIC codes and descriptions from the SEC website."""

import logging
import re

import httpx
import polars as pl
from great_tables import GT

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


def get_sic_codes() -> GT:
    """Scrape the complete SIC code list from the SEC website.

    Fetches the `Standard Industrial Classification (SIC) Code List
    <https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list>`_
    page and parses every row of the table into a
    :class:`great_tables.GT` display table.

    Returns:
        A :class:`great_tables.GT` table with columns for SIC Code,
        Industry Title, and Office.

    Raises:
        httpx.HTTPStatusError: If the SEC website returns a non-2xx
            response.
    """
    logger.info("Fetching SIC codes from %s", _SEC_SIC_URL)

    response = httpx.get(
        _SEC_SIC_URL,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status()

    html = response.text
    matches = _ROW_RE.findall(html)

    logger.info("Parsed %d SIC codes from SEC page", len(matches))

    rows: list[dict[str, str]] = []
    for sic_code, office, title in matches:
        rows.append(
            {
                "SIC Code": sic_code.strip(),
                "Industry Title": _strip_html(title).strip(),
                "Office": _strip_html(office).strip(),
            }
        )

    df = pl.DataFrame(rows)

    logger.info("Built SIC codes GT table with %d rows", len(rows))

    return (
        GT(df)
        .tab_header(
            title="SIC Codes",
            subtitle=f"{len(rows)} Standard Industrial Classification codes",
        )
        .cols_align(align="left", columns=["SIC Code", "Industry Title", "Office"])
    )


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)
