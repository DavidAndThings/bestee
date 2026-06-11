"""Shared column-name constants used across bestee tables.

Centralizing these strings prevents typos and makes it easy to rename a
column in exactly one place when the schema evolves.
"""

# ── Common across most tables ────────────────────────────────────────
TICKER = "Ticker"

# ── All-tickers listing (see bestee_compute.stocks.tickers.get_all_tickers) ──
# Shorter display labels than the equivalents below — the listing view
# trades detail for compactness.
EXCHANGE = "Exchange"
ACTIVE = "Active"

# ── Ticker details (see bestee_compute.stocks.tickers.DETAIL_FIELDS) ────────
NAME = "Name"
DESCRIPTION = "Description"
TYPE = "Type"
MARKET = "Market"
LOCALE = "Locale"
PRIMARY_EXCHANGE = "Primary Exchange"
CURRENCY = "Currency"
CIK = "CIK"
COMPOSITE_FIGI = "Composite FIGI"
SHARE_CLASS_FIGI = "Share Class FIGI"
SIC_CODE = "SIC Code"
SIC_DESCRIPTION = "SIC Description"
MARKET_CAP = "Market Cap"
SHARES_OUTSTANDING = "Shares Outstanding"
WEIGHTED_SHARES_OUTSTANDING = "Weighted Shares Outstanding"
TOTAL_EMPLOYEES = "Total Employees"
LIST_DATE = "List Date"
HOMEPAGE_URL = "Homepage URL"
PHONE_NUMBER = "Phone Number"
TICKER_ROOT = "Ticker Root"
