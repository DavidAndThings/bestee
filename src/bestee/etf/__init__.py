"""ETF data providers.

Each provider submodule (one per issuer) exposes the same shape:

* ``get_holdings(ticker) -> polars.DataFrame``
* ``get_holdings_table(ticker) -> great_tables.GT``

Schemas differ across providers — each issuer publishes a different set
of columns (e.g. VanEck has Market Value; SPDR has SEDOL + Sector).  Use
the provider directly when you need a specific schema, or use the
:mod:`bestee.etf.holdings` dispatcher to pick the right one from a
ticker automatically::

    from bestee.etf import get_holdings, guess_etf_issuer
    guess_etf_issuer("SPY")     # -> "SPDR"
    get_holdings("SPY")         # routes to bestee.etf.spdr.get_holdings
"""

from bestee.etf import holdings, ishares, issuer, spdr, vaneck
from bestee.etf.holdings import (
    IssuerNotSupportedError,
    get_holdings,
    get_holdings_table,
    supported_issuers,
)
from bestee.etf.issuer import guess_etf_issuer

__all__ = [
    "IssuerNotSupportedError",
    "get_holdings",
    "get_holdings_table",
    "guess_etf_issuer",
    "holdings",
    "ishares",
    "issuer",
    "spdr",
    "supported_issuers",
    "vaneck",
]
