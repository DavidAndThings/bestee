"""ETF data providers.

Each provider submodule (one per issuer) exposes the same shape:

* ``get_holdings(ticker) -> polars.DataFrame``
* ``get_holdings_table(ticker) -> great_tables.GT``

Schemas differ across providers — each issuer publishes a different set
of columns (e.g. VanEck has Market Value; SPDR has SEDOL + Sector).  Use
the provider directly when you need a specific schema, or use the
:mod:`bestee_compute.etf.holdings` dispatcher to pick the right one from a
ticker automatically::

    from bestee_compute.etf import get_holdings, guess_etf_issuer
    guess_etf_issuer("SPY")     # -> "SPDR"
    get_holdings("SPY")         # routes to bestee_compute.etf.spdr.get_holdings
"""

from bestee_compute.etf import (
    holdings,
    invesco,
    ishares,
    issuer,
    normalize,
    pca,
    roundhill,
    spdr,
    vaneck,
)
from bestee_compute.etf.holdings import (
    IssuerNotSupportedError,
    get_holdings,
    get_holdings_table,
    supported_issuers,
)
from bestee_compute.etf.issuer import guess_etf_issuer
from bestee_compute.etf.normalize import get_holdings_normalized, normalize_holdings
from bestee_compute.etf.pca import ETFPCAResult, etf_pca

__all__ = [
    "ETFPCAResult",
    "IssuerNotSupportedError",
    "etf_pca",
    "get_holdings",
    "get_holdings_normalized",
    "get_holdings_table",
    "guess_etf_issuer",
    "holdings",
    "invesco",
    "ishares",
    "issuer",
    "normalize",
    "normalize_holdings",
    "pca",
    "roundhill",
    "spdr",
    "supported_issuers",
    "vaneck",
]
