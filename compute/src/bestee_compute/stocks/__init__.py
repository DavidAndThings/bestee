"""Stock-focused data and pipeline operations on the Massive API.

Submodules:

* :mod:`bestee_compute.stocks.tickers` — ticker listings and details.
* :mod:`bestee_compute.stocks.financials` — financial-statement metrics.
* :mod:`bestee_compute.stocks.market` — daily OHLCV market snapshots.
* :mod:`bestee_compute.stocks.sic` — SIC industry classifications.
* :mod:`bestee_compute.stocks.models` — :class:`Metric` / :class:`FinancialMetric`
  data models.
* :mod:`bestee_compute.stocks.columns` — shared column-name constants.
* :mod:`bestee_compute.stocks.decorators` — pipeline stages (table decorators).
* :mod:`bestee_compute.stocks.builder` — DSL → decorator-chain builder.
"""

from bestee_compute.stocks import (
    builder,
    columns,
    decorators,
    financials,
    market,
    models,
    sic,
    tickers,
)

__all__ = [
    "builder",
    "columns",
    "decorators",
    "financials",
    "market",
    "models",
    "sic",
    "tickers",
]
