"""Stock-focused data and pipeline operations on the Massive API.

Submodules:

* :mod:`bestee.stocks.tickers` — ticker listings and details.
* :mod:`bestee.stocks.financials` — financial-statement metrics.
* :mod:`bestee.stocks.market` — daily OHLCV market snapshots.
* :mod:`bestee.stocks.sic` — SIC industry classifications.
* :mod:`bestee.stocks.models` — :class:`Metric` / :class:`FinancialMetric`
  data models.
* :mod:`bestee.stocks.columns` — shared column-name constants.
* :mod:`bestee.stocks.decorators` — DSL-driven table-pipeline stages.
"""

from bestee.stocks import columns, decorators, financials, market, models, sic, tickers

__all__ = [
    "columns",
    "decorators",
    "financials",
    "market",
    "models",
    "sic",
    "tickers",
]
