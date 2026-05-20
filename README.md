# bestee

A Python toolkit for building financial-metrics comparison tables from the
[Massive](https://github.com/polygon-io) market-data API. Tickers, market
snapshots, financial statements, and ratios are returned as
[`great_tables.GT`](https://posit-dev.github.io/great-tables/) display tables
that can be rendered to HTML.

It also ships a small DSL for composing pipelines from a sequence of commands —
useful for building peer-group tables with computed metrics in a single pass.

## Installation

```sh
uv sync
```

Set your Massive API key in a `.env` file at the project root:

```
MASSIVE_API_KEY=...
```

## Library usage

```python
from bestee import FinancialMetric, Metric, build_financials_table

table = build_financials_table(
    tickers=["AAPL", "MSFT", "GOOG"],
    metrics=[
        FinancialMetric(Metric.REVENUE, fiscal_year=2024, fiscal_quarter=4),
        FinancialMetric(Metric.PE_RATIO, fiscal_year=2024, fiscal_quarter=4),
    ],
)
table.show()
```

## DSL pipeline

The `decorator_builder` assembles a chain of `TableDecorator`s from a sequence
of commands. Commands are processed in three logical phases (base table →
financial metrics → computed metrics), so ordering within a phase is
irrelevant.

```text
STOCKS
SAME_SIC_CATEGORY_AS AAPL
FINANCIAL_METRIC REVENUE 2024 4
COMPUTED_METRIC GROSS_MARGIN (FINANCIAL_METRIC GROSS_PROFIT 2024 4) / (FINANCIAL_METRIC REVENUE 2024 4)
```

```python
from bestee.decorators import decorator_builder

commands = [line.split() for line in script.splitlines() if line.strip()]
pipeline = decorator_builder(commands)
pipeline.build().show()
```

Stress-test inputs are bundled in `src/bestee/resources/`.

## CLI

```sh
uv run bestee
```

Prints the full ticker listing.

## Development

```sh
uv sync                 # install deps (incl. dev)
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run ty check         # type check
```

CI runs the same checks on push/PR (`.github/workflows/ci.yml`).
