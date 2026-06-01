"""Data models for bestee financial metrics."""

import datetime as dt
from dataclasses import dataclass
from enum import Enum, StrEnum


class StatementType(Enum):
    """The type of financial statement a metric belongs to."""

    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    RATIOS = "ratios"


class Metric(Enum):
    """Catalog of available financial metrics from the Massive API.

    Each member's value is a ``(StatementType, field_name, display_label)``
    tuple that maps to the corresponding SDK model attribute.
    """

    # ── Income Statement ─────────────────────────────────────────────
    REVENUE = (StatementType.INCOME_STATEMENT, "revenue", "Revenue")
    COST_OF_REVENUE = (
        StatementType.INCOME_STATEMENT,
        "cost_of_revenue",
        "Cost of Revenue",
    )
    GROSS_PROFIT = (StatementType.INCOME_STATEMENT, "gross_profit", "Gross Profit")
    OPERATING_INCOME = (
        StatementType.INCOME_STATEMENT,
        "operating_income",
        "Operating Income",
    )
    EBITDA = (StatementType.INCOME_STATEMENT, "ebitda", "EBITDA")
    NET_INCOME = (
        StatementType.INCOME_STATEMENT,
        "net_income_loss_attributable_common_shareholders",
        "Net Income",
    )
    BASIC_EPS = (
        StatementType.INCOME_STATEMENT,
        "basic_earnings_per_share",
        "Basic EPS",
    )
    DILUTED_EPS = (
        StatementType.INCOME_STATEMENT,
        "diluted_earnings_per_share",
        "Diluted EPS",
    )
    INTEREST_EXPENSE = (
        StatementType.INCOME_STATEMENT,
        "interest_expense",
        "Interest Expense",
    )
    INCOME_TAX = (StatementType.INCOME_STATEMENT, "income_taxes", "Income Taxes")

    # ── Balance Sheet ────────────────────────────────────────────────
    TOTAL_ASSETS = (StatementType.BALANCE_SHEET, "total_assets", "Total Assets")
    TOTAL_LIABILITIES = (
        StatementType.BALANCE_SHEET,
        "total_liabilities",
        "Total Liabilities",
    )
    TOTAL_EQUITY = (StatementType.BALANCE_SHEET, "total_equity", "Total Equity")
    CASH_AND_EQUIVALENTS = (
        StatementType.BALANCE_SHEET,
        "cash_and_equivalents",
        "Cash & Equivalents",
    )
    INVENTORIES = (StatementType.BALANCE_SHEET, "inventories", "Inventories")
    RECEIVABLES = (StatementType.BALANCE_SHEET, "receivables", "Receivables")
    LONG_TERM_DEBT = (
        StatementType.BALANCE_SHEET,
        "long_term_debt_and_capital_lease_obligations",
        "Long-Term Debt",
    )
    RETAINED_EARNINGS = (
        StatementType.BALANCE_SHEET,
        "retained_earnings_deficit",
        "Retained Earnings",
    )

    # ── Cash Flow Statement ──────────────────────────────────────────
    OPERATING_CASH_FLOW = (
        StatementType.CASH_FLOW,
        "net_cash_from_operating_activities",
        "Operating Cash Flow",
    )
    INVESTING_CASH_FLOW = (
        StatementType.CASH_FLOW,
        "net_cash_from_investing_activities",
        "Investing Cash Flow",
    )
    FINANCING_CASH_FLOW = (
        StatementType.CASH_FLOW,
        "net_cash_from_financing_activities",
        "Financing Cash Flow",
    )
    CAPEX = (
        StatementType.CASH_FLOW,
        "purchase_of_property_plant_and_equipment",
        "Capital Expenditures",
    )
    DIVIDENDS_PAID = (StatementType.CASH_FLOW, "dividends", "Dividends Paid")

    # ── Ratios ───────────────────────────────────────────────────────
    MARKET_CAP = (StatementType.RATIOS, "market_cap", "Market Cap")
    PE_RATIO = (StatementType.RATIOS, "price_to_earnings", "P/E Ratio")
    PB_RATIO = (StatementType.RATIOS, "price_to_book", "P/B Ratio")
    PS_RATIO = (StatementType.RATIOS, "price_to_sales", "P/S Ratio")
    DIVIDEND_YIELD = (StatementType.RATIOS, "dividend_yield", "Dividend Yield")
    ROA = (StatementType.RATIOS, "return_on_assets", "Return on Assets")
    ROE = (StatementType.RATIOS, "return_on_equity", "Return on Equity")
    DEBT_TO_EQUITY = (StatementType.RATIOS, "debt_to_equity", "Debt / Equity")
    EV_TO_EBITDA = (StatementType.RATIOS, "ev_to_ebitda", "EV / EBITDA")
    FREE_CASH_FLOW = (StatementType.RATIOS, "free_cash_flow", "Free Cash Flow")
    ENTERPRISE_VALUE = (
        StatementType.RATIOS,
        "enterprise_value",
        "Enterprise Value",
    )
    PRICE = (StatementType.RATIOS, "price", "Price")

    # ── Convenience properties ───────────────────────────────────────

    @property
    def statement(self) -> StatementType:
        """The financial statement this metric belongs to."""
        return self.value[0]

    @property
    def field(self) -> str:
        """The attribute name on the SDK model."""
        return self.value[1]

    @property
    def base_label(self) -> str:
        """Human-readable name without a period suffix."""
        return self.value[2]


@dataclass(frozen=True)
class FinancialMetric:
    """A concrete metric request bound to a specific fiscal period.

    Combines a :class:`Metric` (what to fetch) with a fiscal year and
    quarter (when to fetch it).

    Usage::

        from bestee.stocks.models import FinancialMetric, Metric

        metrics = [
            FinancialMetric(Metric.REVENUE, fiscal_year=2024, fiscal_quarter=4),
            FinancialMetric(Metric.REVENUE, fiscal_year=2023, fiscal_quarter=4),
            FinancialMetric(Metric.PE_RATIO, fiscal_year=2024, fiscal_quarter=4),
        ]

    Note:
        Ratio metrics (``Metric.PE_RATIO``, ``Metric.MARKET_CAP``, etc.)
        are point-in-time market data.  The Massive ratios endpoint does
        not support fiscal-period filtering, so the **most recent**
        available ratio is always returned regardless of the period
        specified here.  The period is still included in the column label
        for consistency.
    """

    metric: Metric
    fiscal_year: int
    fiscal_quarter: int

    # ── Delegated properties ─────────────────────────────────────────

    @property
    def statement(self) -> StatementType:
        """The financial statement this metric belongs to."""
        return self.metric.statement

    @property
    def field(self) -> str:
        """The attribute name on the SDK model."""
        return self.metric.field

    @property
    def label(self) -> str:
        """Column header including the fiscal period."""
        return f"{self.metric.base_label} (FY{self.fiscal_year} Q{self.fiscal_quarter})"


class TimeSeriesSpan(Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TimeSeriesName(Enum):
    OPEN_PRICE = "open_price"
    HIGH_PRICE = "high_price"
    LOW_PRICE = "low_price"
    CLOSE_PRICE = "close_price"


@dataclass(frozen=True)
class TimeSeriesDef:
    name: TimeSeriesName
    span: TimeSeriesSpan
    start: str | dt.date | dt.datetime
    end: str | dt.date | dt.datetime
    multiplier: int = 1
    tag: str | None = None
    """Optional symbolic name for chain-walking disambiguation.

    The decorator pipeline routes time-series requests by
    ``TimeSeriesDef`` equality, so two definitions that share OHLC
    parameters but represent semantically different series (e.g. a raw
    ``ts1`` and a derived ``ts3 = ts2 - ts1`` whose shape was copied
    from ``ts1``) would otherwise collide.  Setting ``tag`` to a
    unique string makes them unequal.
    """


class CommandHeader(StrEnum):
    """All DSL command keywords.

    Subclassing :class:`StrEnum` makes each member ``==`` the same
    string value, so ``match c[0]: case CommandHeader.X:`` works
    directly against the tokenised input.
    """

    ASSET_SCOPE = "ASSET_SCOPE"
    SAME_SIC_CATEGORY_AS = "SAME_SIC_CATEGORY_AS"
    PICK_TICKERS = "PICK_TICKERS"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    COMPUTED_METRIC = "COMPUTED_METRIC"
    TIME_SERIES = "TIME_SERIES"
    TIME_SERIES_DERIVED = "TIME_SERIES_DERIVED"
    TIME_SERIES_METRIC = "TIME_SERIES_METRIC"
    STOCHASTIC_OSCILLATOR = "STOCHASTIC_OSCILLATOR"
    RSI = "RSI"
    SMA = "SMA"
