import polars as pl

from bestee_compute.workflow.tools import (
    AnalysisConfig,
    OHLCHeader,
    get_ohlc_column_for_benchmark,
    get_ohlc_column_for_tickers,
    get_tickers,
    z_score_normalize,
)

_REFERENCE_COLUMN_NAME = "reference"
_RELATIVE_STRENGTH_COLUMN_NAME = "rel_strength"
_RELATIVE_MOMENTUM_COLUMN_NAME = "rel_momentum"


def add_reference(config: AnalysisConfig) -> pl.DataFrame:
    match config.rrg_reference_type:
        case "mean":
            wide = get_ohlc_column_for_tickers(config)
            constituents = [
                column for column in wide.columns if column != OHLCHeader.TIMESTAMP
            ]
            # Equal-weight, share-price-invariant basket. Each constituent
            # is indexed to its first bar and the reference is a rebalanced
            # equal-weight index built from the mean of per-bar log returns,
            # so RS = constituent / reference weights every name equally
            # rather than by share price (as a raw-price mean would).
            reference = (
                pl.mean_horizontal(
                    [pl.col(column).log().diff() for column in constituents]
                )
                .fill_null(0.0)
                .cum_sum()
                .exp()
                .alias(_REFERENCE_COLUMN_NAME)
            )
            rebased = [
                (pl.col(column) / pl.col(column).first()).alias(column)
                for column in constituents
            ]
            return wide.with_columns(*rebased, reference)
        case "ticker":
            benchmark_ticker = config.benchmark_ticker
            if benchmark_ticker is None:
                raise ValueError(
                    "reference_ticker is required when reference_type is 'ticker'"
                )
            reference = get_ohlc_column_for_benchmark(config).rename(
                {f"{benchmark_ticker}_{config.ohlc_column}": _REFERENCE_COLUMN_NAME}
            )
            return get_ohlc_column_for_tickers(config).join(
                reference, on=OHLCHeader.TIMESTAMP, how="inner"
            )


def compute_relative_strength(config: AnalysisConfig) -> pl.DataFrame:
    def relative_strength(ticker: str) -> pl.Expr:
        expr = pl.col(f"{ticker}_{config.ohlc_column}") / pl.col(_REFERENCE_COLUMN_NAME)
        # Smooth the relative-strength line (not the raw prices) ahead of
        # the z-score, so both the RS-Ratio and the momentum derived from
        # it are denoised. A larger span is smoother but lags more.
        if config.rrg_smoothing_span is not None:
            expr = expr.ewm_mean(span=config.rrg_smoothing_span)
        return expr.alias(f"{ticker}_{_RELATIVE_STRENGTH_COLUMN_NAME}")

    return add_reference(config).select(
        OHLCHeader.TIMESTAMP,
        *[relative_strength(ticker) for ticker in get_tickers(config)],
    )


def normalized_relative_strength(config: AnalysisConfig) -> pl.DataFrame:
    normalization_window = config.normalization_window
    if normalization_window is None:
        raise ValueError(
            "AnalysisConfig.normalization_window must be set for this analysis"
        )
    return (
        compute_relative_strength(config)
        .select(
            OHLCHeader.TIMESTAMP,
            *z_score_normalize(
                normalization_window,
                [
                    f"{ticker}_{_RELATIVE_STRENGTH_COLUMN_NAME}"
                    for ticker in get_tickers(config)
                ],
            ),
        )
        .drop_nulls()
    )


def compute_relative_momentum(config: AnalysisConfig) -> pl.DataFrame:
    momentum_lookback = config.rrg_momentum_lookback
    if momentum_lookback is None:
        raise ValueError(
            "AnalysisConfig.rrg_momentum_lookback must be set for this analysis"
        )
    return (
        compute_relative_strength(config)
        .select(
            OHLCHeader.TIMESTAMP,
            *[
                _rolling_slope(
                    pl.col(f"{ticker}_{_RELATIVE_STRENGTH_COLUMN_NAME}"),
                    momentum_lookback,
                ).alias(f"{ticker}_{_RELATIVE_MOMENTUM_COLUMN_NAME}")
                for ticker in get_tickers(config)
            ],
        )
        .drop_nulls()
    )


def normalized_relative_momentum(config: AnalysisConfig) -> pl.DataFrame:
    normalization_window = config.normalization_window
    if normalization_window is None:
        raise ValueError(
            "AnalysisConfig.normalization_window must be set for this analysis"
        )
    return (
        compute_relative_momentum(config)
        .select(
            OHLCHeader.TIMESTAMP,
            *z_score_normalize(
                normalization_window,
                [
                    f"{ticker}_{_RELATIVE_MOMENTUM_COLUMN_NAME}"
                    for ticker in get_tickers(config)
                ],
            ),
        )
        .drop_nulls()
    )


def _rolling_slope(value: pl.Expr, window: int) -> pl.Expr:
    # Slope of an ordinary-least-squares fit of `value` against the bar index
    # over a trailing window: slope = cov(t, value) / var(t). This is a smooth
    # (low-pass) rate of change, unlike diff(), which is high-pass and
    # amplifies week-to-week noise. A constant shift of the index per window
    # leaves the slope unchanged, so the global row index can be used as t.
    index = pl.int_range(0, pl.len()).cast(pl.Float64)
    mean_t = index.rolling_mean(window_size=window)
    mean_v = value.rolling_mean(window_size=window)
    mean_tv = (index * value).rolling_mean(window_size=window)
    mean_tt = (index * index).rolling_mean(window_size=window)
    return (mean_tv - mean_t * mean_v) / (mean_tt - mean_t * mean_t)
