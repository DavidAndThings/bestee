"""Auto-tuning layer over the workflow analyses.

Each analysis (clustering, regime labels, Fama-French, RRG) gets:

1. a small pydantic ``*Request`` model that exposes **only** the inputs a user
   must supply -- the universe and the date window, plus the few genuine choices
   an analysis can't default (e.g. the RRG reference);
2. an ``optimize_*`` function that takes that request, fetches the price data
   once, then sweeps the *remaining* parameters (the residualization windows,
   the cluster affinity, the HMM lag, the factor count, the RRG smoothing, ...)
   and returns the best-scoring combination together with the analysis output.

Only the :class:`AnalysisConfig` fields each analysis actually reads are set; the
rest keep their defaults. The search grids below are the tuning space, not user
input. Pass ``panel=`` (or ``variables=`` for Fama-French) to reuse pre-fetched
data and avoid the network.
"""

import datetime as dt
import hashlib
import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics import silhouette_score

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import clustering, fama, regimes, rrg
from bestee_compute.workflow.tools import (
    AnalysisConfig,
    ReferenceType,
    SimilarityMetric,
    compute_similarity_matrix,
    get_clean_panel,
    get_normalized_pca_residuals,
    get_ohlc,
    get_tickers,
)

logger = logging.getLogger(__name__)

# --- Search grids (the parameters each optimizer tunes; not user input) -----
_RESIDUAL_WINDOWS = (20, 40, 60)
_NORMALIZATION_WINDOWS = (20, 40, 60)
_N_COMPONENTS = (1, 2)
_SIMILARITY_METRICS: tuple[SimilarityMetric, ...] = ("aff", "corr")

_REGIME_WINDOWS = (40, 60)
_REGIME_LAGS = (1, 2)

_FACTOR_COUNTS = (3, 4, 5, 6)

_RRG_NORMALIZATION_WINDOWS = (10, 20, 30)
_RRG_MOMENTUM_LOOKBACKS = (10, 20)
_RRG_SMOOTHING_SPANS: tuple[int | None, ...] = (None, 3, 5)


# ----------------------------------------------------------------------------
# Shared data preparation
# ----------------------------------------------------------------------------
def _prepare_panel(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    benchmark_ticker: str | None = None,
) -> pl.DataFrame:
    """Fetch a gap-filled Close+Volume panel once, plus the benchmark's Close.

    Building the panel up front lets the sweep reuse one dataset (injected into
    every candidate :class:`AnalysisConfig`) instead of refetching per trial.
    """
    template = AnalysisConfig(
        tickers=list(tickers), start_date=start_date, end_date=end_date
    )
    panel = get_clean_panel(template, OHLCHeader.CLOSE, OHLCHeader.VOLUME)
    if benchmark_ticker is not None:
        close = OHLCHeader.CLOSE
        benchmark = get_ohlc(benchmark_ticker, template).select(
            OHLCHeader.TIMESTAMP,
            pl.col(close).alias(f"{benchmark_ticker}_{close}"),
        )
        panel = panel.join(benchmark, on=OHLCHeader.TIMESTAMP, how="inner")
    return panel


# ----------------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------------
def _serialize_frame(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert a Polars DataFrame to a JSON-ready list of dicts.

    Temporal columns (Datetime, Date, Duration, Time) are cast to strings so
    the result is JSON-serializable without a custom encoder.
    """
    casts = [
        pl.col(name).cast(pl.String)
        for name, dtype in zip(df.columns, df.dtypes)
        if dtype.base_type() in (pl.Datetime, pl.Date, pl.Duration, pl.Time)
    ]
    return (df.with_columns(casts) if casts else df).to_dicts()


def frame_to_table(frame: pl.DataFrame) -> dict[str, Any]:
    """A JSON-safe table view of *frame*: ordered column names + record rows.

    Used by the results API to serve an analysis-specific ``aspect`` -- the
    ``columns`` give the canonical order, ``rows`` are the records (temporal
    columns stringified by :func:`_serialize_frame`).
    """
    return {"columns": frame.columns, "rows": _serialize_frame(frame)}


def _unpivot_by_ticker(
    frame: pl.DataFrame, suffix: str, value_name: str
) -> pl.DataFrame:
    """Wide ``{ticker}_{suffix}`` frame -> long ``[timestamp, ticker, value_name]``.

    The per-ticker value columns are melted into rows; the ``_{suffix}`` tag is
    stripped back off the column name to recover the bare ticker symbol.
    """
    value_columns = [c for c in frame.columns if c != OHLCHeader.TIMESTAMP]
    return (
        frame.unpivot(
            index=OHLCHeader.TIMESTAMP,
            on=value_columns,
            variable_name="ticker",
            value_name=value_name,
        )
        .with_columns(pl.col("ticker").str.replace(f"_{suffix}$", ""))
        .rename({OHLCHeader.TIMESTAMP: "timestamp"})
    )


def result_id(analysis: str, request: BaseModel) -> str:
    """Stable ``{analysis}_{sha256[:16]}`` key for a (analysis_type, request) pair.

    The same request always produces the same ID, making the storage directory
    a request-keyed cache -- re-running an identical analysis overwrites the
    previous result.
    """
    payload = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return f"{analysis}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
class ClusteringRequest(BaseModel):
    """User inputs for auto-tuned spectral clustering.

    The universe and window, plus the inclusive cluster-count bounds within
    which the best ``k`` is selected by silhouette. With ``benchmark_ticker``
    set, names are residualized against it via the rolling market-model (OLS)
    regression; left unset, residualization falls back to rolling PCA over the
    basket itself, so a benchmark is optional.
    """

    tickers: Sequence[str] = Field(min_length=2)
    start_date: str
    end_date: str
    benchmark_ticker: str | None = None
    min_num_clusters: int = Field(default=2, ge=2)
    max_num_clusters: int = Field(default=50, ge=2)

    @model_validator(mode="after")
    def _check_cluster_bounds(self) -> Self:
        if self.min_num_clusters > self.max_num_clusters:
            raise ValueError("min_num_clusters cannot exceed max_num_clusters")
        return self


@dataclass
class ClusteringTuning:
    """Best clustering found, with the parameters that produced it."""

    labels: dict[str, int]
    residualization_window: int
    normalization_window: int
    n_components: int
    similarity_metric: SimilarityMetric
    silhouette: float
    n_clusters: int

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable representation (all primitive fields)."""
        return {
            "labels": self.labels,
            "residualization_window": self.residualization_window,
            "normalization_window": self.normalization_window,
            "n_components": self.n_components,
            "similarity_metric": self.similarity_metric,
            "silhouette": self.silhouette,
            "n_clusters": self.n_clusters,
        }

    def cluster_label_table(self) -> pl.DataFrame:
        """``cluster_label`` aspect: one ``[ticker, cluster_label]`` row per name."""
        return pl.DataFrame(
            {
                "ticker": list(self.labels.keys()),
                "cluster_label": list(self.labels.values()),
            },
            schema={"ticker": pl.Utf8, "cluster_label": pl.Int64},
        ).sort("ticker")

    def save(self, path: Path) -> None:
        """Persist to *path* as a single ``metadata.json`` file."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(
            json.dumps(self.to_dict(), separators=(",", ":"))
        )

    @classmethod
    def load(cls, path: Path) -> ClusteringTuning:
        """Reconstruct from a directory written by :meth:`save`."""
        return cls(**json.loads((path / "metadata.json").read_text()))


def optimize_clustering(
    request: ClusteringRequest, *, panel: pl.DataFrame | None = None
) -> ClusteringTuning:
    """Sweep the residualization/affinity parameters; keep the partition whose
    mean silhouette is highest.

    The cluster *count* is already chosen inside
    :func:`clustering.run_spectral_clustering` (best ``k`` by silhouette); this
    tunes what feeds it -- the rolling residualization window, the z-score
    window, the correlation-to-affinity mapping, and (PCA only) the number of
    removed components.

    Residualization follows ``request.benchmark_ticker``: a market-model (OLS)
    fit against the benchmark when one is given, else rolling PCA over the
    basket (see :func:`clustering._get_residuals`). The removed-component sweep
    is PCA-specific, so it collapses to a single trial under OLS.
    """
    if panel is None:
        panel = _prepare_panel(
            request.tickers,
            request.start_date,
            request.end_date,
            benchmark_ticker=request.benchmark_ticker,
        )
    n_tickers = len(request.tickers)
    # n_components only feeds PCA; under an OLS benchmark it is inert, so one
    # value suffices and the sweep doesn't recompute identical residuals.
    n_component_grid = (1,) if request.benchmark_ticker else _N_COMPONENTS
    best: ClusteringTuning | None = None
    for window, norm, n_comp, metric in itertools.product(
        _RESIDUAL_WINDOWS, _NORMALIZATION_WINDOWS, n_component_grid, _SIMILARITY_METRICS
    ):
        if n_comp > min(window, n_tickers):
            continue
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=panel,
            benchmark_ticker=request.benchmark_ticker,
            residualization_window=window,
            normalization_window=norm,
            n_components=n_comp,
            clustering_similarity_metric=metric,
            clustering_min_num_clusters=request.min_num_clusters,
            clustering_max_num_clusters=request.max_num_clusters,
        )
        try:
            labels = clustering.run_spectral_clustering(config)
            score = _clustering_silhouette(config, labels, metric)
        except ValueError as exc:
            logger.debug("clustering trial %s skipped: %s", config, exc)
            continue
        if best is None or score > best.silhouette:
            best = ClusteringTuning(
                labels=dict(labels),
                residualization_window=window,
                normalization_window=norm,
                n_components=n_comp,
                similarity_metric=metric,
                silhouette=score,
                n_clusters=len(set(labels.values())),
            )
    if best is None:
        raise ValueError("No clustering configuration produced a valid partition.")
    logger.info(
        "clustering: silhouette %.3f (window=%d, norm=%d, n=%d, metric=%s, k=%d)",
        best.silhouette,
        best.residualization_window,
        best.normalization_window,
        best.n_components,
        best.similarity_metric,
        best.n_clusters,
    )
    return best


def _clustering_silhouette(
    config: AnalysisConfig, labels: Mapping[str, int], metric: SimilarityMetric
) -> float:
    """Mean silhouette of *labels* on the residual affinity (same as clustering).

    Recomputes the affinity the runner used so partitions from different
    parameter trials are scored on a common ``[-1, 1]`` scale.
    """
    residuals = get_normalized_pca_residuals(config, "liquid")
    order = get_tickers(config, "liquid")
    label_vector = [labels[t] for t in order]
    if len(set(label_vector)) < 2:
        return -1.0
    affinity = compute_similarity_matrix(residuals, metric).to_numpy()
    affinity = np.clip(0.5 * (affinity + affinity.T), 0.0, 1.0)
    distance = 1.0 - affinity
    np.fill_diagonal(distance, 0.0)
    return float(silhouette_score(distance, label_vector, metric="precomputed"))


# ----------------------------------------------------------------------------
# Regime labels
# ----------------------------------------------------------------------------
class RegimeRequest(BaseModel):
    """User inputs for auto-tuned per-asset regime detection.

    With ``benchmark_ticker`` set, each asset is residualized against it via the
    rolling market-model (OLS) regression; left unset, the residualization falls
    back to rolling PCA over the basket itself, so a benchmark is optional.

    ``oos_dates`` are optional out-of-sample dates (each strictly after
    ``end_date``). When given, the best-fit model is held **fixed** and used to
    classify the regime at each date; those labels are carried in the result
    alongside the full in-sample regime path.
    """

    tickers: Sequence[str] = Field(min_length=1)
    start_date: str
    end_date: str
    benchmark_ticker: str | None = None
    oos_dates: Sequence[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _oos_dates_after_end(self) -> Self:
        end = dt.date.fromisoformat(self.end_date[:10])
        for oos_date in self.oos_dates:
            if dt.date.fromisoformat(oos_date[:10]) <= end:
                raise ValueError(
                    f"oos_date {oos_date!r} must be strictly after end_date "
                    f"{self.end_date!r}"
                )
        return self


@dataclass
class RegimeTuning:
    """Best regime fit found, with the parameters that produced it.

    ``oos_regimes`` is the wide ``[Timestamp, {ticker}_Regime,
    {ticker}_Regime_Prob]`` frame from the fixed-model out-of-sample nowcast
    (empty when the request carried no ``oos_dates``).
    """

    labels: dict[str, int]
    results: Mapping[str, regimes.RegimeResult]
    residualization_window: int
    normalization_window: int
    hmm_lag: int
    mean_bic: float
    oos_regimes: pl.DataFrame = field(default_factory=pl.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable representation including per-asset decoded states."""
        return {
            "labels": self.labels,
            "residualization_window": self.residualization_window,
            "normalization_window": self.normalization_window,
            "hmm_lag": self.hmm_lag,
            "mean_bic": self.mean_bic,
            "results": {
                ticker: {
                    "current_regime": result.current_regime(),
                    "n_states": result.n_states,
                    "log_likelihood": result.log_likelihood,
                    "bic": result.bic(),
                    "regime_summary": result.regime_summary().to_dicts(),
                    "states": _serialize_frame(result.states),
                }
                for ticker, result in self.results.items()
            },
            "oos_regimes": _serialize_frame(self.oos_regimes),
        }

    def regime_label_table(self) -> pl.DataFrame:
        """``regime_label`` aspect: long ``[timestamp, ticker, regime_label]``.

        Stacks every asset's full in-sample decoded path with the fixed-model
        out-of-sample labels (when ``oos_dates`` were requested), so one table
        carries both. Labels are the canonical 0 = calmest ordering, comparable
        in and out of sample because the out-of-sample nowcast reuses the same
        fitted model.
        """
        frames: list[pl.DataFrame] = []
        for ticker, result in self.results.items():
            frames.append(
                result.states.select(
                    pl.col(OHLCHeader.TIMESTAMP).alias("timestamp"),
                    pl.lit(ticker).alias("ticker"),
                    pl.col("Regime").alias("regime_label"),
                )
            )
        if self.oos_regimes.width > 0:
            for column in self.oos_regimes.columns:
                if not column.endswith("_Regime"):
                    continue
                ticker = column.removesuffix("_Regime")
                frames.append(
                    self.oos_regimes.select(
                        pl.col(OHLCHeader.TIMESTAMP).alias("timestamp"),
                        pl.lit(ticker).alias("ticker"),
                        pl.col(column).alias("regime_label"),
                    ).drop_nulls()
                )
        if not frames:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "ticker": pl.Utf8,
                    "regime_label": pl.Int64,
                }
            )
        return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "timestamp"])

    def save(self, path: Path) -> None:
        """Persist to *path*: scalar params + numpy model params in
        ``metadata.json``, per-asset state DataFrames as
        ``states_{ticker}.parquet``."""
        path.mkdir(parents=True, exist_ok=True)
        if self.oos_regimes.width > 0:
            self.oos_regimes.write_parquet(path / "oos_regimes.parquet")
        ticker_meta: dict[str, Any] = {}
        for ticker, result in self.results.items():
            result.states.write_parquet(path / f"states_{ticker}.parquet")
            ticker_meta[ticker] = {
                "feature_names": result.feature_names,
                "n_states": result.n_states,
                "lag": result.lag,
                "log_likelihood": result.log_likelihood,
                "n_params": result.n_params,
                "transition_matrix": result.transition_matrix.tolist(),
                "start_prob": result.start_prob.tolist(),
                "coef": result.coef.tolist(),
                "covars": result.covars.tolist(),
            }
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "labels": self.labels,
                    "residualization_window": self.residualization_window,
                    "normalization_window": self.normalization_window,
                    "hmm_lag": self.hmm_lag,
                    "mean_bic": self.mean_bic,
                    "results": ticker_meta,
                },
                separators=(",", ":"),
            )
        )

    @classmethod
    def load(cls, path: Path) -> RegimeTuning:
        """Reconstruct from a directory written by :meth:`save`."""
        meta = json.loads((path / "metadata.json").read_text())
        ticker_results = {
            ticker: regimes.RegimeResult(
                ticker=ticker,
                states=pl.read_parquet(path / f"states_{ticker}.parquet"),
                transition_matrix=np.array(data["transition_matrix"]),
                start_prob=np.array(data["start_prob"]),
                coef=np.array(data["coef"]),
                covars=np.array(data["covars"]),
                feature_names=data["feature_names"],
                n_states=data["n_states"],
                lag=data["lag"],
                log_likelihood=data["log_likelihood"],
                n_params=data["n_params"],
            )
            for ticker, data in meta["results"].items()
        }
        oos_path = path / "oos_regimes.parquet"
        oos_regimes = pl.read_parquet(oos_path) if oos_path.exists() else pl.DataFrame()
        return cls(
            labels=meta["labels"],
            results=ticker_results,
            residualization_window=meta["residualization_window"],
            normalization_window=meta["normalization_window"],
            hmm_lag=meta["hmm_lag"],
            mean_bic=meta["mean_bic"],
            oos_regimes=oos_regimes,
        )


def optimize_regime(
    request: RegimeRequest, *, panel: pl.DataFrame | None = None
) -> RegimeTuning:
    """Sweep the residualization window and AR lag; keep the fit with the lowest
    mean BIC across assets.

    The HMM state count is BIC-selected per asset inside
    :func:`regimes.run_regime_analysis` (``regime_auto_select_states``); this
    tunes the residual window the features are built on and the VAR lag. Mean
    BIC across assets is a heuristic for the best global window/lag.

    Residualization follows ``request.benchmark_ticker``: a market-model (OLS)
    fit against the benchmark when one is given, else rolling PCA over the
    basket (see :func:`regimes._residual_frames`).

    When ``request.oos_dates`` are given, the panel is fetched through the last
    of them; the sweep still fits **in-sample only** (the panel is sliced at
    ``end_date``), and the winning model is then held fixed to nowcast the
    out-of-sample regimes (:func:`regimes.predict_oos_regimes`), stored on the
    result's ``oos_regimes``.
    """
    horizon = max([request.end_date, *request.oos_dates])
    if panel is None:
        panel = _prepare_panel(
            request.tickers,
            request.start_date,
            horizon,
            benchmark_ticker=request.benchmark_ticker,
        )
    # The sweep fits in-sample only; trim any out-of-sample tail the panel
    # carries so the BIC and decoded path stop at end_date.
    in_sample = panel.filter(
        pl.col(OHLCHeader.TIMESTAMP).cast(pl.Date)
        <= dt.date.fromisoformat(request.end_date[:10])
    )
    best: RegimeTuning | None = None
    for window, lag in itertools.product(_REGIME_WINDOWS, _REGIME_LAGS):
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=in_sample,
            benchmark_ticker=request.benchmark_ticker,
            residualization_window=window,
            normalization_window=window,
            regime_hmm_lag=lag,
            regime_auto_select_states=True,
        )
        results = regimes.run_regime_analysis(config)
        if not results:
            continue
        mean_bic = float(np.mean([result.bic() for result in results.values()]))
        if best is None or mean_bic < best.mean_bic:
            best = RegimeTuning(
                labels={t: r.current_regime() for t, r in results.items()},
                results=results,
                residualization_window=window,
                normalization_window=window,
                hmm_lag=lag,
                mean_bic=mean_bic,
            )
    if best is None:
        raise ValueError("No regime configuration fit any asset on this window.")
    if request.oos_dates:
        # Re-fit the winning configuration on the full panel; predict_oos_regimes
        # holds the fit fixed and classifies each out-of-sample date causally.
        oos_config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=panel,
            benchmark_ticker=request.benchmark_ticker,
            residualization_window=best.residualization_window,
            normalization_window=best.normalization_window,
            regime_hmm_lag=best.hmm_lag,
            regime_auto_select_states=True,
        )
        best.oos_regimes = regimes.predict_oos_regimes(
            oos_config, list(request.oos_dates)
        )
    logger.info(
        "regime: mean BIC %.1f (window=%d, lag=%d, assets=%d, oos dates=%d)",
        best.mean_bic,
        best.residualization_window,
        best.hmm_lag,
        len(best.results),
        len(request.oos_dates),
    )
    return best


# ----------------------------------------------------------------------------
# Fama-French factor model
# ----------------------------------------------------------------------------
class DateInterval(BaseModel):
    """One estimation window ``[start_date, end_date]`` (inclusive ISO YYYY-MM-DD)."""

    start_date: str
    end_date: str


class FamaFrenchRequest(BaseModel):
    """User inputs for an auto-tuned Fama-French fit.

    The model is fit over each ``intervals`` window and the factor count
    (FF3 / Carhart-4 / FF5 / FF6) that maximises mean adjusted R\u00b2 across all
    intervals is chosen.  If ``oos_date`` is supplied (must be strictly after
    every interval's ``end_date``), out-of-sample residuals for that date are
    computed and included in the result.
    """

    tickers: Sequence[str] = Field(min_length=1)
    intervals: Sequence[DateInterval] = Field(min_length=1)
    oos_dates: Sequence[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _oos_dates_after_all_intervals(self) -> Self:
        for oos_date in self.oos_dates:
            oos = dt.date.fromisoformat(oos_date[:10])
            for interval in self.intervals:
                end = dt.date.fromisoformat(interval.end_date[:10])
                if oos <= end:
                    raise ValueError(
                        f"oos_date {oos_date!r} must be strictly after every "
                        f"interval's end_date "
                        f"(got interval ending {interval.end_date!r})"
                    )
        return self


_TRADING_DAYS_PER_YEAR = 252


@dataclass
class FamaFrenchTuning:
    """Best factor model found across all requested estimation intervals.

    ``specifications`` holds one :class:`fama.FamaFrenchSpecification` per
    requested interval, all using the chosen ``factors_to_use``.  ``results``
    maps each specification's name to its per-ticker
    :class:`fama.FamaFrenchResult`.  ``oos_residuals`` carries the long
    ``[Date, Ticker, Specification, Residual, ZScore, PValue]`` frame.
    """

    factors_to_use: int
    specifications: Sequence[fama.FamaFrenchSpecification]
    results: Mapping[str, Mapping[str, fama.FamaFrenchResult]]
    mean_adjusted_r_squared: float
    oos_residuals: pl.DataFrame

    def summarize_results(self) -> pl.DataFrame:
        """Headline per-(specification, ticker) statistics as a flat frame.

        Flattens the nested ``results`` mapping into one row per fitted
        ``(Specification, Ticker)`` pair, sorted by specification then ticker.
        Useful for a quick cross-sectional review without navigating the
        nested ``results`` dict.

        Columns:

        * ``Specification`` -- window + factor-count label (e.g.
          ``FF6_2020-01-01_2022-12-31``).
        * ``Ticker``
        * ``Alpha`` -- daily intercept; the excess return unexplained by
          the factors.
        * ``AnnualizedAlpha`` -- ``Alpha × 252`` for easier interpretation.
        * ``AlphaTStat`` -- t-statistic on alpha; values beyond ±2 suggest
          statistical significance.
        * ``MktBeta`` -- loading on ``Mkt-RF``, the dominant risk factor.
        * ``AdjR2`` -- adjusted R² of the in-sample fit.
        * ``ResidualStd`` -- daily idiosyncratic volatility (std of
          residuals); used to standardize OOS residuals into z-scores.
        * ``N`` -- number of in-window trading days used to fit the model.
        """
        rows = [
            {
                "Specification": spec_name,
                "Ticker": ticker,
                "Alpha": result.alpha,
                "AnnualizedAlpha": result.alpha * _TRADING_DAYS_PER_YEAR,
                "AlphaTStat": result.t_stats.get("alpha", float("nan")),
                "MktBeta": result.betas.get("Mkt-RF", float("nan")),
                "AdjR2": result.adj_r_squared,
                "ResidualStd": result.residual_std,
                "N": result.n_observations,
            }
            for spec_name, ticker_results in self.results.items()
            for ticker, result in ticker_results.items()
        ]
        return pl.DataFrame(
            rows,
            schema={
                "Specification": pl.Utf8,
                "Ticker": pl.Utf8,
                "Alpha": pl.Float64,
                "AnnualizedAlpha": pl.Float64,
                "AlphaTStat": pl.Float64,
                "MktBeta": pl.Float64,
                "AdjR2": pl.Float64,
                "ResidualStd": pl.Float64,
                "N": pl.Int64,
            },
        ).sort(["Specification", "Ticker"])

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable representation."""
        return {
            "factors_to_use": self.factors_to_use,
            "specifications": [s.model_dump() for s in self.specifications],
            "mean_adjusted_r_squared": self.mean_adjusted_r_squared,
            "results": {
                spec_name: {
                    ticker: r.model_dump(mode="json")
                    for ticker, r in spec_results.items()
                }
                for spec_name, spec_results in self.results.items()
            },
            "oos_residuals": self.oos_residuals.to_dicts(),
        }

    def ff_residuals_table(self) -> pl.DataFrame:
        """``ff_residuals`` aspect: long ``[date, ticker, specification, residual,
        p_value]``.

        ``p_value`` is the two-sided tail probability of the standardized
        out-of-sample residual; it is **local to each specification** (hence the
        ``specification`` column) and must not be compared across them.
        """
        return self.oos_residuals.select(
            pl.col("Date").alias("date"),
            pl.col("Ticker").alias("ticker"),
            pl.col("Specification").alias("specification"),
            pl.col("Residual").alias("residual"),
            pl.col("PValue").alias("p_value"),
        ).sort(["specification", "date", "ticker"])

    def save(self, path: Path) -> None:
        """Persist to *path*: pydantic models in ``metadata.json``,
        OOS residuals as ``oos_residuals.parquet``."""
        path.mkdir(parents=True, exist_ok=True)
        self.oos_residuals.write_parquet(path / "oos_residuals.parquet")
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "factors_to_use": self.factors_to_use,
                    "mean_adjusted_r_squared": self.mean_adjusted_r_squared,
                    "specifications": [s.model_dump() for s in self.specifications],
                    "results": {
                        spec_name: {
                            ticker: r.model_dump(mode="json")
                            for ticker, r in spec_results.items()
                        }
                        for spec_name, spec_results in self.results.items()
                    },
                },
                separators=(",", ":"),
            )
        )

    @classmethod
    def load(cls, path: Path) -> FamaFrenchTuning:
        """Reconstruct from a directory written by :meth:`save`."""
        meta = json.loads((path / "metadata.json").read_text())
        return cls(
            factors_to_use=meta["factors_to_use"],
            mean_adjusted_r_squared=meta["mean_adjusted_r_squared"],
            specifications=[
                fama.FamaFrenchSpecification(**s) for s in meta["specifications"]
            ],
            results={
                spec_name: {
                    ticker: fama.FamaFrenchResult(**r)
                    for ticker, r in spec_results.items()
                }
                for spec_name, spec_results in meta["results"].items()
            },
            oos_residuals=pl.read_parquet(path / "oos_residuals.parquet"),
        )


def optimize_fama_french(
    request: FamaFrenchRequest,
    *,
    variables: Mapping[str, pl.DataFrame] | None = None,
) -> FamaFrenchTuning:
    """Pick the Fama-French factor count (FF3/4/5/6) that maximises mean
    adjusted R\u00b2 across all requested estimation intervals.

    One :class:`fama.FamaFrenchSpecification` is created per interval with the
    winning factor count.  If ``request.oos_date`` is set, out-of-sample
    residuals for that date are computed across all fitted specifications and
    attached to the result.
    """
    config = AnalysisConfig(
        tickers=list(request.tickers),
        start_date=min(i.start_date for i in request.intervals),
        end_date=max(
            max(request.oos_dates),
            max(i.end_date for i in request.intervals),
        ),
    )
    if variables is None:
        variables = fama.get_variables(config)

    best_factors: int | None = None
    best_specs: list[fama.FamaFrenchSpecification] = []
    best_interval_results: dict[str, dict[str, fama.FamaFrenchResult]] = {}
    best_mean_adj = -1.0
    for factors in _FACTOR_COUNTS:
        specs = [
            fama.FamaFrenchSpecification(
                start_date=interval.start_date,
                end_date=interval.end_date,
                factors_to_use=factors,
            )
            for interval in request.intervals
        ]
        interval_results: dict[str, dict[str, fama.FamaFrenchResult]] = {}
        adj_r_squareds: list[float] = []
        for spec in specs:
            spec_results = fama._fit(variables, spec)
            if not spec_results:
                continue
            interval_results[spec.name] = spec_results
            adj_r_squareds.extend(r.adj_r_squared for r in spec_results.values())
        if not adj_r_squareds:
            continue
        mean_adj = float(np.mean(adj_r_squareds))
        if mean_adj > best_mean_adj:
            best_factors = factors
            best_specs = specs
            best_interval_results = interval_results
            best_mean_adj = mean_adj

    if best_factors is None:
        raise ValueError(
            "No Fama-French model could be fit on any interval: none of the "
            f"{len(request.tickers)} ticker(s) had enough overlapping daily "
            "price and factor observations in any window (each fit needs more "
            "rows than factors). Check the tickers have daily history over the "
            "interval(s)."
        )

    oos_residuals = fama.get_residuals(
        config, request.oos_dates, best_specs, variables=variables
    )

    logger.info(
        "fama-french: FF%d, mean adj R^2 %.3f "
        "(%d interval(s), %d oos date(s), %d tickers)",
        best_factors,
        best_mean_adj,
        len(best_specs),
        len(request.oos_dates),
        len(request.tickers),
    )
    return FamaFrenchTuning(
        factors_to_use=best_factors,
        specifications=best_specs,
        results=best_interval_results,
        mean_adjusted_r_squared=best_mean_adj,
        oos_residuals=oos_residuals,
    )


# ----------------------------------------------------------------------------
# Relative-rotation graph
# ----------------------------------------------------------------------------
class RRGRequest(BaseModel):
    """User inputs for an auto-tuned RRG: the universe, window, and reference."""

    tickers: Sequence[str] = Field(min_length=1)
    start_date: str
    end_date: str
    reference_type: ReferenceType = "mean"
    benchmark_ticker: str | None = None

    @model_validator(mode="after")
    def _require_benchmark_for_ticker_reference(self) -> Self:
        if self.reference_type == "ticker" and self.benchmark_ticker is None:
            raise ValueError(
                "benchmark_ticker is required when reference_type is 'ticker'"
            )
        return self


@dataclass
class RRGTuning:
    """Best RRG smoothing found, with the resulting RS / momentum frames."""

    normalization_window: int
    momentum_lookback: int
    smoothing_span: int | None
    signal_to_noise: float
    relative_strength: pl.DataFrame
    relative_momentum: pl.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable representation."""
        return {
            "normalization_window": self.normalization_window,
            "momentum_lookback": self.momentum_lookback,
            "smoothing_span": self.smoothing_span,
            "signal_to_noise": self.signal_to_noise,
            "relative_strength": _serialize_frame(self.relative_strength),
            "relative_momentum": _serialize_frame(self.relative_momentum),
        }

    def coordinates_table(self) -> pl.DataFrame:
        """``coordinates`` aspect: long ``[timestamp, ticker, relative_strength,
        relative_momentum]`` -- one RRG point per (date, ticker).

        The wide per-ticker RS-Ratio and momentum frames are melted to long form
        and inner-joined, so a row exists only where both coordinates do.
        """
        strength = _unpivot_by_ticker(
            self.relative_strength,
            rrg._RELATIVE_STRENGTH_COLUMN_NAME,
            "relative_strength",
        )
        momentum = _unpivot_by_ticker(
            self.relative_momentum,
            rrg._RELATIVE_MOMENTUM_COLUMN_NAME,
            "relative_momentum",
        )
        return strength.join(momentum, on=["timestamp", "ticker"], how="inner").sort(
            ["ticker", "timestamp"]
        )

    def save(self, path: Path) -> None:
        """Persist to *path*: scalars in ``metadata.json``, DataFrames as
        ``relative_strength.parquet`` and ``relative_momentum.parquet``."""
        path.mkdir(parents=True, exist_ok=True)
        self.relative_strength.write_parquet(path / "relative_strength.parquet")
        self.relative_momentum.write_parquet(path / "relative_momentum.parquet")
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "normalization_window": self.normalization_window,
                    "momentum_lookback": self.momentum_lookback,
                    "smoothing_span": self.smoothing_span,
                    "signal_to_noise": self.signal_to_noise,
                },
                separators=(",", ":"),
            )
        )

    @classmethod
    def load(cls, path: Path) -> RRGTuning:
        """Reconstruct from a directory written by :meth:`save`."""
        meta = json.loads((path / "metadata.json").read_text())
        return cls(
            normalization_window=meta["normalization_window"],
            momentum_lookback=meta["momentum_lookback"],
            smoothing_span=meta["smoothing_span"],
            signal_to_noise=meta["signal_to_noise"],
            relative_strength=pl.read_parquet(path / "relative_strength.parquet"),
            relative_momentum=pl.read_parquet(path / "relative_momentum.parquet"),
        )


def optimize_rrg(
    request: RRGRequest, *, panel: pl.DataFrame | None = None
) -> RRGTuning:
    """Sweep the normalization window, momentum lookback, and smoothing span;
    keep the combination with the best signal-to-noise.

    "Smoothness without compromising the signal" is scored as the ratio of each
    series' movement (its standard deviation) to its jaggedness (the standard
    deviation of its second difference), averaged over the RS-Ratio and the
    momentum across all assets -- high when the curves move meaningfully yet
    turn smoothly.
    """
    if panel is None:
        panel = _prepare_panel(
            request.tickers,
            request.start_date,
            request.end_date,
            benchmark_ticker=request.benchmark_ticker,
        )
    best: RRGTuning | None = None
    for norm, lookback, span in itertools.product(
        _RRG_NORMALIZATION_WINDOWS, _RRG_MOMENTUM_LOOKBACKS, _RRG_SMOOTHING_SPANS
    ):
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=panel,
            benchmark_ticker=request.benchmark_ticker,
            rrg_reference_type=request.reference_type,
            normalization_window=norm,
            rrg_momentum_lookback=lookback,
            rrg_smoothing_span=span,
        )
        strength = rrg.normalized_relative_strength(config)
        momentum = rrg.normalized_relative_momentum(config)
        if strength.height < 4 or momentum.height < 4:
            continue
        score = 0.5 * (_signal_to_noise(strength) + _signal_to_noise(momentum))
        if best is None or score > best.signal_to_noise:
            best = RRGTuning(
                normalization_window=norm,
                momentum_lookback=lookback,
                smoothing_span=span,
                signal_to_noise=score,
                relative_strength=strength,
                relative_momentum=momentum,
            )
    if best is None:
        raise ValueError("No RRG configuration produced enough points to score.")
    logger.info(
        "rrg: SNR %.2f (norm=%d, lookback=%d, span=%s)",
        best.signal_to_noise,
        best.normalization_window,
        best.momentum_lookback,
        best.smoothing_span,
    )
    return best


def _signal_to_noise(frame: pl.DataFrame) -> float:
    """Mean over asset columns of std(series) / std(second difference).

    Movement (signal) over jaggedness (noise): larger means the line travels
    while turning smoothly. Flat or too-short columns are ignored.
    """
    ratios: list[float] = []
    for column in frame.columns:
        if column == OHLCHeader.TIMESTAMP:
            continue
        values = frame[column].to_numpy()
        values = values[~np.isnan(values)]
        if values.size < 4:
            continue
        signal = float(np.std(values))
        noise = float(np.std(np.diff(values, n=2)))
        if noise > 1e-12:
            ratios.append(signal / noise)
    return float(np.mean(ratios)) if ratios else 0.0
