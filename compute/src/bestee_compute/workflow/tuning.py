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

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

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
# Clustering
# ----------------------------------------------------------------------------
class ClusteringRequest(BaseModel):
    """User inputs for auto-tuned spectral clustering.

    The universe and window, plus the inclusive cluster-count bounds within
    which the best ``k`` is selected by silhouette.
    """

    tickers: Sequence[str] = Field(min_length=2)
    start_date: str
    end_date: str
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


def optimize_clustering(
    request: ClusteringRequest, *, panel: pl.DataFrame | None = None
) -> ClusteringTuning:
    """Sweep the residualization/affinity parameters; keep the partition whose
    mean silhouette is highest.

    The cluster *count* is already chosen inside
    :func:`clustering.run_spectral_clustering` (best ``k`` by silhouette); this
    tunes what feeds it -- the rolling PCA window, the z-score window, the number
    of removed components, and the correlation-to-affinity mapping.
    """
    if panel is None:
        panel = _prepare_panel(request.tickers, request.start_date, request.end_date)
    n_tickers = len(request.tickers)
    best: ClusteringTuning | None = None
    for window, norm, n_comp, metric in itertools.product(
        _RESIDUAL_WINDOWS, _NORMALIZATION_WINDOWS, _N_COMPONENTS, _SIMILARITY_METRICS
    ):
        if n_comp > min(window, n_tickers):
            continue
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=panel,
            residualization_method="pca",
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
    """User inputs for auto-tuned per-asset regime detection."""

    tickers: Sequence[str] = Field(min_length=1)
    start_date: str
    end_date: str


@dataclass
class RegimeTuning:
    """Best regime fit found, with the parameters that produced it."""

    labels: dict[str, int]
    results: Mapping[str, regimes.RegimeResult]
    residualization_window: int
    normalization_window: int
    hmm_lag: int
    mean_bic: float


def optimize_regime(
    request: RegimeRequest, *, panel: pl.DataFrame | None = None
) -> RegimeTuning:
    """Sweep the residualization window and AR lag; keep the fit with the lowest
    mean BIC across assets.

    The HMM state count is BIC-selected per asset inside
    :func:`regimes.run_regime_analysis` (``regime_auto_select_states``); this
    tunes the residual window the features are built on and the VAR lag. Mean
    BIC across assets is a heuristic for the best global window/lag.
    """
    if panel is None:
        panel = _prepare_panel(request.tickers, request.start_date, request.end_date)
    best: RegimeTuning | None = None
    for window, lag in itertools.product(_REGIME_WINDOWS, _REGIME_LAGS):
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            injected_panel=panel,
            residualization_method="pca",
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
    logger.info(
        "regime: mean BIC %.1f (window=%d, lag=%d, assets=%d)",
        best.mean_bic,
        best.residualization_window,
        best.hmm_lag,
        len(best.results),
    )
    return best


# ----------------------------------------------------------------------------
# Fama-French factor model
# ----------------------------------------------------------------------------
class FamaFrenchRequest(BaseModel):
    """User inputs for an auto-tuned Fama-French fit: the universe and the
    estimation window."""

    tickers: Sequence[str] = Field(min_length=1)
    start_date: str
    end_date: str


@dataclass
class FamaFrenchTuning:
    """Best factor model found, with the per-ticker fits that produced it."""

    factors_to_use: int
    results: Mapping[str, fama.FamaFrenchResult]
    mean_adjusted_r_squared: float


def optimize_fama_french(
    request: FamaFrenchRequest,
    *,
    variables: Mapping[str, pl.DataFrame] | None = None,
) -> FamaFrenchTuning:
    """Pick the factor count (FF3 / Carhart-4 / FF5 / FF6) that best explains the
    cross-section by mean adjusted R-squared.

    Adjusted R-squared penalizes extra factors, so the winner is the most
    parsimonious model that still fits -- the proper "remaining parameter" to
    optimize. The regression-ready variables are built once and reused.
    """
    if variables is None:
        config = AnalysisConfig(
            tickers=list(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
        )
        variables = fama.get_variables(config)
    best: FamaFrenchTuning | None = None
    for factors in _FACTOR_COUNTS:
        specification = fama.FamaFrenchSpecification(
            start_date=request.start_date,
            end_date=request.end_date,
            factors_to_use=factors,
        )
        results = fama._fit(variables, specification)
        if not results:
            continue
        mean_adj = float(np.mean([r.adj_r_squared for r in results.values()]))
        if best is None or mean_adj > best.mean_adjusted_r_squared:
            best = FamaFrenchTuning(
                factors_to_use=factors,
                results=results,
                mean_adjusted_r_squared=mean_adj,
            )
    if best is None:
        raise ValueError("No Fama-French model could be fit on the given window.")
    logger.info(
        "fama-french: FF%d, mean adj R^2 %.3f (%d tickers)",
        best.factors_to_use,
        best.mean_adjusted_r_squared,
        len(best.results),
    )
    return best


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
