"""Cluster securities by residual co-movement via spectral clustering.

All knobs live on :class:`AnalysisConfig` (the ``clustering_*`` fields plus the
shared ``residualization_*`` ones); :func:`run_spectral_clustering` is the entry
point. The data layer is reached through the free functions in
:mod:`bestee_compute.workflow.tools`.
"""

import logging
from collections.abc import Mapping

import numpy as np
import polars as pl
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow.tools import (
    AnalysisConfig,
    TickerClass,
    compute_similarity_matrix,
    get_normalized_ols_residuals,
    get_normalized_pca_residuals,
    get_tickers,
)

logger = logging.getLogger(__name__)


def run_spectral_clustering(config: AnalysisConfig) -> Mapping[str, int]:
    """Cluster securities by residual co-movement via spectral clustering.

    Residualizes ``tickers`` (rolling PCA or OLS per ``residualization_method``),
    turns the residual correlations into a non-negative affinity, and partitions
    that graph with spectral clustering -- which handles the elongated,
    non-convex structure correlation graphs often have. The cluster count is
    auto-selected in ``[clustering_min_num_clusters, clustering_max_num_clusters]``
    by the mean silhouette.

    Only the **liquid** tickers (median dollar volume >= ``min_dollar_volume``)
    are clustered; illiquid micro-caps have near-pure-noise residuals that would
    blur the graph. With ``clustering_assign_illiquid`` the whole covered set is
    residualized together (one shared factor space) and each illiquid name is
    folded back in by assigning it to the liquid cluster its residual most
    correlates with.

    Returns:
        A ``ticker -> cluster id`` mapping (0-indexed). Without
        ``clustering_assign_illiquid`` only the liquid names are returned.

    Raises:
        ValueError: If there are too few liquid names for
            ``clustering_min_num_clusters``, or no candidate ``k`` yields a
            valid (>= 2 group) partition.
    """
    liquid = get_tickers(config, "liquid")
    illiquid = (
        get_tickers(config, "illiquid") if config.clustering_assign_illiquid else []
    )
    if not illiquid:
        # Cluster the liquid names alone (residualized over the liquid set).
        affinity = _affinity(config, _get_residuals(config, "liquid"))
        labels = _spectral_labels(config, affinity)
        return {t: int(c) for t, c in zip(liquid, labels, strict=True)}

    # Residualize the whole covered set so liquid + illiquid share one factor
    # space; cluster the liquid columns, assign the illiquid ones.
    residuals = _get_residuals(config, "all")
    order = get_tickers(config, "all")
    liquid_labels = _spectral_labels(
        config, _affinity(config, _residual_subframe(residuals, order, liquid))
    )
    labels = {t: int(c) for t, c in zip(liquid, liquid_labels, strict=True)}

    clean = residuals.drop_nulls()
    liquid_features = _residual_matrix(clean, order, liquid)
    illiquid_features = _residual_matrix(clean, order, illiquid)
    labels.update(
        _assign_illiquid(liquid_features, illiquid_features, liquid_labels, illiquid)
    )
    logger.info(
        "Spectral: clustered %d liquid + assigned %d illiquid into %d clusters",
        len(liquid),
        len(illiquid),
        len(set(labels.values())),
    )
    return labels


def _affinity(config: AnalysisConfig, residuals: pl.DataFrame) -> np.ndarray:
    """Symmetric, non-negative affinity in ``[0, 1]`` from a residual frame."""
    affinity = compute_similarity_matrix(
        residuals, config.clustering_similarity_metric
    ).to_numpy()
    # Symmetrize (guard float asymmetry) and clamp to a valid similarity.
    return np.clip(0.5 * (affinity + affinity.T), 0.0, 1.0)


def _spectral_labels(config: AnalysisConfig, affinity: np.ndarray) -> np.ndarray:
    """Best spectral partition over the configured cluster range by silhouette."""
    lo = config.clustering_min_num_clusters
    hi_max = config.clustering_max_num_clusters
    distance = 1.0 - affinity
    np.fill_diagonal(distance, 0.0)
    hi = min(hi_max, affinity.shape[0] - 1)
    if hi < lo:
        raise ValueError(
            f"Need at least {lo + 1} liquid names to find "
            f"{lo}..{hi_max} clusters; got {affinity.shape[0]}."
        )
    best_labels: np.ndarray | None = None
    best_k, best_score = 0, -np.inf
    for k in range(lo, hi + 1):
        labels = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=0
        ).fit_predict(affinity)
        if len(np.unique(labels)) < 2:
            continue  # degenerate split; silhouette is undefined
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        if score > best_score:
            best_labels, best_k, best_score = labels, k, score
    if best_labels is None:
        raise ValueError("Spectral clustering produced no valid partition.")
    logger.info(
        "Spectral clustering: selected k=%d in [%d, %d] (silhouette %.3f)",
        best_k,
        lo,
        hi,
        best_score,
    )
    return best_labels


def _residual_subframe(
    residuals: pl.DataFrame, order: list[str], subset: list[str]
) -> pl.DataFrame:
    """``Timestamp`` + the residual columns for *subset* (*order* indexes them)."""
    columns = [c for c in residuals.columns if c != OHLCHeader.TIMESTAMP]
    position = {ticker: index for index, ticker in enumerate(order)}
    return residuals.select(
        OHLCHeader.TIMESTAMP, *[columns[position[t]] for t in subset]
    )


def _residual_matrix(
    residuals: pl.DataFrame, order: list[str], subset: list[str]
) -> np.ndarray:
    """Residual series for *subset* as a ``(n_obs, len(subset))`` array."""
    return (
        _residual_subframe(residuals, order, subset)
        .drop(OHLCHeader.TIMESTAMP)
        .to_numpy()
    )


def _assign_illiquid(
    liquid_features: np.ndarray,
    illiquid_features: np.ndarray,
    liquid_labels: np.ndarray,
    illiquid: list[str],
) -> dict[str, int]:
    """Assign each illiquid name to the most-correlated liquid cluster.

    A cluster's signature is the mean residual series of its liquid members; an
    illiquid name joins the cluster its own residual series correlates with most.
    """
    cluster_ids = sorted({int(c) for c in liquid_labels})
    centroids = np.column_stack(
        [liquid_features[:, liquid_labels == c].mean(axis=1) for c in cluster_ids]
    )
    a = illiquid_features - illiquid_features.mean(axis=0)
    b = centroids - centroids.mean(axis=0)
    denom = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    denom[denom == 0.0] = 1.0
    correlation = (a.T @ b) / denom
    nearest = correlation.argmax(axis=1)
    return {illiquid[j]: cluster_ids[nearest[j]] for j in range(len(illiquid))}


def _get_residuals(
    config: AnalysisConfig,
    ticker_class: TickerClass = "all",
) -> pl.DataFrame:
    """Normalized residuals for *ticker_class* per ``residualization_method``."""
    match config.residualization_method:
        case "pca":
            return get_normalized_pca_residuals(config, ticker_class)
        case "ols":
            return get_normalized_ols_residuals(config, ticker_class)
        case _:
            raise ValueError(
                f"Unknown residualization method: {config.residualization_method}"
            )
