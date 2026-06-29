"""Principal-component analysis of ETFs based on their weighted holdings.

Each ETF is represented as a point in "holdings space" — one dimension
per unique constituent across the input ETFs, valued at the weight
that constituent has in that ETF (zero where it isn't held).  Running
PCA on this matrix surfaces the directions in holdings space that
explain the most variance across the input funds, which is a useful
way to see what makes them look similar or different (e.g. an "AI
exposure" axis, a "geographic concentration" axis).

This module joins constituents on uppercased ticker symbol (with a
name-based fallback for holdings the issuer publishes without a
ticker, e.g. iShares bond funds).  It works best on US-equity ETFs;
funds whose constituents have no usable cross-issuer identifier will
overlap poorly with the rest of the input set.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from bestee_compute.etf.normalize import get_holdings_normalized


def _join_key(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``(holding, weight)`` rows ready to feed into the matrix.

    Uses the canonical normalize schema:  ``ticker`` is the join key
    when present; otherwise the row falls back to a normalized
    ``name`` so bond ETFs (which iShares publishes without tickers)
    still contribute.  Identifiers are uppercased + stripped;  rows
    with no usable identifier or no weight are dropped;  same-key
    duplicates are summed (some ETFs split a position across multiple
    rows for tax lots or derivatives).
    """
    return (
        frame.select(
            pl.coalesce(
                pl.col("ticker").cast(pl.Utf8).str.strip_chars(),
                pl.col("name").cast(pl.Utf8).str.strip_chars(),
            )
            .str.to_uppercase()
            .alias("holding"),
            pl.col("weight").cast(pl.Float64).alias("weight"),
        )
        .filter(
            pl.col("holding").is_not_null()
            & (pl.col("holding") != "")
            & pl.col("weight").is_not_null()
        )
        .group_by("holding")
        .agg(pl.col("weight").sum())
    )


@dataclass(frozen=True)
class ETFPCAResult:
    """Output of :func:`etf_pca`.

    The decomposition is built from the *centered* weight matrix
    (each holding's mean weight across the input ETFs subtracted), so
    projections describe how each ETF deviates from the average
    portfolio of the input set rather than from the zero vector.

    Attributes:
        tickers: Input ETF tickers in row order of :attr:`weights`.
        holdings: Constituent identifiers in column order of
            :attr:`weights`.
        weights: ``(n_etfs, n_holdings)`` raw (uncentered) weight
            matrix.  Zero where an ETF doesn't hold a constituent.
        components: ``(n_components, n_holdings)`` — principal
            components in holdings space.  Each row is a unit vector;
            the magnitude of an entry is how strongly that holding
            loads onto the component.
        explained_variance: ``(n_components,)`` — variance captured
            by each component.
        explained_variance_ratio: ``(n_components,)`` — same, scaled
            so the full decomposition sums to 1.0.
        projections: ``(n_etfs, n_components)`` — each ETF's
            coordinates in the principal-component basis (the
            "scores").
    """

    tickers: list[str]
    holdings: list[str]
    weights: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    projections: np.ndarray

    def projections_frame(self) -> pl.DataFrame:
        """One row per ETF, one column per principal component."""
        n_comp = self.projections.shape[1]
        data: dict[str, list[object]] = {"ticker": list(self.tickers)}
        for i in range(n_comp):
            data[f"PC{i + 1}"] = self.projections[:, i].tolist()
        return pl.DataFrame(data)

    def components_frame(self) -> pl.DataFrame:
        """One row per ``(component, holding)`` loading, sorted within
        each component by absolute loading (largest first)."""
        n_comp, _ = self.components.shape
        comp_idx = np.repeat(np.arange(1, n_comp + 1), repeats=len(self.holdings))
        holdings = np.tile(self.holdings, reps=n_comp)
        loadings = self.components.reshape(-1)
        return (
            pl.DataFrame(
                {
                    "component": [f"PC{i}" for i in comp_idx],
                    "holding": holdings.tolist(),
                    "loading": loadings.tolist(),
                }
            )
            .with_columns(pl.col("loading").abs().alias("_abs"))
            .sort(["component", "_abs"], descending=[False, True])
            .drop("_abs")
        )


def etf_pca(
    tickers: list[str],
    *,
    n_components: int | None = None,
) -> ETFPCAResult:
    """Run PCA on a set of ETFs based on their weighted holdings.

    Steps:

    1. Fetch each ETF's holdings in canonical schema via
       :func:`bestee_compute.etf.normalize.get_holdings_normalized` (dispatches
       by issuer and projects to the cross-issuer column layout).
    2. Build a ``(n_etfs, n_holdings)`` weight matrix over the union of
       all constituents, with zeros where an ETF doesn't hold one.
       Constituents are keyed by uppercased ticker, falling back to
       uppercased name for issuers/funds that report no ticker.
    3. Center each column on its mean across ETFs, then run SVD.
    4. Components = right-singular vectors;  explained variance =
       ``s ** 2 / (n_etfs - 1)``;  projections = centered @ Vᵀ.

    Note: PCA on a centered matrix of shape ``(n, p)`` has at most
    ``n - 1`` informative components, so passing many fewer ETFs than
    holdings limits how many directions are meaningful — the rest will
    have ~zero explained variance.

    Args:
        tickers: At least two ETF tickers.  Each is dispatched via the
            issuer-aware :mod:`bestee_compute.etf.holdings` registry, so any
            ticker supported there works (VanEck, SPDR, iShares,
            Roundhill at time of writing).
        n_components: Truncate to this many components.  Defaults to
            all (``min(n_etfs, n_holdings)``).

    Returns:
        :class:`ETFPCAResult`.

    Raises:
        ValueError: If fewer than two tickers are passed, none of the
            input ETFs has any usable holdings, or all input ETFs
            collapse to the same weight vector (variance is zero).
        IssuerNotSupportedError: If an input ETF's issuer has no
            registered scraper.
    """
    if len(tickers) < 2:
        msg = f"PCA needs at least 2 ETFs; got {len(tickers)}."
        raise ValueError(msg)

    canonical: list[pl.DataFrame] = [
        _join_key(get_holdings_normalized(ticker)) for ticker in tickers
    ]

    all_holdings = sorted(
        {h for frame in canonical for h in frame["holding"].to_list()}
    )
    if not all_holdings:
        msg = "None of the input ETFs has any usable holdings."
        raise ValueError(msg)
    holding_idx = {h: i for i, h in enumerate(all_holdings)}

    weights = np.zeros((len(tickers), len(all_holdings)), dtype=np.float64)
    for row, frame in enumerate(canonical):
        for h, w in zip(
            frame["holding"].to_list(), frame["weight"].to_list(), strict=True
        ):
            weights[row, holding_idx[h]] = w

    centered = weights - weights.mean(axis=0, keepdims=True)

    # SVD of the centered matrix:  centered = U @ diag(s) @ Vt.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if not np.any(s > 0):
        msg = "All input ETFs have identical weight vectors — PCA is undefined."
        raise ValueError(msg)

    explained_variance = (s**2) / max(len(tickers) - 1, 1)
    total = explained_variance.sum()
    explained_variance_ratio = (
        explained_variance / total if total > 0 else np.zeros_like(explained_variance)
    )

    # Projections (scores) = centered @ Vt.T = U @ diag(s).
    projections = centered @ vt.T

    if n_components is not None:
        k = min(n_components, vt.shape[0])
        vt = vt[:k]
        explained_variance = explained_variance[:k]
        explained_variance_ratio = explained_variance_ratio[:k]
        projections = projections[:, :k]

    return ETFPCAResult(
        tickers=[t.upper() for t in tickers],
        holdings=all_holdings,
        weights=weights,
        components=vt,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        projections=projections,
    )
