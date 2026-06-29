"""Cluster the US common-stock universe (no ETFs) by residual co-movement.

Screening is kept *out* of the analysis pipeline. ``get_clean_panel`` fetches the
whole-universe daily Close+Volume once (null-preserving, so no inner-join
collapse), drops short-history names, and fills the price gaps -> a rectangular
panel. That panel is injected into the ``AnalysisConfig`` that
``run_spectral_clustering`` reads from, which clusters the liquid names
spectrally and assigns each illiquid name to the nearest cluster.

Spectral clustering's per-k search is infeasible on a few-thousand-name affinity,
so the cluster count is pinned (``--n-clusters``) to a single eigendecomposition.

Credentials are read from ``compute/.env.local`` (must define MASSIVE_API_KEY).

Run::

    uv run python scripts/run_cluster_stocks.py
    uv run python scripts/run_cluster_stocks.py --use-cache --n-clusters 14
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.stocks.tickers import get_all_tickers_df
from bestee_compute.workflow.clustering import run_spectral_clustering
from bestee_compute.workflow.tools import AnalysisConfig, get_clean_panel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
# The grouped-daily fan-out runs more workers than urllib3's default pool size,
# which logs a churn warning per connection -- harmless, just noisy.
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
log = logging.getLogger("cluster_stocks")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env.local"
OUT_DIR = Path(__file__).resolve().parent
PANEL_CACHE = OUT_DIR / "stock_panel.parquet"
# CS = common stock, ADRC = ADR common: the equity universe people call
# "stocks". Everything ETF-like and non-equity is excluded by not being here.
DEFAULT_TYPES = ("CS", "ADRC")


def _stock_universe(types: tuple[str, ...]) -> list[str]:
    """Active common-equity tickers of the requested types (ETFs excluded)."""
    df = get_all_tickers_df(market="stocks", active=True)
    universe = df.filter(pl.col("Type").is_in(types))
    log.info("Stock universe: %d tickers of types %s", universe.height, list(types))
    return sorted(universe["Ticker"].to_list())


def _summarize(labels: Mapping[str, int]) -> pl.DataFrame:
    """Log cluster sizes + samples and return a tidy Ticker/Cluster frame."""
    groups: dict[int, list[str]] = {}
    for ticker, cluster_id in labels.items():
        groups.setdefault(cluster_id, []).append(ticker)
    log.info("Clustered %d stocks into %d clusters", len(labels), len(groups))
    for cluster_id, members in sorted(
        groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        sample = ", ".join(sorted(members)[:10])
        more = "" if len(members) <= 10 else f", +{len(members) - 10} more"
        log.info(
            "  cluster %d: %d members [%s%s]", cluster_id, len(members), sample, more
        )
    return pl.DataFrame(
        {"Ticker": list(labels), "Cluster": [labels[t] for t in labels]}
    ).sort("Cluster", "Ticker")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="Comma-separated ticker types to include (default: CS,ADRC).",
    )
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=5_000_000.0,
        help=(
            "Liquidity floor: names below this median daily dollar volume are "
            "assigned to the nearest cluster rather than clustered."
        ),
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=11,
        help="Cluster count (pinned -- a single spectral eigendecomposition).",
    )
    parser.add_argument("--residualization-window", type=int, default=60)
    parser.add_argument("--normalization-window", type=int, default=60)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse the cached screened panel (skips the universe download).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        log.info("Loaded credentials from %s", ENV_PATH)
    else:
        log.warning("No .env.local at %s; relying on existing environment", ENV_PATH)

    if args.use_cache:
        panel = pl.read_parquet(PANEL_CACHE)
        log.info(
            "Loaded cached panel: %d days x %d columns", panel.height, panel.width - 1
        )
    else:
        types = tuple(t.strip() for t in args.types.split(",") if t.strip())
        universe = _stock_universe(types)
        screen_config = AnalysisConfig(
            tickers=universe,
            start_date=args.start,
            end_date=args.end,
            min_coverage=args.min_coverage,
            max_workers=args.workers,
        )
        # Close+Volume so the runner reads the liquidity median straight off the
        # panel (Volume keeps its non-trading-day nulls; prices are gap-filled).
        panel = get_clean_panel(screen_config, OHLCHeader.CLOSE, OHLCHeader.VOLUME)
        panel.write_parquet(PANEL_CACHE)
        log.info("Cached screened panel to %s (reuse with --use-cache)", PANEL_CACHE)

    close_suffix = f"_{OHLCHeader.CLOSE}"
    covered = [
        col.removesuffix(close_suffix)
        for col in panel.columns
        if col.endswith(close_suffix)
    ]
    config = AnalysisConfig(
        tickers=covered,
        injected_panel=panel,
        start_date=args.start,
        end_date=args.end,
        min_dollar_volume=args.min_dollar_volume,
        max_workers=args.workers,
        ohlc_column=OHLCHeader.CLOSE,
        residualization_window=args.residualization_window,
        normalization_window=args.normalization_window,
        n_components=args.n_components,
        clustering_assign_illiquid=True,
        clustering_min_num_clusters=args.n_clusters,
        clustering_max_num_clusters=args.n_clusters,
    )
    log.info(
        "Spectral clustering %d screened stocks (liquid + assigned illiquid)...",
        len(covered),
    )
    labels = run_spectral_clustering(config)

    assignments = _summarize(labels)
    csv_path = OUT_DIR / "stock_clusters.csv"
    assignments.write_csv(csv_path)
    assignments.write_parquet(OUT_DIR / "stock_clusters.parquet")
    log.info("Wrote assignments to %s", csv_path)
    print(f"Clustered {len(labels)} stocks into {len(set(labels.values()))} clusters")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
