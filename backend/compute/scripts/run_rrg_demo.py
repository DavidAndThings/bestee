"""Sample Relative Rotation Graph (RRG) built from ``RRGConfig`` on live data.

Pulls real OHLC bars from the Massive API through the existing ``RRGConfig``
pipeline unchanged (``normalized_relative_strength`` for the X axis and
``normalized_relative_momentum`` for the Y axis) and renders the classic RRG
quadrant scatter with a rotation "tail" per ticker.

Credentials are read from ``compute/.env.local`` (must define
``MASSIVE_API_KEY``). RRGs are conventionally built on weekly bars, so that is
the default here; switch ``TIMESPAN`` to ``"day"`` for a noisier daily view.

Note: this module's normalized series are z-scores, so the chart is centered
on ``0`` rather than the ``100`` of a conventional (offset) RRG.

Run::

    uv run python scripts/run_rrg_demo.py
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from matplotlib.patches import Rectangle

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow import rrg
from bestee_compute.workflow.tools import AnalysisConfig, ReferenceType, Timespan

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("rrg_demo")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env.local"

TICKERS = ["AIRR", "SPY", "SMH", "ARKF", "QQQ"]
REFERENCE_TYPE: ReferenceType = "mean"  # "mean" (basket) or "ticker"
REFERENCE_TICKER = "SPY"  # used only when REFERENCE_TYPE == "ticker"
REFERENCE_LABEL = (
    REFERENCE_TICKER if REFERENCE_TYPE == "ticker" else "equal-weight basket"
)
TIMESPAN: Timespan = "day"
MULTIPLIER = 1  # daily bars
START = "2024-01-01"
END = "2026-06-01"
if MULTIPLIER > 1:
    BAR_LABEL = f"{MULTIPLIER}-{TIMESPAN}"
elif TIMESPAN == "day":
    BAR_LABEL = "daily"
else:
    BAR_LABEL = f"{TIMESPAN}ly"

# normalization_window_size defines the quadrant baseline ("strong/weak vs the
# last N bars") -- an analytical horizon. For daily bars smoothness is flat
# across windows, so a shorter window costs ~no smoothness but SEPARATES similar
# names: a long window bunches the laggards together; window=14 (~3-week horizon)
# maximizes separation -- the most responsive / most-recent view. span=5 and
# lookback=5 remain the smoothing optimum.
WINDOW = 14
LOOKBACK = 3  # momentum_lookback = window of the rolling regression-slope momentum
SMOOTHING_SPAN = 19  # EMA span applied to the relative-strength line
TAIL = 8  # number of trailing points drawn per ticker (~2.5 weeks on daily bars)


def _rrg_frame(config: AnalysisConfig) -> pl.DataFrame:
    """Join the normalized RS (X) and normalized momentum (Y) on Timestamp."""
    rs_ratio = rrg.normalized_relative_strength(config)
    rs_momentum = rrg.normalized_relative_momentum(config)
    return rs_ratio.join(rs_momentum, on=OHLCHeader.TIMESTAMP, how="inner")


def _plot_rrg(
    frame: pl.DataFrame,
    tickers: list[str],
    out_path: Path,
    smoothing_span: int | None,
) -> None:
    plt.switch_backend("Agg")
    rs_suffix = rrg._RELATIVE_STRENGTH_COLUMN_NAME
    mom_suffix = rrg._RELATIVE_MOMENTUM_COLUMN_NAME

    paths = {
        tk: (
            frame[f"{tk}_{rs_suffix}"].to_numpy(),
            frame[f"{tk}_{mom_suffix}"].to_numpy(),
        )
        for tk in tickers
    }
    extent = np.concatenate([np.abs(np.concatenate(xy)) for xy in paths.values()])
    lim = max(float(np.nanmax(extent)) * 1.15, 0.5)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Standard RRG quadrant tints.
    quadrants = [
        ((0, 0), "#3fae5a", "Leading", (1, 1)),
        ((0, -lim), "#d8c63a", "Weakening", (1, -1)),
        ((-lim, -lim), "#e0544f", "Lagging", (-1, -1)),
        ((-lim, 0), "#4f8fe0", "Improving", (-1, 1)),
    ]
    for (x0, y0), color, label, (sx, sy) in quadrants:
        ax.add_patch(Rectangle((x0, y0), lim, lim, color=color, alpha=0.10, zorder=0))
        ax.text(
            sx * lim * 0.96,
            sy * lim * 0.96,
            label,
            ha="right" if sx > 0 else "left",
            va="top" if sy > 0 else "bottom",
            fontsize=12,
            weight="bold",
            color=color,
            alpha=0.9,
        )

    ax.axhline(0, color="0.35", lw=1.0, zorder=1)
    ax.axvline(0, color="0.35", lw=1.0, zorder=1)

    cmap = plt.get_cmap("tab10")
    for i, tk in enumerate(tickers):
        x, y = paths[tk]
        xt, yt = x[-TAIL:], y[-TAIL:]
        color = cmap(i % 10)
        ax.plot(xt, yt, "-", color=color, lw=1.6, alpha=0.55, zorder=2)
        ax.scatter(
            xt, yt, s=np.linspace(12, 55, len(xt)), color=color, alpha=0.7, zorder=3
        )
        ax.scatter(
            xt[-1], yt[-1], s=150, color=color, edgecolor="black", lw=1.2, zorder=4
        )
        ax.annotate(
            tk,
            (xt[-1], yt[-1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color=color,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, color="0.92", lw=0.6, zorder=0)
    ax.set_xlabel("JdK RS-Ratio  (normalized relative strength, 0-centered)")
    ax.set_ylabel("JdK RS-Momentum  (normalized, 0-centered)")
    smoothing = f"EMA span {smoothing_span}" if smoothing_span else "no smoothing"
    ax.set_title(
        f"Relative Rotation Graph  (reference: {REFERENCE_LABEL}, {smoothing})\n"
        f"{BAR_LABEL} bars, {START} to {END}; "
        f"last {TAIL} points per ticker, large dot = most recent"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        log.info("Loaded credentials from %s", ENV_PATH)
    else:
        log.warning("No .env.local at %s; relying on existing environment", ENV_PATH)

    if not os.environ.get("MASSIVE_API_KEY"):
        log.error("MASSIVE_API_KEY is not set (looked in %s).", ENV_PATH)
        return 1

    # Render raw (unsmoothed) and EMA-smoothed versions side by side.
    for smoothing_span, out_name in [
        (None, "rrg_raw.png"),
        (SMOOTHING_SPAN, "rrg_smoothed.png"),
    ]:
        config = AnalysisConfig(
            tickers=TICKERS,
            benchmark_ticker=(REFERENCE_TICKER if REFERENCE_TYPE == "ticker" else None),
            timespan=TIMESPAN,
            bar_multiplier=MULTIPLIER,
            start_date=START,
            end_date=END,
            ohlc_column=OHLCHeader.CLOSE,
            rrg_reference_type=REFERENCE_TYPE,
            normalization_window=WINDOW,
            rrg_momentum_lookback=LOOKBACK,
            rrg_smoothing_span=smoothing_span,
        )
        log.info(
            "Computing RRG (%s bars, %s..%s, reference=%s, smoothing=%s)...",
            BAR_LABEL,
            START,
            END,
            REFERENCE_LABEL,
            smoothing_span,
        )
        frame = _rrg_frame(config)
        if frame.is_empty():
            log.error("No overlapping bars returned for the requested range/tickers.")
            return 1
        out_path = Path(__file__).resolve().parent / out_name
        _plot_rrg(frame, TICKERS, out_path, smoothing_span)
        log.info("RRG (%s): %d rows -> %s", out_name, frame.height, out_path)
        print(f"RRG written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
