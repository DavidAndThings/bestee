"""Per-asset regime detection on live data via ``RegimeAnalysisRunner``.

Fetches real daily OHLC from the Massive API, residualizes each name against a
benchmark (market-model OLS), and fits a per-asset autoregressive Gaussian HMM
on ``[residual, log_vol]`` to label every day as a calm or turbulent regime.
Prints a per-asset summary and renders a regime-shaded price panel per name.

Credentials are read from ``compute/.env.local`` (must define ``MASSIVE_API_KEY``).

Run::

    uv run python scripts/run_regime_demo.py
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from matplotlib.patches import Patch

from bestee_compute.stocks.models import OHLCHeader
from bestee_compute.workflow.regimes import RegimeResult, run_regime_analysis
from bestee_compute.workflow.tools import AnalysisConfig, RegimeFeature, column

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
# A sticky prior makes EM a MAP fit, so hmmlearn's likelihood monitor logs benign
# "not converging" lines; the n_init restarts keep the best fit regardless.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
log = logging.getLogger("regime_demo")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env.local"
OUT_PATH = Path(__file__).resolve().parent / "regime_demo.png"

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
BENCHMARK = "SPY"
START = "2019-01-01"
END = "2026-06-01"

HIDDEN_STATES = 2  # calm vs. turbulent
LAG = 1
RESIDUALIZATION_WINDOW = 60  # rolling market-model (beta) window
NORMALIZATION_WINDOW = 60
VOL_HALFLIFE = 20
# Sticky persistence prior: daily residuals otherwise flip regimes every ~2 days;
# this biases toward multi-week regimes without adding fitted parameters.
STICKY = 3.0
FEATURES: list[RegimeFeature] = ["residual", "log_vol"]


def _build_config() -> AnalysisConfig:
    return AnalysisConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start_date=START,
        end_date=END,
        ohlc_column=OHLCHeader.CLOSE,
        residualization_method="ols",
        residualization_window=RESIDUALIZATION_WINDOW,
        normalization_window=NORMALIZATION_WINDOW,
        regime_hmm_hidden_states=HIDDEN_STATES,
        regime_hmm_lag=LAG,
        regime_features=FEATURES,
        regime_vol_halflife=VOL_HALFLIFE,
        regime_sticky=STICKY,
        regime_covariance_type="diag",
        regime_n_init=10,
        regime_n_iter=200,
        regime_random_state=0,
    )


def _summarize(results: dict[str, RegimeResult]) -> None:
    for ticker, result in results.items():
        summary = result.regime_summary()
        log.info(
            "%s: current regime %d/%d | %s",
            ticker,
            result.current_regime(),
            result.n_states - 1,
            "  ".join(
                f"R{int(row['Regime'])}: {row['Frequency']:.0%} of days, "
                f"~{row['RealizedDwell']:.0f}d runs"
                for row in summary.iter_rows(named=True)
            ),
        )


def _regime_runs(dates: list, regimes: np.ndarray) -> list[tuple]:
    """Contiguous ``(start_date, end_date, regime)`` spans for shading."""
    if len(regimes) == 0:
        return []
    breaks = [0, *(np.where(np.diff(regimes) != 0)[0] + 1).tolist(), len(regimes)]
    spans = []
    for start, stop in zip(breaks[:-1], breaks[1:], strict=True):
        spans.append(
            (dates[start], dates[min(stop, len(dates) - 1)], int(regimes[start]))
        )
    return spans


def _plot(results: dict[str, RegimeResult], closes: pl.DataFrame) -> None:
    plt.switch_backend("Agg")
    cmap = plt.get_cmap("RdYlGn_r")
    n_states = max(r.n_states for r in results.values())
    colors = [cmap(i / max(n_states - 1, 1)) for i in range(n_states)]

    tickers = list(results)
    fig, axes = plt.subplots(
        len(tickers), 1, figsize=(13, 2.6 * len(tickers)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, ticker in zip(axes, tickers, strict=True):
        result = results[ticker]
        merged = result.states.join(
            closes.select(OHLCHeader.TIMESTAMP, f"{ticker}_{OHLCHeader.CLOSE}"),
            on=OHLCHeader.TIMESTAMP,
            how="inner",
        )
        dates = merged[OHLCHeader.TIMESTAMP].to_list()
        price = merged[f"{ticker}_{OHLCHeader.CLOSE}"].to_numpy()
        regimes = merged["Regime"].to_numpy()
        for start, stop, regime in _regime_runs(dates, regimes):
            ax.axvspan(start, stop, color=colors[regime], alpha=0.25, zorder=0)
        ax.plot(dates, price, color="black", lw=1.0, zorder=2)
        ax.set_yscale("log")
        ax.set_ylabel(f"{ticker}\nclose ($)")
        ax.margins(x=0)
        ax.set_title(
            f"{ticker} — current regime {result.current_regime()} "
            f"(0=calm … {result.n_states - 1}=turbulent)",
            fontsize=10,
            loc="left",
        )

    legend = [
        Patch(facecolor=colors[s], alpha=0.4, label=f"regime {s}")
        for s in range(n_states)
    ]
    axes[0].legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Per-asset AR-HMM regimes (OLS residual vs {BENCHMARK}, "
        f"features=[residual, log_vol], sticky={STICKY:g}) — {START} to {END}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PATH, dpi=130)
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

    config = _build_config()
    log.info("Fitting per-asset AR-HMM regimes for %s ...", ", ".join(TICKERS))
    results = run_regime_analysis(config)
    if not results:
        log.error("No asset cleared the parameter budget; nothing to plot.")
        return 1

    _summarize(results)
    closes = column(config, OHLCHeader.CLOSE)
    _plot(results, closes)
    log.info("Wrote %s", OUT_PATH)
    print(f"Regime plot written to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
