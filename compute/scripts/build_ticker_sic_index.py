"""Rebuild the bundled ticker -> SIC code cache.

Fetches every active stock from the Massive API, reads each company's SIC code
from its ``TickerDetails``, and writes the ``{ticker: sic_code}`` map shipped
with the package at ``resources/ticker_sic_codes.json``. This is the source for
``bestee_compute.stocks.tickers.get_tickers_by_sic_code``.

Credentials come from the repo-root ``.env`` (``MASSIVE_API_KEY``), loaded by the
Massive client. Run after a refresh is needed::

    uv run python scripts/build_ticker_sic_index.py
"""

import json
import logging
import sys
from pathlib import Path

from bestee_compute.stocks.tickers import build_ticker_sic_index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("build_ticker_sic_index")

_OUT = (
    Path(__file__).resolve().parent.parent
    / "src/bestee_compute/resources/ticker_sic_codes.json"
)


def main() -> int:
    index = build_ticker_sic_index()
    # Sorted keys + one entry per line keeps diffs small and review-friendly.
    _OUT.write_text(json.dumps(index, indent=0, sort_keys=True) + "\n")
    log.info("Wrote %d ticker->SIC entries to %s", len(index), _OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
