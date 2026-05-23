"""Run every bundled DSL stress input end-to-end against the live API."""

import importlib.resources
import logging
import sys
import time
import traceback

import bestee.resources
from bestee.stocks.decorators import decorator_builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("stress")


def main() -> int:
    files = sorted(
        p.name
        for p in importlib.resources.files(bestee.resources).iterdir()
        if p.name.endswith(".txt")
    )
    results: list[tuple[str, str, float]] = []

    for fname in files:
        log.info("===== START %s =====", fname)
        t0 = time.monotonic()
        try:
            text = (
                importlib.resources.files(bestee.resources)
                .joinpath(fname)
                .read_text(encoding="utf-8")
            )
            commands = [line.split() for line in text.splitlines() if line.strip()]
            pipeline = decorator_builder(commands)
            gt = pipeline.build()
            # Force GT to materialize so any deferred work runs.
            html_len = len(gt.as_raw_html())
            elapsed = time.monotonic() - t0
            log.info("PASS %s in %.1fs (html=%d bytes)", fname, elapsed, html_len)
            results.append((fname, "PASS", elapsed))
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("FAIL %s in %.1fs: %s", fname, elapsed, e)
            traceback.print_exc()
            results.append((fname, f"FAIL: {type(e).__name__}: {e}", elapsed))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    any_fail = False
    for fname, status, elapsed in results:
        print(f"  {status:<60s} {elapsed:>7.1f}s  {fname}")
        if status != "PASS":
            any_fail = True
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
