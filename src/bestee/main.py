"""Main entry point for bestee."""

from bestee.tickers import get_all_tickers


def main() -> None:
    """Run the main application."""
    tickers = get_all_tickers()
    print(f"Fetched {len(tickers)} tickers\n")
    for t in tickers[:10]:
        print(f"  {t.ticker:<10} {t.name}")
    if len(tickers) > 10:
        print(f"  ... and {len(tickers) - 10} more")


if __name__ == "__main__":
    main()
