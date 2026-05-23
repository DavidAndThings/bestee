"""Main entry point for bestee."""

from bestee.stocks.tickers import get_all_tickers


def main() -> None:
    """Run the main application."""
    table = get_all_tickers()
    table.show()


if __name__ == "__main__":
    main()
