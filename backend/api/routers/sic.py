"""Browse SIC industry codes and the tickers classified under each.

``GET /sic`` returns every SIC industry code and its title; ``GET
/sic/{sic_code}/tickers`` returns the ticker symbols classified under one code
(read from the bundled ticker -> SIC index, so it's offline and fast).
"""

from bestee_compute.stocks.sic import get_sic_codes_df
from bestee_compute.stocks.tickers import get_ticker_name_map, get_tickers_by_sic_code
from fastapi import APIRouter

from schemas import SicCode, SicCodeList, SicTicker, SicTickers

router = APIRouter(prefix="/sic", tags=["sic"])


@router.get("", response_model=SicCodeList)
def list_sic_codes() -> SicCodeList:
    """Every SIC industry code and its title."""
    df = get_sic_codes_df()
    codes = [
        SicCode(sic_code=row["SIC Code"], industry_title=row["Industry Title"])
        for row in df.iter_rows(named=True)
    ]
    return SicCodeList(count=len(codes), codes=codes)


@router.get("/{sic_code}/tickers", response_model=SicTickers)
def tickers_for_sic(sic_code: str) -> SicTickers:
    """The tickers classified under *sic_code*, each with its company name."""
    names = get_ticker_name_map()
    items = [
        SicTicker(ticker=ticker, name=names.get(ticker))
        for ticker in get_tickers_by_sic_code(sic_code)
    ]
    return SicTickers(sic_code=sic_code, count=len(items), tickers=items)
