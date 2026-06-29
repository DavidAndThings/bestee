"""Resolve user-supplied search terms into ticker symbols before dispatch.

The UI lets users pick *terms* -- ticker symbols, company names, or SIC industry
titles -- so the task-submission routers run each request's ``tickers`` (and an
RRG ``benchmark_ticker``) through :func:`resolve_terms_to_tickers` here, turning
the terms into the concrete symbols the analyses operate on.  The resolved
payload is then re-validated against the same request model, so an analysis' own
constraints (e.g. clustering's two-ticker minimum) apply to the *resolved*
universe rather than to the raw term count.
"""

from typing import Any

from bestee_compute.stocks.tickers import (
    resolve_term_to_ticker,
    resolve_terms_to_tickers,
)
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError


def resolve_request(request: BaseModel) -> dict[str, Any]:
    """Return a dispatch-ready payload with ticker terms resolved to symbols.

    Resolves the ``tickers`` list and, when present, a single
    ``benchmark_ticker``, then re-validates the result against *request*'s model.

    Args:
        request: A submitted ``*Request`` model whose ticker fields hold the
            user's search terms.

    Returns:
        The re-validated payload (``model_dump``) with terms replaced by symbols.

    Raises:
        HTTPException: 422 when terms resolve to no securities, or when the
            resolved payload fails the request model's own validation.
    """
    data = request.model_dump()

    terms = data.get("tickers")
    if isinstance(terms, list):
        resolved = resolve_terms_to_tickers(terms)
        if not resolved:
            raise HTTPException(
                status_code=422,
                detail="No securities matched the provided terms.",
            )
        data["tickers"] = resolved

    benchmark = data.get("benchmark_ticker")
    if benchmark:
        symbol = resolve_term_to_ticker(benchmark)
        if symbol is None:
            raise HTTPException(
                status_code=422,
                detail=f"Benchmark term {benchmark!r} matched no security.",
            )
        data["benchmark_ticker"] = symbol

    try:
        return type(request).model_validate(data).model_dump()
    except ValidationError as exc:
        messages = "; ".join(error["msg"] for error in exc.errors(include_url=False))
        raise HTTPException(
            status_code=422,
            detail=f"Resolved securities are invalid for this analysis: {messages}",
        ) from exc
