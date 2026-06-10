"""Small shared helpers for the wiki subpackage.

These were previously duplicated across :mod:`download`, :mod:`vectors`,
:mod:`builder`, :mod:`query` and :mod:`knowledge`; they live here so there is a
single definition of each.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bestee_chat.embeddings import Encoder

#: Progress callback: ``(items_done, total_or_None) -> None``. ``total`` is
#: ``None`` when the work's size is not known up front (e.g. a streamed
#: download with no Content-Length).
ProgressCallback = Callable[[int, int | None], None]


def normalise(values: list[float]) -> list[float]:
    """Min-max scale ``values`` into ``[0, 1]`` (all ``0.5`` when degenerate)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    span = hi - lo
    return [(value - lo) / span for value in values]


def optional_encoder() -> Encoder | None:
    """Return the process-wide encoder, or ``None`` if ``torch`` is unavailable.

    Imported lazily so that lexical-only use never pulls in the ML stack.
    """
    try:
        from bestee_chat.embeddings import default_encoder
    except Exception:
        return None
    return default_encoder()
