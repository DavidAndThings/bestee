"""Shared Massive API client utility."""

import os

from dotenv import load_dotenv
from massive import RESTClient


def get_client(api_key: str | None = None) -> RESTClient:
    """Resolve the API key and return a configured RESTClient.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.

    Returns:
        A configured :class:`massive.RESTClient`.

    Raises:
        RuntimeError: If no API key is available.
    """
    load_dotenv()
    key = api_key or os.environ.get("MASSIVE_API_KEY")
    if key is None:
        msg = (
            "No API key provided. Pass one via the 'api_key' parameter "
            "or set the MASSIVE_API_KEY environment variable."
        )
        raise RuntimeError(msg)
    return RESTClient(api_key=key)
