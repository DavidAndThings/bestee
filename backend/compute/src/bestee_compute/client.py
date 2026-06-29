"""Shared Massive API client utility."""

import logging
import os

from dotenv import load_dotenv
from massive import RESTClient

logger = logging.getLogger(__name__)


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
    logger.debug("Resolving API key (explicit=%s)", api_key is not None)
    load_dotenv()
    key = api_key or os.environ.get("MASSIVE_API_KEY")
    logger.debug("API key source: %s", "explicit" if api_key else "environment")
    if key is None:
        msg = (
            "No API key provided. Pass one via the 'api_key' parameter "
            "or set the MASSIVE_API_KEY environment variable."
        )
        logger.error("No API key found in parameter or MASSIVE_API_KEY env var")
        raise RuntimeError(msg)
    logger.debug("RESTClient created successfully")
    return RESTClient(api_key=key)
