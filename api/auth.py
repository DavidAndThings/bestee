"""Clerk session-token authentication for FastAPI.

Every protected endpoint declares a dependency on :func:`require_auth`, which
verifies the ``Authorization: Bearer <session_token>`` header against Clerk's
JWKS endpoint and returns the decoded JWT claims.

Configure by adding to your ``.env`` file::

    CLERK_JWKS_URL=https://<instance>.clerk.accounts.dev/.well-known/jwks.json

The JWKS client caches public keys in memory and re-fetches automatically on
key rotation, so there is no per-request network call after warm-up.
"""

import os
from datetime import timedelta
from typing import Annotated, Any

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

load_dotenv()

_CLERK_JWKS_URL: str = os.environ.get("CLERK_JWKS_URL", "")
_bearer = HTTPBearer(auto_error=False)
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Return a cached :class:`PyJWKClient`, initialised on first use."""
    global _jwks_client
    if not _CLERK_JWKS_URL:
        raise RuntimeError(
            "CLERK_JWKS_URL is not configured. "
            "Add it to your .env file — e.g. "
            "CLERK_JWKS_URL=https://<instance>.clerk.accounts.dev/.well-known/jwks.json"
        )
    if _jwks_client is None:
        _jwks_client = PyJWKClient(_CLERK_JWKS_URL, cache_keys=True)
    return _jwks_client


def require_auth(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer),
    ],
) -> dict[str, Any]:
    """FastAPI dependency that verifies a Clerk session token.

    Reads the ``Authorization: Bearer <session_token>`` header, verifies the
    JWT signature against Clerk's JWKS, and returns the decoded claims.
    Raises ``HTTP 401`` if the token is missing, malformed, or expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(
            credentials.credentials
        )
        claims: dict[str, Any] = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=["RS256"],
            leeway=timedelta(seconds=10),
            options={"verify_aud": False},
        )
        return claims
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
