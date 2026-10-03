"""Authentication dependency.

Routes depend only on `get_principal` and read `principal.owner_id`. Swapping API-key
auth for real user auth later means replacing this one function.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import Settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    owner_id: str


def get_principal(
    request: Request, supplied: str | None = Depends(_api_key_header)
) -> Principal:
    settings: Settings = request.app.state.settings
    if not supplied or not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key (send the X-API-Key header).",
        )
    return Principal(owner_id=settings.default_owner_id)
