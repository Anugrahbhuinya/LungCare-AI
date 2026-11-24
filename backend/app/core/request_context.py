"""Request-scoped context tracking for LungCare AI Backend.

Provides thread-safe, async-safe request correlation tracking using contextvars.
Supports safe accessor, validation, and ContextVar token lifecycle management.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
import re
from typing import Any, Optional
import uuid

# Valid Request ID: 1 to 64 alphanumeric characters, underscores, or hyphens
REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
FALLBACK_REQUEST_ID = "-"
HEADER_REQUEST_ID = "X-Request-ID"

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default=FALLBACK_REQUEST_ID)


def is_valid_request_id(request_id: Optional[str]) -> bool:
    """Validate incoming request ID against allowed character and length constraints."""
    if not request_id or not isinstance(request_id, str):
        return False
    return bool(REQUEST_ID_REGEX.fullmatch(request_id))


def generate_request_id() -> str:
    """Generate a cryptographically random UUID4 request correlation identifier."""
    return str(uuid.uuid4())


def get_request_id(request: Optional[Any] = None) -> str:
    """Return the current request ID from contextvars, request state, or fallback '-'."""
    req_id = _request_id_ctx.get()
    if req_id != FALLBACK_REQUEST_ID:
        return req_id
    if request is not None:
        state = getattr(request, "state", None)
        if state is not None:
            state_id = getattr(state, "request_id", None)
            if state_id:
                return state_id
        scope = getattr(request, "scope", None)
        if isinstance(scope, dict):
            scope_state = scope.get("state")
            if isinstance(scope_state, dict):
                state_id = scope_state.get("request_id")
                if state_id:
                    return state_id
    return FALLBACK_REQUEST_ID


def set_request_id(request_id: str) -> Token[str]:
    """Set the request ID in contextvars and return the token for cleanup."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Reset the request ID ContextVar to its prior value using the provided token."""
    _request_id_ctx.reset(token)
