"""ASGI Request Correlation Middleware for LungCare AI Backend.

Extracts, validates, or generates unique request correlation identifiers (X-Request-ID),
binds them to async contextvars for downstream logging, and guarantees header propagation
on all HTTP responses including error responses.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import (
    FALLBACK_REQUEST_ID,
    HEADER_REQUEST_ID,
    generate_request_id,
    is_valid_request_id,
    reset_request_id,
    set_request_id,
)

logger = logging.getLogger("lungcare.middleware.correlation")


class RequestCorrelationMiddleware:
    """Pure ASGI middleware providing request-scoped correlation tracking."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. Extract X-Request-ID from raw ASGI headers
        incoming_request_id = None
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == b"x-request-id":
                try:
                    incoming_request_id = raw_value.decode("latin1").strip()
                except UnicodeDecodeError:
                    incoming_request_id = None
                break

        # 2. Validate or generate request ID
        if is_valid_request_id(incoming_request_id):
            effective_request_id = incoming_request_id
        else:
            effective_request_id = generate_request_id()

        # 3. Store in scope state for FastAPI dependency access if needed
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = effective_request_id

        # 4. Set in contextvars for logging and request-scoped access
        token = set_request_id(effective_request_id)

        # 5. Intercept response to inject X-Request-ID header
        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers: List[Tuple[bytes, bytes]] = list(message.get("headers", []))
                # Remove any existing x-request-id headers to prevent duplicate header emissions
                filtered_headers = [
                    (k, v) for k, v in raw_headers if k.lower() != b"x-request-id"
                ]
                filtered_headers.append((b"x-request-id", effective_request_id.encode("latin1")))
                message["headers"] = filtered_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)
