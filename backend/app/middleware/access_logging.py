"""ASGI HTTP Access Logging Middleware for LungCare AI Backend.

Provides structured, privacy-preserving HTTP access logging:
1. Records HTTP method, request path, response status code, execution duration (ms),
   and correlation request_id.
2. Intercepts http.response.start to capture the effective response status without
   buffering or modifying request/response bodies.
3. Operates within the RequestCorrelationMiddleware context to ensure consistent
   request correlation tracking.
4. Enforces strict medical and user privacy: strictly excludes image bytes, filenames,
   Base64 overlays, query strings, headers, cookies, and sensitive payload data.
5. Handles exceptions and cancellations cleanly in a finally block without ContextVar leakage.
6. Preserves non-HTTP ASGI scopes (e.g. lifespan, websocket).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import get_request_id

logger = logging.getLogger("lungcare.access")


class AccessLoggingMiddleware:
    """Pure ASGI middleware providing structured HTTP access event logging."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_code: Optional[int] = None
        is_cancelled = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            is_cancelled = True
            raise
        except Exception as exc:
            # Capture client disconnects as cancellation events
            if exc.__class__.__name__ == "ClientDisconnect":
                is_cancelled = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "")
            req_id = get_request_id()

            # Format status representation cleanly:
            # - If response headers were sent, report actual HTTP status (e.g. 200, 404, 500)
            # - If cancelled before response started, report CANCELLED (never a synthetic 500)
            # - If unhandled failure escaped before response started, report NO_RESPONSE
            if status_code is not None:
                status_repr = str(status_code)
            elif is_cancelled:
                status_repr = "CANCELLED"
            else:
                status_repr = "NO_RESPONSE"

            # Strictly log only non-sensitive routing and telemetry fields
            # (query_string, headers, cookies, and payload bodies are strictly excluded)
            # Note: record.request_id is automatically attached to LogRecord by _request_context_log_record_factory
            logger.info(
                "HTTP %s %s -> %s [%.2fms]",
                method,
                path,
                status_repr,
                duration_ms,
                extra={
                    "http_method": method,
                    "request_path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
