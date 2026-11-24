"""Tests for ASGI Request Correlation Middleware and Contextual Logging.

Verifies:
1. Missing request ID generates a valid UUID4.
2. Valid incoming request ID is preserved.
3. Invalid request ID (disallowed characters) is replaced with a valid UUID4.
4. Empty / whitespace request ID is replaced with a valid UUID4.
5. Excessively long request ID (>64 chars) is replaced with a valid UUID4.
6. Response header X-Request-ID contains the effective request ID.
7. Application logs contain the correct request ID.
8. Execution outside HTTP context uses the fallback ID '-'.
9. Concurrent requests do not leak IDs into one another (deterministic async coordination).
10. ContextVar is reliably reset after request completion.
11. Exception responses (404, 500) preserve the correct request ID in headers and error envelope.
12. Validation error responses (422) preserve the correct request ID in headers and error envelope.
"""

import asyncio
import logging
import re
import uuid
import httpx
import pytest

from app.core.request_context import (
    FALLBACK_REQUEST_ID,
    HEADER_REQUEST_ID,
    get_request_id,
    is_valid_request_id,
)
from app.main import create_application

UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@pytest.fixture
def app():
    """Create a fresh FastAPI test application instance."""
    return create_application()


@pytest.fixture
async def client(app):
    """Provide an async test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c


# ------------------------------------------------------------------------------
# 1. Validation & ID Generation
# ------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_missing_request_id_generates_valid_uuid4(client: httpx.AsyncClient) -> None:
    """Requirement 1: Missing X-Request-ID header triggers automatic UUID4 generation."""
    response = await client.get("/health")
    assert response.status_code == 200

    header_id = response.headers.get("x-request-id")
    assert header_id is not None
    assert UUID4_PATTERN.match(header_id) is not None
    assert is_valid_request_id(header_id) is True


@pytest.mark.anyio
async def test_valid_incoming_request_id_is_preserved(client: httpx.AsyncClient) -> None:
    """Requirement 2: Valid client-provided request ID is strictly preserved."""
    custom_id = "req-client_alpha-999"
    response = await client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200

    header_id = response.headers.get("x-request-id")
    assert header_id == custom_id


@pytest.mark.anyio
async def test_invalid_characters_request_id_is_replaced(client: httpx.AsyncClient) -> None:
    """Requirement 3: Incoming request ID with invalid characters is safely replaced."""
    malformed_ids = [
        "req<script>alert(1)</script>",
        "req id with spaces",
        "req$special!symbols#",
        "req;drop table;",
        "req/path/traversal",
    ]
    for bad_id in malformed_ids:
        response = await client.get("/health", headers={"X-Request-ID": bad_id})
        assert response.status_code == 200

        header_id = response.headers.get("x-request-id")
        assert header_id is not None
        assert header_id != bad_id
        assert UUID4_PATTERN.match(header_id) is not None


@pytest.mark.anyio
async def test_empty_request_id_is_replaced(client: httpx.AsyncClient) -> None:
    """Requirement 4: Empty or whitespace-only request ID is replaced with UUID4."""
    for empty_val in ["", "   ", "\t"]:
        response = await client.get("/health", headers={"X-Request-ID": empty_val})
        assert response.status_code == 200

        header_id = response.headers.get("x-request-id")
        assert header_id is not None
        assert header_id.strip() != ""
        assert UUID4_PATTERN.match(header_id) is not None


@pytest.mark.anyio
async def test_excessively_long_request_id_is_replaced(client: httpx.AsyncClient) -> None:
    """Requirement 5: Request ID exceeding 64 characters is replaced with UUID4."""
    oversized_id = "a" * 65
    response = await client.get("/health", headers={"X-Request-ID": oversized_id})
    assert response.status_code == 200

    header_id = response.headers.get("x-request-id")
    assert header_id is not None
    assert header_id != oversized_id
    assert len(header_id) <= 64
    assert UUID4_PATTERN.match(header_id) is not None

    # Boundary test: exactly 64 characters should be accepted
    exact_64 = "a" * 64
    resp_64 = await client.get("/health", headers={"X-Request-ID": exact_64})
    assert resp_64.headers.get("x-request-id") == exact_64


# ------------------------------------------------------------------------------
# 2. Response Headers & Logging
# ------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_response_header_contains_effective_request_id(client: httpx.AsyncClient) -> None:
    """Requirement 6: Response headers contain X-Request-ID across multiple standard endpoints."""
    for path in ["/health", "/ready", "/api/v1/", "/"]:
        resp = await client.get(path)
        assert resp.status_code == 200
        req_id = resp.headers.get("x-request-id")
        assert req_id is not None
        assert is_valid_request_id(req_id) is True


@pytest.mark.anyio
async def test_application_logs_contain_correct_request_id(
    app, caplog: pytest.LogCaptureFixture
) -> None:
    """Requirement 7: Application logs emitted during request processing contain request ID."""
    test_logger = logging.getLogger("lungcare.test")

    # Add a temporary test route that logs an informative message
    @app.get("/test-log-correlation")
    async def log_route():
        test_logger.info("Test correlation log entry during request")
        return {"status": "logged"}

    custom_id = "test-log-id-42"
    with caplog.at_level(logging.INFO):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as c:
            response = await c.get("/test-log-correlation", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id

    # Verify log record has request_id attached and formatted
    matching_records = [r for r in caplog.records if "Test correlation log entry" in r.message]
    assert len(matching_records) == 1
    assert getattr(matching_records[0], "request_id", None) == custom_id


def test_outside_http_context_uses_fallback_id() -> None:
    """Requirement 8: Code running outside HTTP request context uses fallback ID '-'."""
    current_id = get_request_id()
    assert current_id == FALLBACK_REQUEST_ID


# ------------------------------------------------------------------------------
# 3. Concurrency & Isolation
# ------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_concurrent_requests_do_not_leak_ids_into_one_another(app) -> None:
    """Requirement 9: Concurrent requests preserve strict context isolation without leaking."""
    release_barrier = asyncio.Event()

    @app.get("/test-concurrent-correlation")
    async def concurrent_route(idx: int):
        req_id = get_request_id()
        # Wait on barrier to ensure all tasks overlap in execution
        await release_barrier.wait()
        req_id_after = get_request_id()
        return {"idx": idx, "req_id": req_id, "req_id_after": req_id_after}

    num_requests = 10
    client_ids = [f"concurrent-id-{i:03d}" for i in range(num_requests)]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as c:
        async def fetch(index: int, custom_id: str):
            resp = await c.get(
                f"/test-concurrent-correlation?idx={index}",
                headers={"X-Request-ID": custom_id},
            )
            return resp

        # Launch all concurrent coroutines
        tasks = [asyncio.create_task(fetch(i, client_ids[i])) for i in range(num_requests)]

        # Let all coroutines enter the endpoint and wait on barrier
        await asyncio.sleep(0.05)
        release_barrier.set()

        responses = await asyncio.gather(*tasks)

    for i, resp in enumerate(responses):
        assert resp.status_code == 200
        header_id = resp.headers.get("x-request-id")
        body = resp.json()

        assert header_id == client_ids[i]
        assert body["req_id"] == client_ids[i]
        assert body["req_id_after"] == client_ids[i]


@pytest.mark.anyio
async def test_contextvar_is_reset_after_request_completion(client: httpx.AsyncClient) -> None:
    """Requirement 10: ContextVar is guaranteed to be restored to '-' after request finishes."""
    assert get_request_id() == FALLBACK_REQUEST_ID

    resp = await client.get("/health", headers={"X-Request-ID": "test-reset-123"})
    assert resp.status_code == 200

    # Outside the request, contextvar must be reset
    assert get_request_id() == FALLBACK_REQUEST_ID


# ------------------------------------------------------------------------------
# 4. Error Response Integration
# ------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_exception_response_preserves_request_id(app) -> None:
    """Requirement 11: 404 and 500 error responses retain request ID in headers and JSON envelope."""
    @app.get("/test-unhandled-500")
    async def err_route():
        raise RuntimeError("Injected runtime failure for correlation verification")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        # Case A: 404 Not Found
        custom_404_id = "req-error-404-check"
        resp_404 = await c.get("/non-existent-route-404", headers={"X-Request-ID": custom_404_id})
        assert resp_404.status_code == 404
        assert resp_404.headers.get("x-request-id") == custom_404_id
        data_404 = resp_404.json()
        assert data_404["status"] == "error"
        assert data_404.get("request_id") == custom_404_id

        # Case B: 500 Internal Server Error
        custom_500_id = "req-error-500-check"
        resp_500 = await c.get("/test-unhandled-500", headers={"X-Request-ID": custom_500_id})
        assert resp_500.status_code == 500
        assert resp_500.headers.get("x-request-id") == custom_500_id
        data_500 = resp_500.json()
        assert data_500["status"] == "error"
        assert data_500.get("request_id") == custom_500_id


@pytest.mark.anyio
async def test_validation_error_response_preserves_request_id(client: httpx.AsyncClient) -> None:
    """Requirement 12: 422 validation errors retain request ID in headers and JSON envelope."""
    custom_422_id = "req-val-422-check"

    # Send invalid POST to predictions endpoint (missing multipart file)
    resp = await client.post(
        "/api/v1/predictions",
        headers={"X-Request-ID": custom_422_id},
    )
    assert resp.status_code == 422
    assert resp.headers.get("x-request-id") == custom_422_id

    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["error_type"] == "ValidationError"
    assert data.get("request_id") == custom_422_id
