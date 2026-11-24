"""Test suite for Health, Readiness, OpenAPI documentation, and API metadata.

Uses native httpx.AsyncClient with ASGITransport to ensure non-blocking, modern
ASGI testing without deprecated test-client dependencies.
"""

from __future__ import annotations

from typing import AsyncGenerator
import pytest
try:
    import httpx
    from app.core.logging import LOG_FORMAT
    from app.main import app
    HAS_BACKEND_DEPS = True
except ImportError:
    httpx = None  # type: ignore
    app = None  # type: ignore
    LOG_FORMAT = ""  # type: ignore
    HAS_BACKEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_BACKEND_DEPS,
    reason="FastAPI backend dependencies required (execute using host backend environment)",
)


@pytest.fixture
def anyio_backend() -> str:
    """Specify asyncio as the AnyIO backend for async tests."""
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Fixture providing a modern, async ASGI test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as test_client:
        yield test_client


@pytest.mark.anyio
async def test_health_returns_200(client: httpx.AsyncClient) -> None:
    """Requirement 1: GET /health returns HTTP 200."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_health_response_contract(client: httpx.AsyncClient) -> None:
    """Requirement 2: Health response contains expected service information."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "lungcare-api"
    assert data["version"] == "1.0.0"


@pytest.mark.anyio
async def test_ready_returns_200(client: httpx.AsyncClient) -> None:
    """Requirement 3: GET /ready returns HTTP 200."""
    response = await client.get("/ready")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_readiness_does_not_claim_model_loaded(client: httpx.AsyncClient) -> None:
    """Requirement 4: Readiness response does NOT claim that the ML model is loaded."""
    response = await client.get("/ready")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ready"
    assert data["service"] == "lungcare-api"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is False
    assert "unloaded" in str(data.get("dependencies", {}).get("ml_model", "")).lower()


@pytest.mark.anyio
async def test_openapi_docs_available_in_development(client: httpx.AsyncClient) -> None:
    """Requirement 5: OpenAPI documentation endpoint is available in development."""
    # Test swagger UI endpoint
    docs_response = await client.get("/docs")
    assert docs_response.status_code == 200
    assert "swagger-ui" in docs_response.text.lower() or "html" in docs_response.headers.get("content-type", "")

    # Test openapi schema JSON endpoint at versioned route
    openapi_response = await client.get("/api/v1/openapi.json")
    assert openapi_response.status_code == 200
    schema = openapi_response.json()
    assert schema["info"]["title"] == "LungCare AI API"
    assert schema["info"]["version"] == "1.0.0"


@pytest.mark.anyio
async def test_openapi_root_alias_consistency(client: httpx.AsyncClient) -> None:
    """Remediation 1: Verify root /openapi.json alias exists and returns identical schema."""
    root_openapi_resp = await client.get("/openapi.json")
    assert root_openapi_resp.status_code == 200

    v1_openapi_resp = await client.get("/api/v1/openapi.json")
    assert v1_openapi_resp.status_code == 200

    root_schema = root_openapi_resp.json()
    v1_schema = v1_openapi_resp.json()

    # Schemas must be completely identical
    assert root_schema == v1_schema
    assert root_schema["info"]["title"] == "LungCare AI API"
    assert root_schema["info"]["version"] == "1.0.0"


@pytest.mark.anyio
async def test_api_v1_versioned_probes(client: httpx.AsyncClient) -> None:
    """Verify that versioned API v1 endpoints also provide health & readiness probes."""
    # /api/v1/health
    health_v1 = await client.get("/api/v1/health")
    assert health_v1.status_code == 200
    assert health_v1.json()["status"] == "healthy"

    # /api/v1/ready
    ready_v1 = await client.get("/api/v1/ready")
    assert ready_v1.status_code == 200
    assert ready_v1.json()["model_loaded"] is False


@pytest.mark.anyio
async def test_api_v1_root_metadata(client: httpx.AsyncClient) -> None:
    """Remediation 5: Verify GET /api/v1/ returns versioned metadata."""
    response = await client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()

    assert data["app_name"] == "LungCare AI API"
    assert data["version"] == "1.0.0"
    assert data["api_version"] == "v1"
    assert data["health_url"] == "/api/v1/health"
    assert data["ready_url"] == "/api/v1/ready"
    assert data["openapi_url"] == "/api/v1/openapi.json"
    assert "health" in data["endpoints"]
    assert "ready" in data["endpoints"]


@pytest.mark.anyio
async def test_root_endpoint_metadata(client: httpx.AsyncClient) -> None:
    """Verify root endpoint provides discoverability information and links to alias."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "LungCare AI API"
    assert data["version"] == "1.0.0"
    assert data["health_url"] == "/health"
    assert data["ready_url"] == "/ready"
    assert data["openapi_alias_url"] == "/openapi.json"


@pytest.mark.anyio
async def test_cors_headers_enforced(client: httpx.AsyncClient) -> None:
    """Verify CORS middleware permits configured origins and rejects wildcards."""
    headers = {"Origin": "http://localhost:3000"}
    response = await client.get("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.anyio
async def test_404_error_envelope(client: httpx.AsyncClient) -> None:
    """Verify unhandled paths return consistent error schema without exposing internal details."""
    response = await client.get("/non-existent-path")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert "error" in data
    assert "message" in data["error"]
    assert data["error"]["error_type"] == "HTTPException"


def test_logging_precision_format() -> None:
    """Remediation 4: Verify log format includes millisecond precision."""
    assert "%(msecs)03d" in LOG_FORMAT


class TestReadinessLifecycleContracts:
    """Deterministic contract tests for all lifecycle states of the readiness probe."""

    @pytest.mark.anyio
    async def test_readiness_state_ready(self, client: httpx.AsyncClient) -> None:
        """READY: HTTP 200, status='ready', model_loaded=True, ReadinessResponse schema."""
        from app.services.model_loader import get_model_loader
        loader = get_model_loader()
        orig_loaded = loader._is_loaded
        orig_error = loader._load_error
        try:
            loader._is_loaded = True
            loader._load_error = None
            loader._coordinator.complete_loading()

            resp = await client.get("/ready", headers={"X-Request-ID": "ready-test-req-200"})
            assert resp.status_code == 200
            assert resp.headers.get("x-request-id") == "ready-test-req-200"

            data = resp.json()
            assert data["status"] == "ready"
            assert data["service"] == "lungcare-api"
            assert data["version"] == "1.0.0"
            assert data["model_loaded"] is True
            assert data["dependencies"]["ml_model"] == "loaded"
            assert data["dependencies"]["database"] == "not_configured"
            assert "timestamp" in data
        finally:
            loader._is_loaded = orig_loaded
            loader._load_error = orig_error

    @pytest.mark.anyio
    async def test_readiness_state_error(self, client: httpx.AsyncClient) -> None:
        """ERROR: HTTP 503, status='error', error_type='MODEL_NOT_READY', ErrorResponse schema."""
        from app.services.model_loader import get_model_loader
        loader = get_model_loader()
        orig_loaded = loader._is_loaded
        orig_error = loader._load_error
        try:
            loader._is_loaded = False
            loader._load_error = RuntimeError("Artifact checksum mismatch or corrupt weights at /var/models/weights.h5")

            resp = await client.get("/ready", headers={"X-Request-ID": "ready-test-req-503-err"})
            assert resp.status_code == 503
            assert resp.headers.get("x-request-id") == "ready-test-req-503-err"

            data = resp.json()
            assert data["status"] == "error"
            assert "error" in data
            assert data["error"]["error_type"] == "MODEL_NOT_READY"
            assert "not ready" in data["error"]["message"].lower() or "failed" in data["error"]["message"].lower()
            assert data["error"]["details"]["lifecycle_state"] == "ERROR"
            assert data["error"]["details"]["dependencies"]["ml_model"] == "unloaded (failed to load: model unavailable)"
            assert data["request_id"] == "ready-test-req-503-err"
            assert "timestamp" in data

            # Verify no internal sensitive path leakage
            assert "/var/models/weights.h5" not in str(data)
            assert "Traceback" not in str(data)
        finally:
            loader._is_loaded = orig_loaded
            loader._load_error = orig_error

    @pytest.mark.anyio
    async def test_readiness_state_draining(self, client: httpx.AsyncClient) -> None:
        """DRAINING: HTTP 503, status='error', error_type='SERVICE_DRAINING', ErrorResponse schema."""
        from app.services.model_loader import get_model_loader
        loader = get_model_loader()
        coordinator = loader.coordinator
        orig_state = coordinator.state
        try:
            coordinator.start_draining()

            resp = await client.get("/ready", headers={"X-Request-ID": "ready-test-req-503-drain"})
            assert resp.status_code == 503
            assert resp.headers.get("x-request-id") == "ready-test-req-503-drain"

            data = resp.json()
            assert data["status"] == "error"
            assert data["error"]["error_type"] == "SERVICE_DRAINING"
            assert "shutting down" in data["error"]["message"].lower() or "draining" in data["error"]["message"].lower()
            assert data["error"]["details"]["lifecycle_state"] == "DRAINING"
            assert "draining" in data["error"]["details"]["dependencies"]["ml_model"]
            assert data["request_id"] == "ready-test-req-503-drain"
        finally:
            coordinator._state = orig_state

    @pytest.mark.anyio
    async def test_readiness_state_pending_unloaded(self, client: httpx.AsyncClient) -> None:
        """UNLOADED/pending: HTTP 200, status='ready', model_loaded=False, ReadinessResponse schema."""
        from app.services.model_loader import get_model_loader
        loader = get_model_loader()
        orig_loaded = loader._is_loaded
        orig_error = loader._load_error
        try:
            loader._is_loaded = False
            loader._load_error = None

            resp = await client.get("/ready", headers={"X-Request-ID": "ready-test-req-200-pending"})
            assert resp.status_code == 200
            assert resp.headers.get("x-request-id") == "ready-test-req-200-pending"

            data = resp.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is False
            assert "unloaded (model initialization pending or disabled)" in data["dependencies"]["ml_model"]
            assert data["service"] == "lungcare-api"
        finally:
            loader._is_loaded = orig_loaded
            loader._load_error = orig_error

    @pytest.mark.anyio
    async def test_versioned_readiness_probe_parity(self, client: httpx.AsyncClient) -> None:
        """Verify parity between /ready and /api/v1/ready across success and error states."""
        # 1. Parity in UNLOADED / pending state
        resp_root = await client.get("/ready")
        resp_v1 = await client.get("/api/v1/ready")

        assert resp_root.status_code == resp_v1.status_code == 200
        data_root = resp_root.json()
        data_v1 = resp_v1.json()

        assert data_root["status"] == data_v1["status"] == "ready"
        assert data_root["model_loaded"] == data_v1["model_loaded"] == False
        assert data_root["service"] == data_v1["service"]
        assert data_root["dependencies"] == data_v1["dependencies"]

        # 2. Parity in ERROR state
        from app.services.model_loader import get_model_loader
        loader = get_model_loader()
        orig_loaded = loader._is_loaded
        orig_error = loader._load_error
        try:
            loader._is_loaded = False
            loader._load_error = RuntimeError("Artifact checksum error")

            err_root = await client.get("/ready", headers={"X-Request-ID": "parity-err-root"})
            err_v1 = await client.get("/api/v1/ready", headers={"X-Request-ID": "parity-err-v1"})

            assert err_root.status_code == err_v1.status_code == 503
            err_data_root = err_root.json()
            err_data_v1 = err_v1.json()

            assert err_data_root["status"] == err_data_v1["status"] == "error"
            assert err_data_root["error"]["error_type"] == err_data_v1["error"]["error_type"] == "MODEL_NOT_READY"
            assert err_root.headers.get("x-request-id") == err_data_root["request_id"] == "parity-err-root"
            assert err_v1.headers.get("x-request-id") == err_data_v1["request_id"] == "parity-err-v1"
        finally:
            loader._is_loaded = orig_loaded
            loader._load_error = orig_error

