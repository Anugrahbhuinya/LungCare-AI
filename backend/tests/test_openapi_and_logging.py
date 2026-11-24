"""Integration and contract test suite for OpenAPI Specification & Access Logging.

Validates Phase 2.3.4.4 requirements:
1. Generated OpenAPI schema defines explicit ErrorResponse models for HTTP 400, 413, 415, 422, 500, 502, 503.
2. HTTP 422 schema references ErrorResponse (eliminating default HTTPValidationError).
3. Components include ErrorResponse and ErrorDetail.
4. /health, /ready, /, and /api/v1/ routes have appropriate probe error schemas.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import pytest

try:
    import httpx
    from app.api.v1.endpoints.predictions import (
        get_gradcam_service_dep,
        get_gradcam_visualizer_dep,
        get_image_preprocessor_dep,
        get_inference_service_dep,
    )
    from app.main import app, create_application
    from app.middleware.access_logging import AccessLoggingMiddleware
    from app.schemas.error import ErrorResponse
    from app.schemas.gradcam import TARGET_CONV_LAYER, GradCAMResult, RESEARCH_DISCLAIMER
    from app.services.inference_service import PredictionResult
    from app.services.gradcam_visualizer import VisualizerResult
    from app.services.image_preprocessor import PreprocessedImageResult
    HAS_BACKEND_DEPS = True
except ImportError:
    HAS_BACKEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_BACKEND_DEPS,
    reason="FastAPI backend dependencies required",
)


class TestOpenAPIHarmonization:
    """Validate that OpenAPI schema reflects the unified error handling architecture."""

    def test_components_contain_error_response_models(self) -> None:
        """Verify ErrorResponse and ErrorDetail are properly registered in components/schemas."""
        schema = app.openapi()
        components = schema.get("components", {}).get("schemas", {})

        assert "ErrorResponse" in components
        assert "ErrorDetail" in components

        # Verify ErrorResponse structure in components
        err_resp_schema = components["ErrorResponse"]
        assert "properties" in err_resp_schema
        assert "status" in err_resp_schema["properties"]
        assert "error" in err_resp_schema["properties"]
        assert "timestamp" in err_resp_schema["properties"]
        assert "request_id" in err_resp_schema["properties"]

    def test_predictions_endpoint_documents_all_domain_error_responses(self) -> None:
        """Verify POST /api/v1/predictions documents 400, 413, 415, 422, 500, 502, 503 as ErrorResponse."""
        schema = app.openapi()
        responses = schema["paths"]["/api/v1/predictions"]["post"]["responses"]

        expected_error_codes = ["400", "413", "415", "422", "500", "502", "503"]
        for code in expected_error_codes:
            assert code in responses, f"Response code {code} missing from /predictions"
            content = responses[code]["content"]["application/json"]
            schema_ref = content["schema"]["$ref"]
            assert schema_ref == "#/components/schemas/ErrorResponse", (
                f"Response code {code} references {schema_ref} instead of ErrorResponse"
            )

    def test_explain_endpoint_documents_all_domain_error_responses(self) -> None:
        """Verify POST /api/v1/predictions/explain documents 400, 413, 415, 422, 500, 502, 503 as ErrorResponse."""
        schema = app.openapi()
        responses = schema["paths"]["/api/v1/predictions/explain"]["post"]["responses"]

        expected_error_codes = ["400", "413", "415", "422", "500", "502", "503"]
        for code in expected_error_codes:
            assert code in responses, f"Response code {code} missing from /predictions/explain"
            content = responses[code]["content"]["application/json"]
            schema_ref = content["schema"]["$ref"]
            assert schema_ref == "#/components/schemas/ErrorResponse", (
                f"Response code {code} references {schema_ref} instead of ErrorResponse"
            )

    def test_validation_error_schema_replaced_with_error_response(self) -> None:
        """Verify that HTTP 422 does NOT reference the default HTTPValidationError."""
        schema = app.openapi()

        pred_422 = schema["paths"]["/api/v1/predictions"]["post"]["responses"]["422"]
        assert pred_422["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"

        exp_422 = schema["paths"]["/api/v1/predictions/explain"]["post"]["responses"]["422"]
        assert exp_422["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"

    def test_probe_endpoints_document_error_responses(self) -> None:
        """Verify /health and /ready probe routes document appropriate error models."""
        schema = app.openapi()

        # Health probe
        health_resp = schema["paths"]["/health"]["get"]["responses"]
        assert "200" in health_resp
        assert "500" in health_resp
        assert health_resp["500"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"

        # Ready probe
        ready_resp = schema["paths"]["/ready"]["get"]["responses"]
        assert "200" in ready_resp
        assert "500" in ready_resp
        assert "503" in ready_resp
        assert ready_resp["503"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"


# ==============================================================================
# Helper Doubles & Fixtures for Explain Endpoint Verification
# ==============================================================================

def create_synthetic_png_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    """Generate in-memory synthetic PNG bytes."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockExplainInferenceService:
    def predict_tensor(self, tensor: np.ndarray, **kwargs: Any) -> PredictionResult:
        return PredictionResult(
            predicted_class="COVID19",
            predicted_label="COVID-19",
            confidence=0.95,
            probabilities={"COVID19": 0.95, "NORMAL": 0.02, "PNEUMONIA": 0.02, "TURBERCULOSIS": 0.01},
            model_name="LungCare DenseNet121",
            model_architecture="DenseNet121",
            input_width=100,
            input_height=100,
            processed_width=224,
            processed_height=224,
            inference_time_ms=30.0,
            preprocessing_time_ms=10.0,
            disclaimer=RESEARCH_DISCLAIMER,
        )


class MockExplainGradCAMService:
    def generate_heatmap(self, image_tensor: np.ndarray, class_index: int, **kwargs: Any) -> GradCAMResult:
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        classes = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
        selected = classes[class_index] if 0 <= class_index < 4 else "UNKNOWN"
        return GradCAMResult(
            selected_class=selected,
            selected_label=selected,
            class_index=class_index,
            target_layer=TARGET_CONV_LAYER,
            heatmap=heatmap,
            heatmap_width=224,
            heatmap_height=224,
            status="success",
            confidence=0.95,
            probabilities={"COVID19": 0.95, "NORMAL": 0.02, "PNEUMONIA": 0.02, "TURBERCULOSIS": 0.01},
            latency_ms=40.0,
        )


class MockExplainVisualizer:
    def create_overlay(self, original_image: Any, heatmap: np.ndarray, alpha: Optional[float] = None, **kwargs: Any) -> VisualizerResult:
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        colored = np.zeros((100, 100, 3), dtype=np.uint8)
        return VisualizerResult(
            overlay_array=overlay,
            colored_heatmap=colored,
            original_width=100,
            original_height=100,
            alpha=alpha if alpha is not None else 0.4,
            colormap="jet",
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.fixture
async def explain_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    app.dependency_overrides[get_inference_service_dep] = lambda: MockExplainInferenceService()
    app.dependency_overrides[get_gradcam_service_dep] = lambda: MockExplainGradCAMService()
    app.dependency_overrides[get_gradcam_visualizer_dep] = lambda: MockExplainVisualizer()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c

    app.dependency_overrides.clear()


# ==============================================================================
# Parameter Deduplication and Backward Compatibility Tests
# ==============================================================================

class TestExplainParameterDeduplicationAndCompatibility:
    """Verify that explain endpoint parameters are deduplicated in OpenAPI while preserving query fallback."""

    def test_openapi_has_no_duplicate_query_parameters_for_explain(self) -> None:
        """Requirement: OpenAPI parameters contains NO query parameters for target_class or alpha."""
        schema = app.openapi()
        explain_op = schema["paths"]["/api/v1/predictions/explain"]["post"]

        # parameters list should not contain target_class or alpha query parameters
        params = explain_op.get("parameters", [])
        query_param_names = [p["name"] for p in params if p.get("in") == "query"]
        assert "target_class" not in query_param_names, "target_class query parameter unexpectedly found in OpenAPI"
        assert "alpha" not in query_param_names, "alpha query parameter unexpectedly found in OpenAPI"

        # Form body properties should declare target_class and alpha
        form_schema = explain_op["requestBody"]["content"]["multipart/form-data"]["schema"]
        # Resolve $ref if present
        if "$ref" in form_schema:
            ref_name = form_schema["$ref"].split("/")[-1]
            props = schema["components"]["schemas"][ref_name]["properties"]
        else:
            props = form_schema.get("properties", {})

        assert "image" in props
        assert "target_class" in props
        assert "alpha" in props

    @pytest.mark.anyio
    async def test_multipart_form_target_class_and_alpha(self, explain_client: httpx.AsyncClient) -> None:
        """Requirement 1 & 2: Multipart form target_class and alpha work correctly."""
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}
        data = {"target_class": "PNEUMONIA", "alpha": "0.75"}

        response = await explain_client.post("/api/v1/predictions/explain", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        assert payload["selected_target_class"] == "PNEUMONIA"
        assert payload["target_class_index"] == 2
        assert np.isclose(payload["explanation"]["alpha"], 0.75)

    @pytest.mark.anyio
    async def test_query_string_target_class_and_alpha(self, explain_client: httpx.AsyncClient) -> None:
        """Requirement 3 & 4: Query-string target_class and alpha work as fallback."""
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}

        response = await explain_client.post(
            "/api/v1/predictions/explain?target_class=NORMAL&alpha=0.6", files=files
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["selected_target_class"] == "NORMAL"
        assert payload["target_class_index"] == 1
        assert np.isclose(payload["explanation"]["alpha"], 0.6)

    @pytest.mark.anyio
    async def test_form_values_take_precedence_over_query(self, explain_client: httpx.AsyncClient) -> None:
        """Requirement 5: Form values take precedence when both form and query values are supplied."""
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}
        # Query says NORMAL / 0.2, Form says TURBERCULOSIS / 0.85
        form_data = {"target_class": "TURBERCULOSIS", "alpha": "0.85"}

        response = await explain_client.post(
            "/api/v1/predictions/explain?target_class=NORMAL&alpha=0.2",
            files=files,
            data=form_data,
        )
        assert response.status_code == 200
        payload = response.json()
        # Form values must win
        assert payload["selected_target_class"] == "TURBERCULOSIS"
        assert payload["target_class_index"] == 3
        assert np.isclose(payload["explanation"]["alpha"], 0.85)

    @pytest.mark.anyio
    async def test_invalid_target_class_returns_error_response(self, explain_client: httpx.AsyncClient) -> None:
        """Requirement 6: Invalid target_class via form or query returns ErrorResponse with INVALID_GRADCAM_INPUT."""
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}

        # Form invalid
        resp_form = await explain_client.post(
            "/api/v1/predictions/explain", files=files, data={"target_class": "INVALID_CLASS"}
        )
        assert resp_form.status_code == 400
        data_form = resp_form.json()
        assert data_form["status"] == "error"
        assert data_form["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

        # Query invalid
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}
        resp_query = await explain_client.post(
            "/api/v1/predictions/explain?target_class=UNKNOWN_DISEASE", files=files
        )
        assert resp_query.status_code == 400
        data_query = resp_query.json()
        assert data_query["status"] == "error"
        assert data_query["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

    @pytest.mark.anyio
    async def test_invalid_alpha_returns_error_response(self, explain_client: httpx.AsyncClient) -> None:
        """Requirement 6: Invalid alpha via form or query returns ErrorResponse with INVALID_GRADCAM_INPUT."""
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}

        # Form invalid (out of bounds)
        resp_form = await explain_client.post(
            "/api/v1/predictions/explain", files=files, data={"alpha": "2.5"}
        )
        assert resp_form.status_code == 400
        data_form = resp_form.json()
        assert data_form["status"] == "error"
        assert data_form["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

        # Query invalid (out of bounds)
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}
        resp_query = await explain_client.post(
            "/api/v1/predictions/explain?alpha=-0.5", files=files
        )
        assert resp_query.status_code == 400
        data_query = resp_query.json()
        assert data_query["status"] == "error"
        assert data_query["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

        # Query invalid (non-numeric string)
        files = {"image": ("xray.png", create_synthetic_png_bytes(), "image/png")}
        resp_nan = await explain_client.post(
            "/api/v1/predictions/explain?alpha=not_a_number", files=files
        )
        assert resp_nan.status_code == 400
        data_nan = resp_nan.json()
        assert data_nan["status"] == "error"
        assert data_nan["error"]["error_type"] == "INVALID_GRADCAM_INPUT"


# ==============================================================================
# Structured HTTP Access Logging Tests
# ==============================================================================

class TestStructuredAccessLogging:
    """Verify pure ASGI AccessLoggingMiddleware emits compliant, privacy-preserving telemetry."""

    @pytest.mark.anyio
    async def test_successful_request_logs_access_event(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 1: 200 OK produces structured access log with all required fields."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        response = await client.get("/health", headers={"X-Request-ID": "access-req-200"})
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == "access-req-200"

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1, f"Expected exactly 1 access log, got {len(records)}"

        record = records[0]
        assert "HTTP GET /health -> 200" in record.message
        assert getattr(record, "http_method", None) == "GET"
        assert getattr(record, "request_path", None) == "/health"
        assert getattr(record, "status_code", None) == 200
        assert getattr(record, "duration_ms", None) >= 0.0
        assert getattr(record, "request_id", None) == "access-req-200"

    @pytest.mark.anyio
    async def test_4xx_client_error_logs_access_event(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 2: 404/422/400 errors produce structured access log with status and request ID."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        response = await client.get("/non-existent-endpoint", headers={"X-Request-ID": "access-req-404"})
        assert response.status_code == 404

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1

        record = records[0]
        assert "HTTP GET /non-existent-endpoint -> 404" in record.message
        assert getattr(record, "status_code", None) == 404
        assert getattr(record, "request_id", None) == "access-req-404"
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_5xx_server_error_logs_access_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 3: Handled 500 internal errors produce structured access log with status 500."""
        from fastapi import HTTPException

        caplog.set_level(logging.INFO, logger="lungcare.access")

        custom_app = create_application()

        @custom_app.get("/test-internal-500")
        async def internal_500_endpoint():
            raise HTTPException(status_code=500, detail="Database failure.")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-internal-500", headers={"X-Request-ID": "access-req-500"})

        assert response.status_code == 500

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1

        record = records[0]
        assert "HTTP GET /test-internal-500 -> 500" in record.message
        assert getattr(record, "status_code", None) == 500
        assert getattr(record, "request_id", None) == "access-req-500"
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_unhandled_exception_logs_no_response_without_synthetic_500(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 3b: Unhandled exception escaping before response start logs NO_RESPONSE with status None."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        custom_app = create_application()

        @custom_app.get("/test-internal-crash")
        async def crash_endpoint():
            raise RuntimeError("Database connection exploded.")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-internal-crash", headers={"X-Request-ID": "access-req-crash"})

        assert response.status_code == 500

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1

        record = records[0]
        # Never logs synthetic 500 when no http.response.start was emitted before exit
        assert "HTTP GET /test-internal-crash -> NO_RESPONSE" in record.message
        assert getattr(record, "status_code", None) is None
        assert getattr(record, "request_id", None) == "access-req-crash"
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_access_log_excludes_sensitive_query_strings(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 4: Query parameters (e.g. tokens, patient IDs) are strictly excluded from access logs."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        secret_query = "patient_token=secret_12345&diagnosis=pneumonia_case"
        response = await client.get(f"/health?{secret_query}", headers={"X-Request-ID": "access-req-sec"})
        assert response.status_code == 200

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1

        record = records[0]
        # Pure path must be logged without query string
        assert getattr(record, "request_path", None) == "/health"
        assert "secret_12345" not in record.message
        assert "pneumonia_case" not in record.message
        assert "patient_token" not in record.message

    @pytest.mark.anyio
    async def test_access_log_excludes_image_payloads(
        self, explain_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 5: Uploaded image bytes and base64 payloads are never captured in access logs."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        raw_bytes = create_synthetic_png_bytes()
        files = {"image": ("chest_xray_sensitive.png", raw_bytes, "image/png")}
        response = await explain_client.post(
            "/api/v1/predictions/explain", files=files, headers={"X-Request-ID": "access-req-img"}
        )
        assert response.status_code == 200

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1

        record = records[0]
        assert getattr(record, "request_path", None) == "/api/v1/predictions/explain"
        # Zero leakage of binary data, filenames, or Base64 responses
        assert "chest_xray_sensitive.png" not in record.message
        assert "data:image/png;base64" not in record.message
        assert str(raw_bytes[:20]) not in record.message

    @pytest.mark.anyio
    async def test_streaming_response_compatibility(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 6: Streaming responses flow through without body buffering."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        custom_app = create_application()

        async def stream_generator():
            for i in range(5):
                yield f"chunk_{i}\n".encode("utf-8")

        @custom_app.get("/test-streaming")
        async def streaming_endpoint():
            return StreamingResponse(stream_generator(), media_type="text/plain")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-streaming", headers={"X-Request-ID": "access-req-stream"})

        assert response.status_code == 200
        assert "chunk_0\nchunk_1\nchunk_2\nchunk_3\nchunk_4\n" in response.text

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1
        record = records[0]
        assert getattr(record, "status_code", None) == 200
        assert getattr(record, "request_path", None) == "/test-streaming"
        assert getattr(record, "request_id", None) == "access-req-stream"
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_non_http_asgi_scope_passes_through_without_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 7: Non-HTTP scopes (e.g. lifespan, websocket) pass through with zero access logs."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        called = False

        async def dummy_app(scope, receive, send):
            nonlocal called
            called = True

        middleware = AccessLoggingMiddleware(dummy_app)
        scope = {"type": "lifespan"}

        await middleware(scope, None, None)
        assert called is True

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 0, "Non-HTTP scope must not emit access logs"

    @pytest.mark.anyio
    async def test_cancelled_request_before_response_start_logs_cancelled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cancellation before http.response.start logs CANCELLED and status_code None (no synthetic 500)."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        async def cancelling_app(scope, receive, send):
            raise asyncio.CancelledError()

        middleware = AccessLoggingMiddleware(cancelling_app)
        scope = {"type": "http", "method": "GET", "path": "/test-cancel", "headers": []}

        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, None, None)

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1, "Must log exactly once on cancellation"
        record = records[0]
        assert "HTTP GET /test-cancel -> CANCELLED" in record.message
        assert getattr(record, "status_code", None) is None
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_client_disconnect_before_response_start_logs_cancelled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ClientDisconnect before response start logs CANCELLED without synthetic 500."""
        from starlette.requests import ClientDisconnect

        caplog.set_level(logging.INFO, logger="lungcare.access")

        async def disconnect_app(scope, receive, send):
            raise ClientDisconnect()

        middleware = AccessLoggingMiddleware(disconnect_app)
        scope = {"type": "http", "method": "POST", "path": "/test-disconnect", "headers": []}

        with pytest.raises(ClientDisconnect):
            await middleware(scope, None, None)

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1, "Must log exactly once on client disconnect"
        record = records[0]
        assert "HTTP POST /test-disconnect -> CANCELLED" in record.message
        assert getattr(record, "status_code", None) is None
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_cancelled_request_after_response_start_preserves_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cancellation during streaming after http.response.start preserves actual HTTP status."""
        caplog.set_level(logging.INFO, logger="lungcare.access")

        sent_messages = []

        async def dummy_send(message):
            sent_messages.append(message)

        async def stream_and_cancel_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            raise asyncio.CancelledError()

        middleware = AccessLoggingMiddleware(stream_and_cancel_app)
        scope = {"type": "http", "method": "GET", "path": "/test-stream-cancel", "headers": []}

        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, None, dummy_send)

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1, "Must log exactly once on cancellation during streaming"
        record = records[0]
        assert "HTTP GET /test-stream-cancel -> 200" in record.message
        assert getattr(record, "status_code", None) == 200
        assert getattr(record, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_exactly_once_logging_for_various_http_outcomes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify exactly one access log is emitted per request across 200, 404, 422, handled 500, and unhandled."""
        from fastapi import HTTPException

        custom_app = create_application()

        @custom_app.get("/handled-500")
        async def handled_500():
            raise HTTPException(status_code=500, detail="Handled server error")

        @custom_app.get("/unhandled-crash")
        async def unhandled_crash():
            raise RuntimeError("Unexpected failure")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            # 1. Successful request (200)
            caplog.clear()
            caplog.set_level(logging.INFO, logger="lungcare.access")
            resp_200 = await c.get("/health")
            assert resp_200.status_code == 200
            records_200 = [r for r in caplog.records if r.name == "lungcare.access"]
            assert len(records_200) == 1
            assert getattr(records_200[0], "status_code", None) == 200

            # 2. Handled client error (404)
            caplog.clear()
            resp_404 = await c.get("/api/v1/nonexistent-endpoint-path")
            assert resp_404.status_code == 404
            records_404 = [r for r in caplog.records if r.name == "lungcare.access"]
            assert len(records_404) == 1
            assert getattr(records_404[0], "status_code", None) == 404

            # 3. Validation error (422)
            caplog.clear()
            resp_422 = await c.post("/api/v1/predictions")
            assert resp_422.status_code == 422
            records_422 = [r for r in caplog.records if r.name == "lungcare.access"]
            assert len(records_422) == 1
            assert getattr(records_422[0], "status_code", None) == 422

            # 4. Handled server error (500)
            caplog.clear()
            resp_handled_500 = await c.get("/handled-500")
            assert resp_handled_500.status_code == 500
            records_h500 = [r for r in caplog.records if r.name == "lungcare.access"]
            assert len(records_h500) == 1
            assert getattr(records_h500[0], "status_code", None) == 500

            # 5. Unhandled server error (escapes before response started)
            caplog.clear()
            resp_unhandled = await c.get("/unhandled-crash")
            assert resp_unhandled.status_code == 500
            records_unhandled = [r for r in caplog.records if r.name == "lungcare.access"]
            assert len(records_unhandled) == 1
            # Explicit status representation: None (never a synthetic 500 when no response started)
            assert getattr(records_unhandled[0], "status_code", None) is None
            assert "NO_RESPONSE" in records_unhandled[0].message


class TestUvicornLoggingIntegration:
    """Validate uvicorn logger suppression and integration behavior."""

    def test_uvicorn_access_suppression_leaves_error_and_app_logging_functional(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Verify uvicorn.access is set to WARNING, while uvicorn.error and lungcare.access remain active."""
        from app.core.logging import setup_logging

        setup_logging(log_level="INFO")

        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_error = logging.getLogger("uvicorn.error")
        app_access = logging.getLogger("lungcare.access")

        assert uvicorn_access.level == logging.WARNING
        assert uvicorn_error.level == logging.INFO

        # Drain any earlier output
        capsys.readouterr()

        # uvicorn.access INFO should be suppressed
        uvicorn_access.info("GET /health HTTP/1.1 200")
        captured1 = capsys.readouterr()
        assert "GET /health HTTP/1.1 200" not in captured1.out, "uvicorn.access INFO must be suppressed"

        # uvicorn.error ERROR should be logged to console
        uvicorn_error.error("Uvicorn test error message")
        captured2 = capsys.readouterr()
        assert "Uvicorn test error message" in captured2.out, "uvicorn.error ERROR must be captured on console"

        # lungcare.access INFO should be logged to console
        app_access.info("HTTP GET /health -> 200 [1.23ms]")
        captured3 = capsys.readouterr()
        assert "HTTP GET /health -> 200 [1.23ms]" in captured3.out, "lungcare.access INFO must be captured on console"


