"""Comprehensive unit and integration test suite for Unified Error Handling Architecture.

Validates Phase 2.3.4.3 requirements:
1. Each image preprocessing exception maps to the correct domain error code and HTTP status.
2. Model-unavailable exceptions preserve their error identity (MODEL_UNAVAILABLE, 503).
3. Inference errors preserve their domain codes (INVALID_MODEL_OUTPUT -> 502, INFERENCE_EXECUTION_ERROR -> 500).
4. Grad-CAM errors preserve their domain codes (INVALID_GRADCAM_INPUT -> 400, GRADCAM_GRADIENT_ERROR -> 502, etc.).
5. ML coordinator errors preserve their domain codes (MODEL_NOT_READY -> 503).
6. Exception inheritance and subclass mapping correctness.
7. HTTPException responses remain backward-compatible (HTTPException, 404).
8. Validation errors retain established schema (ValidationError, 422).
9. Unhandled exceptions return sanitized production responses (InternalServerError, 500).
10. Every error response contains a matching request ID in body and header.
11. DEBUG=True vs DEBUG=False behavior.
12. Prediction and explain endpoints preserve domain error codes in real HTTP workflows.
"""

from __future__ import annotations

import io
from typing import Any, AsyncGenerator, Dict, Optional
import httpx
import pytest
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.endpoints.predictions import (
    get_gradcam_service_dep,
    get_gradcam_visualizer_dep,
    get_image_preprocessor_dep,
    get_inference_service_dep,
)
from app.core.config import Settings, get_settings
from app.main import app, create_application
from app.services.image_preprocessor import (
    CorruptImageError,
    DecompressionBombError,
    EmptyImageError,
    ImagePreprocessingError,
    ImageTooLargeError,
    InvalidImageModeError,
    TensorValidationError,
    UnsupportedFormatError,
)
from app.services.inference_service import (
    InferenceError,
    InferenceExecutionError,
    InvalidModelOutputError,
    ModelUnavailableError,
    PredictionResult,
)
from app.services.gradcam_service import (
    GradCAMError,
    GradCAMGenerationError,
    GradCAMGradientError,
    GradCAMLayerNotFoundError,
    InvalidGradCAMInputError,
)
from app.services.gradcam_visualizer import (
    InvalidVisualizerInputError,
    VisualizerError,
)
from app.services.ml_coordinator import (
    MLCoordinatorError,
    MLNotReadyError,
    ModelLifecycleState,
)
from app.services.model_loader import (
    ModelContractError,
    ModelNotLoadedError,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an async test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c


# ------------------------------------------------------------------------------
# 1. Image Preprocessing Domain Error Mapping
# ------------------------------------------------------------------------------

class TestImagePreprocessingErrorMapping:
    """Verify ImagePreprocessingError hierarchy maps to exact status and machine-readable error codes."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "exc,expected_status,expected_code",
        [
            (EmptyImageError("Uploaded image is empty."), 400, "EMPTY_IMAGE"),
            (UnsupportedFormatError("Format GIF not supported."), 415, "UNSUPPORTED_FORMAT"),
            (ImageTooLargeError("File exceeds 10MB limit."), 413, "IMAGE_TOO_LARGE"),
            (DecompressionBombError("Image exceeds 25M pixels."), 413, "DECOMPRESSION_BOMB"),
            (CorruptImageError("Cannot decode bytes."), 400, "CORRUPT_IMAGE"),
            (InvalidImageModeError("CMYK mode invalid."), 400, "INVALID_IMAGE_MODE"),
            (TensorValidationError("Tensor shape invalid."), 500, "TENSOR_VALIDATION_ERROR"),
            (ImagePreprocessingError("Generic preprocessing issue."), 400, "IMAGE_PREPROCESSING_ERROR"),
        ],
    )
    async def test_preprocessing_exceptions(
        self, client: httpx.AsyncClient, exc: ImagePreprocessingError, expected_status: int, expected_code: str
    ) -> None:
        custom_app = create_application()

        @custom_app.get("/test-prep-error")
        async def prep_route():
            raise exc

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-prep-error", headers={"X-Request-ID": "test-req-prep-1"})

        assert response.status_code == expected_status
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == expected_code
        assert response.headers.get("x-request-id") == "test-req-prep-1"
        assert data.get("request_id") == "test-req-prep-1"


# ------------------------------------------------------------------------------
# 2. Inference Domain Error Mapping
# ------------------------------------------------------------------------------

class TestInferenceErrorMapping:
    """Verify InferenceError hierarchy maps to exact status and machine-readable error codes."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "exc,expected_status,expected_code",
        [
            (ModelUnavailableError("Model not ready."), 503, "MODEL_UNAVAILABLE"),
            (InvalidModelOutputError("Output NaN detected."), 502, "INVALID_MODEL_OUTPUT"),
            (InferenceExecutionError("GPU kernel failure."), 500, "INFERENCE_EXECUTION_ERROR"),
            (InferenceError("Base inference fault."), 500, "INFERENCE_ERROR"),
        ],
    )
    async def test_inference_exceptions(
        self, client: httpx.AsyncClient, exc: InferenceError, expected_status: int, expected_code: str
    ) -> None:
        custom_app = create_application()

        @custom_app.get("/test-inf-error")
        async def inf_route():
            raise exc

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-inf-error", headers={"X-Request-ID": "test-req-inf-2"})

        assert response.status_code == expected_status
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == expected_code
        assert response.headers.get("x-request-id") == "test-req-inf-2"
        assert data.get("request_id") == "test-req-inf-2"


# ------------------------------------------------------------------------------
# 3. Grad-CAM Domain Error Mapping
# ------------------------------------------------------------------------------

class TestGradCAMErrorMapping:
    """Verify GradCAMError hierarchy maps to exact status and machine-readable error codes."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "exc,expected_status,expected_code",
        [
            (InvalidGradCAMInputError("Class index 5 out of range."), 400, "INVALID_GRADCAM_INPUT"),
            (GradCAMGradientError("Gradients contain NaN."), 502, "GRADCAM_GRADIENT_ERROR"),
            (GradCAMLayerNotFoundError("Layer convX not found."), 500, "GRADCAM_LAYER_NOT_FOUND"),
            (GradCAMGenerationError("Heatmap norm failed."), 500, "GRADCAM_GENERATION_ERROR"),
            (GradCAMError("Base GradCAM fault."), 500, "GRADCAM_ERROR"),
        ],
    )
    async def test_gradcam_exceptions(
        self, client: httpx.AsyncClient, exc: GradCAMError, expected_status: int, expected_code: str
    ) -> None:
        custom_app = create_application()

        @custom_app.get("/test-cam-error")
        async def cam_route():
            raise exc

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-cam-error", headers={"X-Request-ID": "test-req-cam-3"})

        assert response.status_code == expected_status
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == expected_code
        assert response.headers.get("x-request-id") == "test-req-cam-3"
        assert data.get("request_id") == "test-req-cam-3"


# ------------------------------------------------------------------------------
# 4. Visualizer & Coordinator Error Mapping
# ------------------------------------------------------------------------------

class TestVisualizerAndCoordinatorErrorMapping:
    """Verify VisualizerError and MLCoordinatorError map to exact status and codes."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "exc,expected_status,expected_code",
        [
            (InvalidVisualizerInputError("Heatmap shape mismatch."), 400, "INVALID_VISUALIZER_INPUT"),
            (VisualizerError("Alpha composite crashed."), 500, "VISUALIZER_ERROR"),
            (MLNotReadyError(ModelLifecycleState.DRAINING), 503, "MODEL_NOT_READY"),
            (MLCoordinatorError("Coordinator lock issue."), 503, "ML_COORDINATOR_ERROR"),
            (ModelNotLoadedError("Model not loaded yet."), 503, "MODEL_UNAVAILABLE"),
            (ModelContractError("Architecture violation."), 502, "MODEL_CONTRACT_ERROR"),
        ],
    )
    async def test_other_domain_exceptions(
        self, client: httpx.AsyncClient, exc: Exception, expected_status: int, expected_code: str
    ) -> None:
        custom_app = create_application()

        @custom_app.get("/test-other-error")
        async def other_route():
            raise exc

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-other-error", headers={"X-Request-ID": "test-req-other-4"})

        assert response.status_code == expected_status
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == expected_code
        assert response.headers.get("x-request-id") == "test-req-other-4"
        assert data.get("request_id") == "test-req-other-4"


# ------------------------------------------------------------------------------
# 5. HTTPException, Validation, and Production Sanitization
# ------------------------------------------------------------------------------

class TestFrameworkExceptionsAndSanitization:
    """Verify HTTPException, ValidationError, and unhandled 500 sanitization."""

    @pytest.mark.anyio
    async def test_standard_http_exception_backward_compatibility(self, client: httpx.AsyncClient) -> None:
        """Verify standard HTTPException (e.g. 404) returns error_type='HTTPException'."""
        response = await client.get("/nonexistent-endpoint-404-test", headers={"X-Request-ID": "req-404-compat"})
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "HTTPException"
        assert response.headers.get("x-request-id") == "req-404-compat"
        assert data.get("request_id") == "req-404-compat"

    @pytest.mark.anyio
    async def test_validation_error_preserves_established_schema(self, client: httpx.AsyncClient) -> None:
        """Verify 422 validation failure produces error_type='ValidationError'."""
        response = await client.post(
            "/api/v1/predictions",
            data={"wrong_field": "123"},
            headers={"X-Request-ID": "req-val-schema"},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "ValidationError"
        assert isinstance(data["error"]["details"], list)
        assert response.headers.get("x-request-id") == "req-val-schema"
        assert data.get("request_id") == "req-val-schema"

    @pytest.mark.anyio
    async def test_unhandled_exception_sanitized_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify unhandled 500 server error hides internal details when DEBUG=False."""
        custom_app = create_application()

        @custom_app.get("/test-unhandled-leak")
        async def unhandled_route():
            raise RuntimeError("Database password secret: pass123 at /var/run/secret.key")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-unhandled-leak", headers={"X-Request-ID": "req-leak-test"})

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "InternalServerError"
        # Zero leakage of internal secrets, paths, or exception text
        assert "pass123" not in data["error"]["message"]
        assert "/var/run/secret.key" not in data["error"]["message"]
        assert data["error"]["details"] is None
        assert response.headers.get("x-request-id") == "req-leak-test"
        assert data.get("request_id") == "req-leak-test"

    @pytest.mark.anyio
    async def test_unhandled_exception_exposes_details_when_debug_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify unhandled 500 exposes details when DEBUG=True."""
        from app.core import config
        settings = config.get_settings()
        monkeypatch.setattr(settings, "DEBUG", True)

        custom_app = create_application()

        @custom_app.get("/test-unhandled-debug")
        async def debug_route():
            raise ValueError("Debug variable inspectable.")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            response = await c.get("/test-unhandled-debug", headers={"X-Request-ID": "req-debug-test"})

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "InternalServerError"
        assert "Debug variable inspectable." in data["error"]["message"]
        assert data["error"]["details"] == {"exception_type": "ValueError"}
        assert response.headers.get("x-request-id") == "req-debug-test"


# ------------------------------------------------------------------------------
# 6. Prediction and Explain Endpoint Integration Verification
# ------------------------------------------------------------------------------

class TestEndpointDomainErrorPreservation:
    """Verify prediction and explain endpoints preserve domain error codes in real HTTP calls."""

    @pytest.mark.anyio
    async def test_prediction_endpoint_preserves_empty_image(self, client: httpx.AsyncClient) -> None:
        files = {"image": ("empty.png", b"", "image/png")}
        response = await client.post(
            "/api/v1/predictions", files=files, headers={"X-Request-ID": "pred-empty-id"}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["error_type"] == "EMPTY_IMAGE"
        assert response.headers.get("x-request-id") == "pred-empty-id"
        assert data["request_id"] == "pred-empty-id"

    @pytest.mark.anyio
    async def test_prediction_endpoint_preserves_unsupported_format(self, client: httpx.AsyncClient) -> None:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        files = {"image": ("bad.bmp", bmp_bytes, "image/bmp")}
        response = await client.post(
            "/api/v1/predictions", files=files, headers={"X-Request-ID": "pred-fmt-id"}
        )
        assert response.status_code == 415
        data = response.json()
        assert data["error"]["error_type"] == "UNSUPPORTED_FORMAT"
        assert response.headers.get("x-request-id") == "pred-fmt-id"

    @pytest.mark.anyio
    async def test_prediction_endpoint_preserves_image_too_large(self, client: httpx.AsyncClient) -> None:
        settings = get_settings()
        oversized = b"P" * (settings.MAX_IMAGE_UPLOAD_BYTES + 50)
        files = {"image": ("large.png", oversized, "image/png")}
        response = await client.post(
            "/api/v1/predictions", files=files, headers={"X-Request-ID": "pred-large-id"}
        )
        assert response.status_code == 413
        data = response.json()
        assert data["error"]["error_type"] == "IMAGE_TOO_LARGE"
        assert response.headers.get("x-request-id") == "pred-large-id"

    @pytest.mark.anyio
    async def test_explain_endpoint_preserves_invalid_target_class(self, client: httpx.AsyncClient) -> None:
        files = {"image": ("xray.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")}
        response = await client.post(
            "/api/v1/predictions/explain",
            files=files,
            data={"target_class": "INVALID_CLASS"},
            headers={"X-Request-ID": "exp-target-id"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["error_type"] == "INVALID_GRADCAM_INPUT"
        assert response.headers.get("x-request-id") == "exp-target-id"

    @pytest.mark.anyio
    async def test_explain_endpoint_preserves_invalid_alpha(self, client: httpx.AsyncClient) -> None:
        files = {"image": ("xray.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")}
        response = await client.post(
            "/api/v1/predictions/explain",
            files=files,
            data={"alpha": "2.5"},
            headers={"X-Request-ID": "exp-alpha-id"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["error_type"] == "INVALID_GRADCAM_INPUT"
        assert response.headers.get("x-request-id") == "exp-alpha-id"
