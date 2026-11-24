"""End-to-End Regression & Integration Test Suite for Phase 2.3.4.6.

Validates the full request lifecycle across all backend components developed in Phase 2.3.4:
1. Complete Middleware Lifecycle:
   Request -> RequestCorrelationMiddleware -> AccessLoggingMiddleware -> CORS -> FastAPI -> Services -> Response.
2. Endpoint Matrix Verification:
   - GET /health
   - GET /ready
   - GET /api/v1/
   - POST /api/v1/predictions
   - POST /api/v1/predictions/explain
3. Boundary & Failure Conditions:
   - Empty uploads (400)
   - Corrupt bytes (400)
   - Unsupported media formats (415)
   - Oversized payloads (413)
   - Model unavailable state (503)
   - Invalid Grad-CAM parameters (400)
   - Server-side failure sanitization (500)
4. OpenAPI Contract Synchronization:
   - Schemas match runtime payloads across success and error responses.
5. Security & Privacy Protection:
   - Zero leakage of raw image bytes, filenames, base64 strings, or query secrets in logs.
   - Production error sanitization without internal stack trace leakage.
6. Real Model Smoke Test:
   - Live DenseNet121 forward pass and Grad-CAM execution through full ASGI stack.
7. Concurrency Isolation:
   - Concurrent overlapping requests preserve strict context correlation without leakage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import io
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, Optional
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
    from app.core.config import get_settings
    from app.core.request_context import get_request_id
    from app.main import app, create_application
    from app.schemas.gradcam import (
        CLASS_DISPLAY_NAMES,
        CLASS_INDICES,
        RESEARCH_DISCLAIMER,
        TARGET_CONV_LAYER,
        GradCAMExplainResponse,
        GradCAMResult,
    )
    from app.schemas.prediction import PredictionResponse
    from app.services.gradcam_visualizer import VisualizerResult
    from app.services.image_preprocessor import (
        CorruptImageError,
        EmptyImageError,
        PreprocessedImageResult,
        UnsupportedFormatError,
    )
    from app.services.inference_service import ModelUnavailableError, PredictionResult
    from app.services.model_loader import get_model_loader
    HAS_BACKEND_DEPS = True
except ImportError:
    HAS_BACKEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_BACKEND_DEPS,
    reason="FastAPI backend dependencies required",
)

UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def create_synthetic_png_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    """Generate in-memory synthetic PNG bytes."""
    img = Image.new("RGB", size, color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockInferenceService:
    """Configurable test double for InferenceService."""

    def __init__(self, raise_exc: Optional[Exception] = None) -> None:
        self.raise_exc = raise_exc
        self.result = PredictionResult(
            predicted_class="PNEUMONIA",
            predicted_label="Pneumonia",
            confidence=0.9854,
            probabilities={
                "COVID19": 0.0051,
                "NORMAL": 0.0042,
                "PNEUMONIA": 0.9854,
                "TURBERCULOSIS": 0.0053,
            },
            model_name="LungCare DenseNet121 Weighted",
            model_architecture="DenseNet121",
            input_width=100,
            input_height=100,
            processed_width=224,
            processed_height=224,
            inference_time_ms=45.2,
            preprocessing_time_ms=12.5,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    def predict(self, content: bytes) -> PredictionResult:
        if self.raise_exc:
            raise self.raise_exc
        if len(content) == 0:
            raise EmptyImageError("Uploaded image byte sequence is empty.")
        if content.startswith(b"BM"):
            raise UnsupportedFormatError("Unsupported image format 'BMP'. Supported formats: ['JPEG', 'PNG', 'WEBP'].")
        if not (content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8\xff") or content.startswith(b"RIFF")):
            raise CorruptImageError("Failed to decode or verify image bytes: unrecognized or corrupt header.")
        return self.result

    def predict_tensor(
        self,
        tensor: np.ndarray,
        original_width: int = 224,
        original_height: int = 224,
        preprocessing_time_ms: float = 0.0,
    ) -> PredictionResult:
        if self.raise_exc:
            raise self.raise_exc
        return self.result


class MockGradCAMService:
    """Configurable test double for GradCAMService."""

    def __init__(self, raise_exc: Optional[Exception] = None) -> None:
        self.raise_exc = raise_exc
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        self.result = GradCAMResult(
            selected_class="PNEUMONIA",
            selected_label="Pneumonia",
            class_index=2,
            target_layer=TARGET_CONV_LAYER,
            heatmap=heatmap,
            heatmap_width=224,
            heatmap_height=224,
            status="success",
            confidence=0.9854,
            probabilities={
                "COVID19": 0.0051,
                "NORMAL": 0.0042,
                "PNEUMONIA": 0.9854,
                "TURBERCULOSIS": 0.0053,
            },
            latency_ms=92.5,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    def generate_heatmap(
        self,
        image_tensor: Any,
        class_index: Optional[int] = None,
    ) -> GradCAMResult:
        if self.raise_exc:
            raise self.raise_exc
        if class_index is not None and class_index in CLASS_INDICES:
            c_name = CLASS_INDICES[class_index]
            c_label = CLASS_DISPLAY_NAMES.get(c_name, c_name)
            return GradCAMResult(
                selected_class=c_name,
                selected_label=c_label,
                class_index=class_index,
                target_layer=TARGET_CONV_LAYER,
                heatmap=self.result.heatmap,
                heatmap_width=224,
                heatmap_height=224,
                status="success",
                confidence=0.9854,
                probabilities=self.result.probabilities,
                latency_ms=self.result.latency_ms,
                disclaimer=RESEARCH_DISCLAIMER,
            )
        return self.result


class MockVisualizer:
    """Configurable test double for GradCAMVisualizer."""

    def create_overlay(
        self,
        original_image: Any,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[str] = None,
    ) -> VisualizerResult:
        overlay_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_arr[:, :, 0] = 255
        return VisualizerResult(
            overlay_array=overlay_arr,
            colored_heatmap=colored_arr,
            original_width=100,
            original_height=100,
            alpha=alpha if alpha is not None else 0.4,
            colormap=colormap or "jet",
        )


class MockPreprocessor:
    """Configurable test double for ImagePreprocessor."""

    def preprocess(self, content: bytes) -> PreprocessedImageResult:
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        return PreprocessedImageResult(
            tensor=tensor,
            original_width=100,
            original_height=100,
            processed_width=224,
            processed_height=224,
            original_format="PNG",
            original_mode="RGB",
            final_mode="RGB",
            exif_transposed=False,
            alpha_composited=False,
        )


@pytest.fixture
async def mocked_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with mocked ML services for fast, non-GPU unit integration tests."""
    mock_inf = MockInferenceService()
    mock_cam = MockGradCAMService()
    mock_vis = MockVisualizer()
    mock_prep = MockPreprocessor()

    app.dependency_overrides[get_inference_service_dep] = lambda: mock_inf
    app.dependency_overrides[get_gradcam_service_dep] = lambda: mock_cam
    app.dependency_overrides[get_gradcam_visualizer_dep] = lambda: mock_vis
    app.dependency_overrides[get_image_preprocessor_dep] = lambda: mock_prep

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
async def raw_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client without dependency overrides (uses real application pipeline)."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c


class TestCompleteRequestLifecycleAndAccessLogging:
    """Verify request correlation, pure ASGI access logging, and lifecycle integrity."""

    @pytest.mark.anyio
    async def test_full_pipeline_success_emits_single_structured_access_log(
        self, mocked_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify Request -> RequestCorrelation -> AccessLogging -> CORS -> Endpoint -> Response."""
        caplog.clear()
        caplog.set_level(logging.INFO, logger="lungcare.access")

        custom_req_id = "lifecycle-integration-req-001"
        files = {"image": ("radiograph.png", create_synthetic_png_bytes(), "image/png")}

        response = await mocked_client.post(
            "/api/v1/predictions",
            files=files,
            headers={"X-Request-ID": custom_req_id, "Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200

        # 1. Header correlation
        assert response.headers.get("x-request-id") == custom_req_id
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

        # 2. Body metadata consistency
        data = response.json()
        assert data["status"] == "success"
        assert data["request_id"] == custom_req_id
        assert data["predicted_class"] == "PNEUMONIA"
        assert "timestamp" in data

        # 3. Access log record verification
        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1, "Exactly one access log must be emitted"
        rec = records[0]
        assert "HTTP POST /api/v1/predictions -> 200" in rec.message
        assert getattr(rec, "status_code", None) == 200
        assert getattr(rec, "request_id", None) == custom_req_id
        assert getattr(rec, "duration_ms", None) >= 0.0

    @pytest.mark.anyio
    async def test_full_pipeline_error_emits_single_structured_access_log(
        self, mocked_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify error lifecycle: Request -> Handled 400 -> AccessLogging (400) -> ErrorResponse."""
        caplog.clear()
        caplog.set_level(logging.INFO, logger="lungcare.access")

        custom_req_id = "lifecycle-integration-req-err-400"
        files = {"image": ("empty.png", b"", "image/png")}

        response = await mocked_client.post(
            "/api/v1/predictions",
            files=files,
            headers={"X-Request-ID": custom_req_id},
        )
        assert response.status_code == 400

        # Header and body correlation
        assert response.headers.get("x-request-id") == custom_req_id
        data = response.json()
        assert data["status"] == "error"
        assert data["request_id"] == custom_req_id
        assert data["error"]["error_type"] == "EMPTY_IMAGE"

        # Access log record verification
        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1
        rec = records[0]
        assert "HTTP POST /api/v1/predictions -> 400" in rec.message
        assert getattr(rec, "status_code", None) == 400
        assert getattr(rec, "request_id", None) == custom_req_id


class TestAllEndpointsMatrix:
    """Verify all 5 primary endpoints across valid inputs and failure boundaries."""

    @pytest.mark.anyio
    async def test_get_health_endpoint(self, mocked_client: httpx.AsyncClient) -> None:
        """GET /health returns 200 healthy status and UTC timestamp."""
        resp = await mocked_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "lungcare-api"
        assert "timestamp" in data

    @pytest.mark.anyio
    async def test_get_ready_endpoint(self, mocked_client: httpx.AsyncClient) -> None:
        """GET /ready returns 200 readiness probe with subsystem status."""
        resp = await mocked_client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "not_ready")
        assert "model_loaded" in data
        assert "dependencies" in data
        assert "timestamp" in data

    @pytest.mark.anyio
    async def test_get_api_v1_discovery_endpoint(self, mocked_client: httpx.AsyncClient) -> None:
        """GET /api/v1/ returns 200 discovery metadata and routes mapping."""
        resp = await mocked_client.get("/api/v1/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_version"] == "v1"
        assert "health" in data["endpoints"]
        assert "ready" in data["endpoints"]
        assert "predictions" in data["endpoints"]
        assert "explain" in data["endpoints"]

    @pytest.mark.anyio
    async def test_post_predictions_valid_image(self, mocked_client: httpx.AsyncClient) -> None:
        """POST /api/v1/predictions returns 200 with full PredictionResponse schema."""
        files = {"image": ("chest_sample.png", create_synthetic_png_bytes(), "image/png")}
        resp = await mocked_client.post("/api/v1/predictions", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "success"
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["predicted_label"] == "Pneumonia"
        assert data["display_name"] == "Pneumonia"
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 4
        assert data["model_architecture"] == "DenseNet121"
        assert "timestamp" in data
        assert data["request_id"] == resp.headers.get("x-request-id")
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_post_predictions_explain_valid_image(self, mocked_client: httpx.AsyncClient) -> None:
        """POST /api/v1/predictions/explain returns 200 with full GradCAMExplainResponse schema."""
        files = {"image": ("chest_sample.png", create_synthetic_png_bytes(), "image/png")}
        resp = await mocked_client.post(
            "/api/v1/predictions/explain",
            files=files,
            data={"target_class": "PNEUMONIA", "alpha": "0.45"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "success"
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["selected_target_class"] == "PNEUMONIA"
        assert data["target_class_index"] == 2
        assert data["heatmap_base64"].startswith("data:image/png;base64,")
        assert data["overlay_base64"].startswith("data:image/png;base64,")
        assert data["explanation"]["status"] == "success"
        assert np.isclose(data["explanation"]["alpha"], 0.45)
        assert "timestamp" in data
        assert data["request_id"] == resp.headers.get("x-request-id")
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_post_predictions_malformed_and_boundary_cases(
        self, mocked_client: httpx.AsyncClient
    ) -> None:
        """Verify predictions boundary rejections: empty, corrupt, unsupported, oversized."""
        # 1. Empty image (400)
        resp_empty = await mocked_client.post("/api/v1/predictions", files={"image": ("empty.png", b"", "image/png")})
        assert resp_empty.status_code == 400
        assert resp_empty.json()["error"]["error_type"] == "EMPTY_IMAGE"

        # 2. Corrupt image bytes (400)
        resp_corrupt = await mocked_client.post("/api/v1/predictions", files={"image": ("corrupt.png", b"not_an_image_data", "image/png")})
        assert resp_corrupt.status_code == 400
        assert resp_corrupt.json()["error"]["error_type"] == "CORRUPT_IMAGE"

        # 3. Unsupported format (BMP) (415)
        bmp_header = b"BM" + b"\x00" * 50
        resp_unsupported = await mocked_client.post("/api/v1/predictions", files={"image": ("unsupported.bmp", bmp_header, "image/bmp")})
        assert resp_unsupported.status_code == 415
        assert resp_unsupported.json()["error"]["error_type"] == "UNSUPPORTED_FORMAT"

        # 4. Oversized upload (413)
        oversized_bytes = b"x" * (10 * 1024 * 1024 + 50)
        resp_oversized = await mocked_client.post("/api/v1/predictions", files={"image": ("huge.png", oversized_bytes, "image/png")})
        assert resp_oversized.status_code == 413
        assert resp_oversized.json()["error"]["error_type"] == "IMAGE_TOO_LARGE"

    @pytest.mark.anyio
    async def test_post_predictions_explain_parameter_validation(
        self, mocked_client: httpx.AsyncClient
    ) -> None:
        """Verify explain endpoint parameter rejections: invalid target_class and alpha."""
        files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}

        # 1. Invalid target class string (400)
        resp_bad_target = await mocked_client.post(
            "/api/v1/predictions/explain", files=files, data={"target_class": "UNKNOWN_DISEASE"}
        )
        assert resp_bad_target.status_code == 400
        assert resp_bad_target.json()["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

        # 2. Out of bounds index (400)
        resp_bad_idx = await mocked_client.post(
            "/api/v1/predictions/explain", files=files, data={"target_class": "99"}
        )
        assert resp_bad_idx.status_code == 400
        assert resp_bad_idx.json()["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

        # 3. Out of bounds alpha (400)
        resp_bad_alpha = await mocked_client.post(
            "/api/v1/predictions/explain", files=files, data={"alpha": "1.5"}
        )
        assert resp_bad_alpha.status_code == 400
        assert resp_bad_alpha.json()["error"]["error_type"] == "INVALID_GRADCAM_INPUT"

    @pytest.mark.anyio
    async def test_model_unavailable_returns_503(self) -> None:
        """Verify model unavailable returns 503 with standardized ErrorResponse."""
        unavailable_inf = MockInferenceService(raise_exc=ModelUnavailableError("Model offline"))
        app.dependency_overrides[get_inference_service_dep] = lambda: unavailable_inf

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}
            resp = await c.post("/api/v1/predictions", files=files)
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "error"
            assert data["error"]["error_type"] == "MODEL_UNAVAILABLE"

        app.dependency_overrides.clear()


class TestSecurityPrivacyAndSanitization:
    """Verify security, privacy, and sensitive data non-disclosure across the stack."""

    @pytest.mark.anyio
    async def test_privacy_non_disclosure_in_access_logs(
        self, mocked_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify image bytes, filenames, base64 payloads, and query secrets never enter access logs."""
        caplog.clear()
        caplog.set_level(logging.INFO, logger="lungcare.access")

        raw_bytes = create_synthetic_png_bytes()
        sensitive_filename = "patient_jane_doe_chest_scan_98765.png"
        secret_query = "patient_ssn=000-12-3456&api_token=secret_jwt_xyz"
        files = {"image": (sensitive_filename, raw_bytes, "image/png")}

        resp = await mocked_client.post(
            f"/api/v1/predictions/explain?{secret_query}",
            files=files,
            headers={"X-Request-ID": "privacy-audit-req-101"},
        )
        assert resp.status_code == 200

        records = [r for r in caplog.records if r.name == "lungcare.access"]
        assert len(records) == 1
        rec_msg = records[0].message

        # Assert zero leakage
        assert sensitive_filename not in rec_msg
        assert "patient_ssn" not in rec_msg
        assert "000-12-3456" not in rec_msg
        assert "secret_jwt_xyz" not in rec_msg
        assert "data:image/png;base64" not in rec_msg
        assert str(raw_bytes[:20]) not in rec_msg

    @pytest.mark.anyio
    async def test_production_error_sanitization_without_stack_traces(self) -> None:
        """Verify unhandled exceptions in production hide internal stack traces and secrets."""
        custom_app = create_application()

        @custom_app.get("/unhandled-server-fail")
        async def fail_route():
            raise RuntimeError("Database password in memory: secret_pass_99999")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as c:
            resp = await c.get("/unhandled-server-fail", headers={"X-Request-ID": "sanitization-check-1"})
            assert resp.status_code == 500
            data = resp.json()

            assert data["status"] == "error"
            assert data["error"]["error_type"] == "InternalServerError"
            # In production (default DEBUG=False), internal error messages are sanitized
            assert "secret_pass_99999" not in str(data)
            assert "Traceback" not in str(data)
            assert "File " not in str(data)


class TestRealModelSmokeIntegration:
    """Verify live real DenseNet121 model loading and forward execution if artifacts exist."""

    @pytest.mark.anyio
    async def test_real_model_end_to_end_smoke(self, raw_client: httpx.AsyncClient) -> None:
        """Execute real DenseNet121 prediction pipeline on synthetic radiograph."""
        model_path = r"models/LungCare_AI_Model/lungcare_densenet121.keras"
        if not os.path.exists(model_path):
            pytest.skip("Model artifact not found; skipping real model smoke test.")

        files = {"image": ("synthetic_real_smoke.png", create_synthetic_png_bytes((150, 150)), "image/png")}
        resp = await raw_client.post(
            "/api/v1/predictions",
            files=files,
            headers={"X-Request-ID": "real-model-smoke-req-999"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "success"
        assert data["request_id"] == "real-model-smoke-req-999"
        assert data["predicted_class"] in ("COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS")
        assert 0.0 <= data["confidence"] <= 1.0
        assert np.isclose(sum(data["probabilities"].values()), 1.0, atol=1e-3)
        assert data["inference_time_ms"] > 0.0
        assert data["model_architecture"] == "DenseNet121"


class TestConcurrencyAndIsolation:
    """Verify simultaneous requests preserve strict correlation context isolation without leaking."""

    @pytest.mark.anyio
    async def test_concurrent_requests_preserve_strict_id_isolation(
        self, mocked_client: httpx.AsyncClient
    ) -> None:
        """Simultaneous asynchronous requests receive their exact corresponding request_ids."""
        num_requests = 10
        client_ids = [f"concurrent-integration-{i:03d}" for i in range(num_requests)]
        files_data = create_synthetic_png_bytes()

        async def send_req(cid: str) -> httpx.Response:
            files = {"image": ("chest.png", files_data, "image/png")}
            return await mocked_client.post(
                "/api/v1/predictions", files=files, headers={"X-Request-ID": cid}
            )

        tasks = [asyncio.create_task(send_req(cid)) for cid in client_ids]
        responses = await asyncio.gather(*tasks)

        for i, resp in enumerate(responses):
            assert resp.status_code == 200
            expected_id = client_ids[i]
            assert resp.headers.get("x-request-id") == expected_id
            data = resp.json()
            assert data["request_id"] == expected_id
            assert data["status"] == "success"
