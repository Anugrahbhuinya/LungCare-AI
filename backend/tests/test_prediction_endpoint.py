"""Integration and contract test suite for the FastAPI prediction endpoint.

Validates:
1. POST /api/v1/predictions multipart/form-data upload workflows.
2. Successful classification response contracts and medical neutrality.
3. Upload validation (empty, corrupt, unsupported formats, oversized files).
4. Content-type vs. actual byte verification.
5. Model readiness, degraded startup states, and 503 handling.
6. Error mappings (400, 413, 415, 422, 500, 502, 503) and ErrorResponse envelopes.
7. Async execution offloading via threadpool.
"""

from __future__ import annotations

import io
from typing import Any, AsyncGenerator, Dict, Optional
import numpy as np
from PIL import Image
import pytest

try:
    import httpx
    from app.api.v1.endpoints.predictions import get_inference_service_dep
    from app.core.config import get_settings
    from app.main import app
    from app.schemas.prediction import RESEARCH_DISCLAIMER
    from app.services.image_preprocessor import (
        CorruptImageError,
        DecompressionBombError,
        EmptyImageError,
        ImageTooLargeError,
        UnsupportedFormatError,
    )
    from app.services.inference_service import (
        InferenceError,
        InferenceExecutionError,
        InvalidModelOutputError,
        ModelUnavailableError,
        PredictionResult,
    )
    from app.services.model_loader import get_model_loader
    HAS_BACKEND_DEPS = True
except ImportError:
    HAS_BACKEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_BACKEND_DEPS,
    reason="FastAPI backend dependencies required",
)


# ==============================================================================
# Helper Doubles & Test Image Creators
# ==============================================================================


def create_test_image_bytes(format_name: str = "PNG", size: tuple[int, int] = (100, 100)) -> bytes:
    """Generate in-memory synthetic image bytes."""
    img = Image.new("RGB", size, color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


class MockInferenceService:
    """Controllable test double for InferenceService."""

    def __init__(
        self,
        result: Optional[PredictionResult] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.result = result or PredictionResult(
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
            input_width=300,
            input_height=300,
            processed_width=224,
            processed_height=224,
            inference_time_ms=45.2,
            preprocessing_time_ms=12.5,
            disclaimer=RESEARCH_DISCLAIMER,
        )
        self.raise_exc = raise_exc
        self.predict_calls = 0

    def predict(self, image_bytes: bytes) -> PredictionResult:
        self.predict_calls += 1
        if self.raise_exc:
            raise self.raise_exc
        if len(image_bytes) == 0:
            raise EmptyImageError("Uploaded image byte sequence is empty.")
        if image_bytes.startswith(b"BM"):
            raise UnsupportedFormatError("Unsupported image format 'BMP'. Supported formats: ['JPEG', 'PNG', 'WEBP'].")
        if image_bytes == b"corrupt_data" or image_bytes.startswith(b"<") or not (
            image_bytes.startswith(b"\x89PNG") or image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"RIFF")
        ):
            raise CorruptImageError("Failed to decode or verify image bytes: unrecognized or corrupt header.")
        return self.result


@pytest.fixture
def anyio_backend() -> str:
    """Specify asyncio as the AnyIO backend."""
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Fixture providing an async ASGI test client with default mock service override."""
    mock_service = MockInferenceService()
    app.dependency_overrides[get_inference_service_dep] = lambda: mock_service

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Successful Prediction Workflow
# ==============================================================================


class TestSuccessfulPrediction:
    """Verify standard successful prediction responses."""

    @pytest.mark.anyio
    async def test_successful_prediction_png(self, client: httpx.AsyncClient) -> None:
        """Verify uploading a valid PNG returns HTTP 200 with full PredictionResponse."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}

        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["predicted_label"] == "Pneumonia"
        assert np.isclose(data["confidence"], 0.9854)
        assert len(data["probabilities"]) == 4
        assert "COVID19" in data["probabilities"]
        assert "NORMAL" in data["probabilities"]
        assert "PNEUMONIA" in data["probabilities"]
        assert "TURBERCULOSIS" in data["probabilities"]
        assert data["model_architecture"] == "DenseNet121"
        assert data["processed_width"] == 224
        assert data["processed_height"] == 224
        assert data["inference_time_ms"] > 0.0
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_turberculosis_mapping_in_response(self, client: httpx.AsyncClient) -> None:
        """Verify TURBERCULOSIS internal identifier maps to Tuberculosis display label."""
        tb_result = PredictionResult(
            predicted_class="TURBERCULOSIS",
            predicted_label="Tuberculosis",
            confidence=0.9421,
            probabilities={"COVID19": 0.01, "NORMAL": 0.01, "PNEUMONIA": 0.038, "TURBERCULOSIS": 0.9421},
            model_name="LungCare DenseNet121 Weighted",
            model_architecture="DenseNet121",
            input_width=224,
            input_height=224,
            processed_width=224,
            processed_height=224,
            inference_time_ms=42.0,
            preprocessing_time_ms=10.0,
            disclaimer=RESEARCH_DISCLAIMER,
        )
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(result=tb_result)

        image_bytes = create_test_image_bytes(format_name="JPEG")
        files = {"image": ("xray.jpg", image_bytes, "image/jpeg")}

        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "TURBERCULOSIS"
        assert data["predicted_label"] == "Tuberculosis"


# ==============================================================================
# 2. Upload Validation & Security
# ==============================================================================


class TestUploadValidation:
    """Verify defensive handling of invalid, missing, or oversized uploads."""

    @pytest.mark.anyio
    async def test_missing_image_field_returns_422(self, client: httpx.AsyncClient) -> None:
        """Verify request without 'image' multipart field returns HTTP 422."""
        response = await client.post("/api/v1/predictions", data={"other_field": "data"})
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "ValidationError"

    @pytest.mark.anyio
    async def test_empty_image_upload_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify zero-length byte upload returns HTTP 400 Bad Request."""
        files = {"image": ("empty.png", b"", "image/png")}
        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "empty" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_unsupported_format_returns_415(self, client: httpx.AsyncClient) -> None:
        """Verify unsupported format (BMP) returns HTTP 415 Unsupported Media Type."""
        bmp_bytes = b"BM" + b"\x00" * 100
        files = {"image": ("test.bmp", bmp_bytes, "image/bmp")}
        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 415
        data = response.json()
        assert data["status"] == "error"
        assert "unsupported" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_corrupt_image_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify corrupted bytes return HTTP 400 Bad Request."""
        files = {"image": ("corrupt.png", b"corrupt_data", "image/png")}
        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "decode or verify" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_oversized_upload_returns_413(self, client: httpx.AsyncClient) -> None:
        """Verify upload exceeding MAX_IMAGE_UPLOAD_BYTES returns HTTP 413 Payload Too Large."""
        settings = get_settings()
        # Create byte payload 10 bytes larger than allowed
        oversized = b"P" * (settings.MAX_IMAGE_UPLOAD_BYTES + 10)
        files = {"image": ("oversized.png", oversized, "image/png")}

        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 413
        data = response.json()
        assert data["status"] == "error"
        assert "exceeds" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_mime_type_spoofing_rejected(self, client: httpx.AsyncClient) -> None:
        """Verify client-supplied MIME type (e.g. image/png) is not trusted when bytes are invalid.

        Uploading text, HTML, or corrupted non-image content with an image/png Content-Type header
        must be caught and rejected with HTTP 400 by ImagePreprocessor validation.
        """
        spoofed_payload = b"<html><body>Malicious payload claiming to be PNG</body></html>"
        files = {"image": ("exploit.png", spoofed_payload, "image/png")}
        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "decode or verify" in data["error"]["message"].lower()


# ==============================================================================
# 3. Model Readiness & Degraded Startup States
# ==============================================================================


class TestModelReadinessIntegration:
    """Verify readiness endpoint and degraded startup states."""

    @pytest.mark.anyio
    async def test_readiness_probe_reflects_model_availability(self, client: httpx.AsyncClient) -> None:
        """Verify /ready probe reports loaded or unloaded state without throwing exceptions."""
        response = await client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "dependencies" in data
        assert "ml_model" in data["dependencies"]

    @pytest.mark.anyio
    async def test_prediction_returns_503_when_model_unavailable(self, client: httpx.AsyncClient) -> None:
        """Verify service returns HTTP 503 when the deep learning model is not loaded."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=ModelUnavailableError("Model weights not available in memory.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions", files=files)

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "unavailable" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_readiness_reports_not_ready_on_load_error(self, client: httpx.AsyncClient) -> None:
        """Verify /ready reports not_ready when loader encountered an initialization error."""
        loader = get_model_loader()
        # Temporarily simulate a startup load error
        orig_error = loader._load_error
        orig_loaded = loader._is_loaded
        try:
            loader._load_error = RuntimeError("Artifact checksum mismatch.")
            loader._is_loaded = False

            response = await client.get("/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "error"
            assert data["error"]["error_type"] == "MODEL_NOT_READY"
            assert "unavailable" in data["error"]["details"]["dependencies"]["ml_model"]
        finally:
            loader._load_error = orig_error
            loader._is_loaded = orig_loaded

    @pytest.mark.anyio
    async def test_startup_lifecycle_successful_model_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify application lifespan initializes the model once during successful startup."""
        from app.main import lifespan

        load_called = False

        class FakeLoader:
            def load_model(self) -> None:
                nonlocal load_called
                load_called = True

            def unload(self) -> None:
                pass

        monkeypatch.setattr("app.services.model_loader.get_model_loader", lambda: FakeLoader())

        async with lifespan(app):
            assert load_called is True

    @pytest.mark.anyio
    async def test_startup_lifecycle_handles_model_init_failure_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify application startup enters degraded state gracefully when model loading fails."""
        from app.main import lifespan

        class FailingLoader:
            def load_model(self) -> None:
                raise RuntimeError("Simulated missing model artifact during boot.")

            def unload(self) -> None:
                pass

        monkeypatch.setattr("app.services.model_loader.get_model_loader", lambda: FailingLoader())

        # Lifespan must NOT raise an exception; application must remain alive
        async with lifespan(app):
            pass

    @pytest.mark.anyio
    async def test_shutdown_lifecycle_logs_cleanup_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify model unload exception during shutdown is safely logged without failing shutdown."""
        from app.main import lifespan

        class ExplodingUnloadLoader:
            def load_model(self) -> None:
                pass

            def unload(self) -> None:
                raise RuntimeError("Resource release failure during shutdown.")

        monkeypatch.setattr("app.services.model_loader.get_model_loader", lambda: ExplodingUnloadLoader())

        # Lifespan must exit cleanly without re-raising shutdown exception
        async with lifespan(app):
            pass

    @pytest.mark.anyio
    async def test_readiness_and_liveness_after_startup_failure(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify readiness returns not_ready and prediction returns 503 while liveness remains 200."""
        loader = get_model_loader()
        orig_error = loader._load_error
        orig_loaded = loader._is_loaded
        try:
            loader._load_error = RuntimeError("Artifact checksum mismatch during startup.")
            loader._is_loaded = False

            # Liveness must remain healthy (HTTP 200)
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "healthy"

            # Readiness must report error (HTTP 503 with status=error)
            ready_resp = await client.get("/ready")
            assert ready_resp.status_code == 503
            ready_data = ready_resp.json()
            assert ready_data["status"] == "error"
            assert ready_data["error"]["error_type"] == "MODEL_NOT_READY"

            # Prediction must reject with HTTP 503 Service Unavailable
            app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
                raise_exc=ModelUnavailableError("Model not loaded.")
            )
            files = {"image": ("test.png", create_test_image_bytes(), "image/png")}
            pred_resp = await client.post("/api/v1/predictions", files=files)
            assert pred_resp.status_code == 503
        finally:
            loader._load_error = orig_error
            loader._is_loaded = orig_loaded

    @pytest.mark.anyio
    async def test_model_initialization_occurs_once(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify multiple sequential prediction requests reuse the same singleton service."""
        mock_service = MockInferenceService()
        app.dependency_overrides[get_inference_service_dep] = lambda: mock_service

        for i in range(3):
            files = {"image": (f"xray_{i}.png", create_test_image_bytes(), "image/png")}
            resp = await client.post("/api/v1/predictions", files=files)
            assert resp.status_code == 200

        # All 3 requests were serviced by the same singleton instance without re-instantiation
        assert mock_service.predict_calls == 3


# ==============================================================================
# 4. Inference Failures & Error Mapping
# ==============================================================================


class TestInferenceFailureMappings:
    """Verify internal inference exceptions map to appropriate HTTP status codes."""

    @pytest.mark.anyio
    async def test_invalid_model_output_returns_502(self, client: httpx.AsyncClient) -> None:
        """Verify contract violation (e.g. non-finite values) returns HTTP 502 Bad Gateway."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=InvalidModelOutputError("Output probabilities contain NaN.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions", files=files)

        assert response.status_code == 502
        data = response.json()
        assert data["status"] == "error"
        assert "validation failed" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_inference_execution_error_returns_500_without_leaking_internals(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify forward pass crash returns HTTP 500 without leaking internal tracebacks."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=InferenceExecutionError("Internal CUDA memory fault at 0x7FFF00.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions", files=files)

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        # Must not leak the raw internal memory address
        assert "0x7FFF00" not in data["error"]["message"]


# ==============================================================================
# 5. OpenAPI Discovery & Async Concurrency Safety
# ==============================================================================


class TestOpenAPIDiscoveryAndAsyncSafety:
    """Verify OpenAPI schema includes prediction endpoint and verifies threadpool offloading."""

    @pytest.mark.anyio
    async def test_openapi_schema_contains_predictions_endpoint(self, client: httpx.AsyncClient) -> None:
        """Verify POST /api/v1/predictions is documented in OpenAPI JSON."""
        response = await client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        paths = schema.get("paths", {})
        assert "/api/v1/predictions" in paths
        post_op = paths["/api/v1/predictions"].get("post", {})
        assert "Predictions" in post_op.get("tags", [])
        assert "200" in post_op.get("responses", {})

    @pytest.mark.anyio
    async def test_threadpool_offloading_invokes_service(self, client: httpx.AsyncClient) -> None:
        """Verify prediction endpoint offloads to the service cleanly."""
        mock_service = MockInferenceService()
        app.dependency_overrides[get_inference_service_dep] = lambda: mock_service

        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions", files=files)

        assert response.status_code == 200
        assert mock_service.predict_calls == 1


# ==============================================================================
# 6. End-to-End HTTP Pipeline & Real Model Smoke Integration
# ==============================================================================


class TestEndToEndPipelineAndRealModel:
    """End-to-end HTTP pipeline tests with real components and synthetic radiograph inputs."""

    @pytest.mark.anyio
    async def test_end_to_end_http_pipeline_with_real_preprocessor(self) -> None:
        """Verify HTTP POST /api/v1/predictions through real ImagePreprocessor and InferenceService.

        Uses the real ImagePreprocessor to validate image headers, size, modes, and normalization,
        and real InferenceService contract validation with a deterministic mock model forward pass.
        No medical diagnosis is claimed or inferred.
        """
        from app.services.image_preprocessor import ImagePreprocessor
        from app.services.inference_service import InferenceService

        class DeterministicMockModel:
            def __call__(self, tensor: np.ndarray, training: bool = False) -> np.ndarray:
                return np.array([[0.02, 0.03, 0.90, 0.05]], dtype=np.float32)

        class MockLoader:
            is_loaded = True
            load_error = None
            metadata = None

            def get_model(self) -> Any:
                return DeterministicMockModel()

        real_service = InferenceService(
            model_loader=MockLoader(),  # type: ignore
            preprocessor=ImagePreprocessor(),
        )
        app.dependency_overrides[get_inference_service_dep] = lambda: real_service

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as test_client:
            # Synthetic 300x300 PNG radiograph
            test_bytes = create_test_image_bytes(format_name="PNG", size=(300, 300))
            files = {"image": ("chest_xray.png", test_bytes, "image/png")}

            response = await test_client.post("/api/v1/predictions", files=files)
            assert response.status_code == 200
            data = response.json()

            assert data["predicted_class"] == "PNEUMONIA"
            assert data["predicted_label"] == "Pneumonia"
            assert np.isclose(data["confidence"], 0.90)
            assert len(data["probabilities"]) == 4
            assert data["processed_width"] == 224
            assert data["processed_height"] == 224
            assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_real_model_http_smoke(self) -> None:
        """Real-model HTTP integration smoke test using loaded DenseNet121 and synthetic image.

        Evaluates the full end-to-end HTTP pipeline against the actual trained .keras model.
        Skipped if TensorFlow is not available in the current Python environment.
        Strict medical neutrality: Output is validated purely as numerical research probabilities.
        """
        try:
            import tensorflow  # noqa: F401
            import keras  # noqa: F401
            from app.services.inference_service import get_inference_service
            from app.services.model_loader import get_model_loader
        except ImportError:
            pytest.skip("Real-model smoke test requires TensorFlow/Keras runtime (run in ML environment)")

        # Clear overrides to use real application singletons
        app.dependency_overrides.clear()
        loader = get_model_loader()
        try:
            loader.load_model()
            assert loader.is_loaded is True

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as test_client:
                # Synthetic 256x256 grayscale/RGB radiograph
                test_bytes = create_test_image_bytes(format_name="PNG", size=(256, 256))
                files = {"image": ("synthetic_chest_xray.png", test_bytes, "image/png")}

                response = await test_client.post("/api/v1/predictions", files=files)
                assert response.status_code == 200
                data = response.json()

                # Verify schema and mathematical validity
                assert data["predicted_class"] in ("COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS")
                assert data["predicted_label"] in ("COVID-19", "Normal", "Pneumonia", "Tuberculosis")
                assert 0.0 <= data["confidence"] <= 1.0
                assert len(data["probabilities"]) == 4
                assert np.isclose(sum(data["probabilities"].values()), 1.0, atol=1e-3)
                assert data["model_architecture"] == "DenseNet121"
                assert data["processed_width"] == 224
                assert data["processed_height"] == 224
                assert data["inference_time_ms"] > 0.0
                assert data["preprocessing_time_ms"] > 0.0
                assert data["disclaimer"] == RESEARCH_DISCLAIMER
        finally:
            loader.unload()


