"""Targeted test suite for Phase 2.3.4.5: Response Metadata Consistency.

Verifies:
1. Successful prediction responses expose status="success", request_id, timestamp, and display_name.
2. Successful explanation responses expose status="success", request_id, timestamp, and display_name.
3. Response body request_id matches X-Request-ID response header across custom and generated IDs.
4. Health and readiness responses have appropriate metadata and UTC timestamps without exposing internal details.
5. Error responses continue adhering to the centralized ErrorResponse contract.
6. Generated OpenAPI specification accurately documents all new and existing metadata fields.
7. Existing API contracts and frontend expectations remain unbroken.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
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
    from app.main import app
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
    from app.services.image_preprocessor import PreprocessedImageResult
    from app.services.inference_service import PredictionResult
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
    """Generate valid in-memory PNG bytes."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockInferenceService:
    """Mock for InferenceService."""

    def __init__(self, result: Optional[PredictionResult] = None) -> None:
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
            input_width=100,
            input_height=100,
            processed_width=224,
            processed_height=224,
            inference_time_ms=45.2,
            preprocessing_time_ms=12.5,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    def predict(self, content: bytes) -> PredictionResult:
        return self.result

    def predict_tensor(
        self,
        tensor: np.ndarray,
        original_width: int = 224,
        original_height: int = 224,
        preprocessing_time_ms: float = 0.0,
    ) -> PredictionResult:
        return self.result


class MockGradCAMService:
    """Mock for GradCAMService."""

    def __init__(self, result: Optional[GradCAMResult] = None) -> None:
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        self.result = result or GradCAMResult(
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
        return self.result


class MockVisualizer:
    """Mock for GradCAMVisualizer."""

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
    """Mock for ImagePreprocessor."""

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
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with mocked ML services for deterministic execution."""
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


class TestPredictionResponseMetadataConsistency:
    """Verify metadata consistency for POST /api/v1/predictions."""

    @pytest.mark.anyio
    async def test_prediction_metadata_with_explicit_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify status='success', matching request_id, display_name, and valid ISO timestamp."""
        custom_id = "req-pred-meta-test-001"
        files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}
        resp = await client.post(
            "/api/v1/predictions", files=files, headers={"X-Request-ID": custom_id}
        )
        assert resp.status_code == 200
        header_id = resp.headers.get("x-request-id")
        assert header_id == custom_id

        data = resp.json()

        # 1. Status consistency
        assert data["status"] == "success"

        # 2. Correlation request_id consistency
        assert data["request_id"] == custom_id
        assert data["request_id"] == header_id

        # 3. Prediction results consistency
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["predicted_label"] == "Pneumonia"
        assert data["display_name"] == "Pneumonia"
        assert np.isclose(data["confidence"], 0.9854)

        # 4. Probabilities distribution
        assert isinstance(data["probabilities"], dict)
        assert set(data["probabilities"].keys()) == {"COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"}
        assert np.isclose(sum(data["probabilities"].values()), 1.0, atol=1e-3)

        # 5. Timestamp consistency (valid ISO 8601 UTC)
        assert "timestamp" in data
        parsed_dt = datetime.fromisoformat(data["timestamp"])
        assert parsed_dt.tzinfo is not None

        # 6. Model provenance and disclaimer
        assert data["model_architecture"] == "DenseNet121"
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_prediction_metadata_with_generated_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify automatic request_id generation is reflected in both header and response body."""
        files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}
        resp = await client.post("/api/v1/predictions", files=files)
        assert resp.status_code == 200
        header_id = resp.headers.get("x-request-id")
        assert header_id is not None
        assert UUID4_PATTERN.match(header_id) is not None

        data = resp.json()
        assert data["status"] == "success"
        assert data["request_id"] == header_id


class TestGradCAMExplainResponseMetadataConsistency:
    """Verify metadata consistency for POST /api/v1/predictions/explain."""

    @pytest.mark.anyio
    async def test_explain_metadata_with_explicit_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify status='success', matching request_id, display_name, and valid ISO timestamp."""
        custom_id = "req-explain-meta-test-002"
        files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}
        resp = await client.post(
            "/api/v1/predictions/explain", files=files, headers={"X-Request-ID": custom_id}
        )
        assert resp.status_code == 200
        header_id = resp.headers.get("x-request-id")
        assert header_id == custom_id

        data = resp.json()

        # 1. Status consistency
        assert data["status"] == "success"

        # 2. Correlation request_id consistency
        assert data["request_id"] == custom_id
        assert data["request_id"] == header_id

        # 3. Prediction results consistency
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["predicted_label"] == "Pneumonia"
        assert data["display_name"] == "Pneumonia"
        assert np.isclose(data["confidence"], 0.9854)

        # 4. Probabilities distribution
        assert isinstance(data["probabilities"], dict)
        assert set(data["probabilities"].keys()) == {"COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"}

        # 5. Grad-CAM targeting and explanation metadata
        assert data["selected_target_class"] == "PNEUMONIA"
        assert data["selected_target_label"] == "Pneumonia"
        assert data["target_class_index"] == 2
        assert data["heatmap_base64"].startswith("data:image/png;base64,")
        assert data["overlay_base64"].startswith("data:image/png;base64,")
        assert data["explanation"]["status"] == "success"
        assert data["explanation"]["target_layer"] == TARGET_CONV_LAYER

        # 6. Timestamp consistency (valid ISO 8601 UTC)
        assert "timestamp" in data
        parsed_dt = datetime.fromisoformat(data["timestamp"])
        assert parsed_dt.tzinfo is not None

        # 7. Disclaimer
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_explain_metadata_with_generated_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify automatic request_id generation is reflected in both header and response body."""
        files = {"image": ("chest.png", create_synthetic_png_bytes(), "image/png")}
        resp = await client.post("/api/v1/predictions/explain", files=files)
        assert resp.status_code == 200
        header_id = resp.headers.get("x-request-id")
        assert header_id is not None
        assert UUID4_PATTERN.match(header_id) is not None

        data = resp.json()
        assert data["status"] == "success"
        assert data["request_id"] == header_id


class TestHealthAndReadinessMetadataConsistency:
    """Verify metadata consistency for /health and /ready probes."""

    @pytest.mark.anyio
    async def test_health_probe_metadata_and_timestamp(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify health check exposes status='healthy', service, version, and ISO timestamp."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "healthy"
        assert data["service"] == "lungcare-api"
        assert data["version"] == "1.0.0"

        # Timestamp is present and valid
        assert "timestamp" in data
        parsed_dt = datetime.fromisoformat(data["timestamp"])
        assert parsed_dt.tzinfo is not None

        # No sensitive internal details or file paths
        assert "/var" not in str(data)
        assert "C:\\" not in str(data)

    @pytest.mark.anyio
    async def test_readiness_probe_metadata_and_timestamp(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify readiness check exposes subsystem status, environment, and ISO timestamp."""
        resp = await client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] in ("ready", "not_ready")
        assert data["service"] == "lungcare-api"
        assert data["version"] == "1.0.0"
        assert isinstance(data["model_loaded"], bool)
        assert "environment" in data
        assert "dependencies" in data
        assert "ml_model" in data["dependencies"]

        # Timestamp is present and valid
        assert "timestamp" in data
        parsed_dt = datetime.fromisoformat(data["timestamp"])
        assert parsed_dt.tzinfo is not None

        # No sensitive internal details or file paths
        assert "/var" not in str(data)
        assert "C:\\" not in str(data)


class TestErrorResponseContractPreservation:
    """Verify error responses consistently expose status='error', request_id, and timestamp."""

    @pytest.mark.anyio
    async def test_400_error_response_metadata(self, client: httpx.AsyncClient) -> None:
        """Verify 400 error returns standard ErrorResponse envelope with matching request_id."""
        custom_id = "req-err-400-check"
        # Empty payload triggers EmptyImageError (400)
        files = {"image": ("empty.png", b"", "image/png")}
        resp = await client.post(
            "/api/v1/predictions", files=files, headers={"X-Request-ID": custom_id}
        )
        assert resp.status_code == 400
        assert resp.headers.get("x-request-id") == custom_id

        data = resp.json()
        assert data["status"] == "error"
        assert data["request_id"] == custom_id
        assert data["error"]["error_type"] == "EMPTY_IMAGE"
        assert "timestamp" in data
        parsed_dt = datetime.fromisoformat(data["timestamp"])
        assert parsed_dt.tzinfo is not None

    @pytest.mark.anyio
    async def test_422_validation_error_metadata(self, client: httpx.AsyncClient) -> None:
        """Verify 422 error returns standard ErrorResponse envelope with matching request_id."""
        custom_id = "req-err-422-check"
        # Missing file field triggers 422
        resp = await client.post(
            "/api/v1/predictions", headers={"X-Request-ID": custom_id}
        )
        assert resp.status_code == 422
        assert resp.headers.get("x-request-id") == custom_id

        data = resp.json()
        assert data["status"] == "error"
        assert data["request_id"] == custom_id
        assert data["error"]["error_type"] == "ValidationError"
        assert "timestamp" in data


class TestOpenAPISchemaMetadataHarmonization:
    """Verify OpenAPI component schemas accurately document harmonized metadata."""

    def test_prediction_schema_documents_metadata_fields(self) -> None:
        """Verify PredictionResponse schema in OpenAPI contains status, display_name, timestamp, request_id."""
        schema = app.openapi()
        pred_props = schema["components"]["schemas"]["PredictionResponse"]["properties"]

        assert "status" in pred_props
        assert "predicted_class" in pred_props
        assert "predicted_label" in pred_props
        assert "display_name" in pred_props
        assert "confidence" in pred_props
        assert "probabilities" in pred_props
        assert "timestamp" in pred_props
        assert "request_id" in pred_props
        assert "disclaimer" in pred_props

    def test_explain_schema_documents_metadata_fields(self) -> None:
        """Verify GradCAMExplainResponse schema in OpenAPI contains status, display_name, timestamp, request_id."""
        schema = app.openapi()
        exp_props = schema["components"]["schemas"]["GradCAMExplainResponse"]["properties"]

        assert "status" in exp_props
        assert "predicted_class" in exp_props
        assert "predicted_label" in exp_props
        assert "display_name" in exp_props
        assert "confidence" in exp_props
        assert "probabilities" in exp_props
        assert "selected_target_class" in exp_props
        assert "selected_target_label" in exp_props
        assert "target_class_index" in exp_props
        assert "explanation" in exp_props
        assert "timestamp" in exp_props
        assert "request_id" in exp_props
        assert "disclaimer" in exp_props

    def test_probe_schemas_document_timestamp(self) -> None:
        """Verify HealthResponse and ReadinessResponse schemas in OpenAPI contain timestamp."""
        schema = app.openapi()
        health_props = schema["components"]["schemas"]["HealthResponse"]["properties"]
        ready_props = schema["components"]["schemas"]["ReadinessResponse"]["properties"]

        assert "status" in health_props
        assert "service" in health_props
        assert "version" in health_props
        assert "timestamp" in health_props

        assert "status" in ready_props
        assert "service" in ready_props
        assert "version" in ready_props
        assert "model_loaded" in ready_props
        assert "dependencies" in ready_props
        assert "timestamp" in ready_props
