"""Integration and contract test suite for the FastAPI Grad-CAM explainability endpoint.

Validates:
1. POST /api/v1/predictions/explain multipart/form-data upload workflows.
2. Successful classification + visual explanation response contracts.
3. Default predicted-class targeting vs. explicit target class selection (by name and index).
4. Custom alpha blending factors.
5. Base64 data URI decoding into valid PNG images.
6. Upload validation (missing, empty, corrupt, unsupported formats, oversized files).
7. Error mappings (400, 413, 415, 422, 500, 502, 503) and ErrorResponse envelopes.
8. Zero filesystem writes during processing.
9. Real model integration smoke test using the loaded DenseNet121 model artifact.
"""

from __future__ import annotations

import base64
import io
import os
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
    from app.main import app
    from app.schemas.gradcam import (
        CLASS_DISPLAY_NAMES,
        CLASS_INDICES,
        RESEARCH_DISCLAIMER,
        TARGET_CONV_LAYER,
        GradCAMExplainResponse,
    )
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
    from app.services.gradcam_service import (
        GradCAMError,
        GradCAMGenerationError,
        GradCAMGradientError,
        GradCAMLayerNotFoundError,
        GradCAMResult,
        InvalidGradCAMInputError,
    )
    from app.services.gradcam_visualizer import (
        InvalidVisualizerInputError,
        VisualizerError,
        VisualizerResult,
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
        self.predict_tensor_calls = 0

    def predict_tensor(
        self,
        tensor: np.ndarray,
        original_width: int = 224,
        original_height: int = 224,
        preprocessing_time_ms: float = 0.0,
    ) -> PredictionResult:
        self.predict_tensor_calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.result


class MockGradCAMService:
    """Controllable test double for GradCAMService."""

    def __init__(
        self,
        result: Optional[GradCAMResult] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
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
        self.raise_exc = raise_exc
        self.generate_heatmap_calls = 0
        self.last_class_index: Optional[int] = None

    def generate_heatmap(
        self,
        image_tensor: Any,
        class_index: Optional[int] = None,
    ) -> GradCAMResult:
        self.generate_heatmap_calls += 1
        self.last_class_index = class_index
        if self.raise_exc:
            raise self.raise_exc
        if class_index is not None and class_index in CLASS_INDICES:
            c_name = CLASS_INDICES[class_index]
            c_label = CLASS_DISPLAY_NAMES[c_name]
            return GradCAMResult(
                selected_class=c_name,
                selected_label=c_label,
                class_index=class_index,
                target_layer=TARGET_CONV_LAYER,
                heatmap=self.result.heatmap,
                heatmap_width=224,
                heatmap_height=224,
                status=self.result.status,
                confidence=self.result.probabilities.get(c_name, 0.5),
                probabilities=self.result.probabilities,
                latency_ms=self.result.latency_ms,
                disclaimer=RESEARCH_DISCLAIMER,
            )
        return self.result


class MockGradCAMVisualizer:
    """Controllable test double for GradCAMVisualizer."""

    def __init__(self, raise_exc: Optional[Exception] = None) -> None:
        self.raise_exc = raise_exc
        self.create_overlay_calls = 0
        self.last_alpha: Optional[float] = None

    def create_overlay(
        self,
        original_image: Any,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[str] = None,
    ) -> VisualizerResult:
        self.create_overlay_calls += 1
        self.last_alpha = alpha
        if self.raise_exc:
            raise self.raise_exc
        # Return synthetic 100x100 overlay
        overlay_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_arr[:, :, 0] = 255  # Red channel
        return VisualizerResult(
            overlay_array=overlay_arr,
            colored_heatmap=colored_arr,
            original_width=100,
            original_height=100,
            alpha=alpha if alpha is not None else 0.4,
            colormap=colormap or "jet",
        )


@pytest.fixture
def anyio_backend() -> str:
    """Specify asyncio as the AnyIO backend."""
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Fixture providing an async ASGI test client with default mock overrides."""
    mock_inf = MockInferenceService()
    mock_cam = MockGradCAMService()
    mock_vis = MockGradCAMVisualizer()

    app.dependency_overrides[get_inference_service_dep] = lambda: mock_inf
    app.dependency_overrides[get_gradcam_service_dep] = lambda: mock_cam
    app.dependency_overrides[get_gradcam_visualizer_dep] = lambda: mock_vis

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Successful Prediction & Explanation Workflows
# ==============================================================================


class TestSuccessfulExplanation:
    """Verify standard successful prediction + Grad-CAM explainability responses."""

    @pytest.mark.anyio
    async def test_successful_explain_default_predicted_class(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify POST /api/v1/predictions/explain defaults to predicted class."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}

        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["predicted_label"] == "Pneumonia"
        assert data["display_name"] == "Pneumonia"
        assert np.isclose(data["confidence"], 0.9854)
        assert len(data["probabilities"]) == 4

        # Selected Grad-CAM target defaults to predicted class (PNEUMONIA, index 2)
        assert data["selected_target_class"] == "PNEUMONIA"
        assert data["selected_target_label"] == "Pneumonia"
        assert data["target_class_index"] == 2

        # Base64 encodings
        assert data["heatmap_base64"].startswith("data:image/png;base64,")
        assert data["overlay_base64"].startswith("data:image/png;base64,")

        # Explanation metadata
        exp = data["explanation"]
        assert exp["target_layer"] == TARGET_CONV_LAYER
        assert exp["differentiation_target"] == "post-softmax probability S_c in [0.0, 1.0]"
        assert exp["status"] == "success"
        assert exp["alpha"] == 0.4
        assert exp["colormap"] == "jet"
        assert exp["processed_width"] == 224
        assert exp["processed_height"] == 224
        assert exp["inference_time_ms"] > 0.0
        assert exp["gradcam_time_ms"] >= 0.0
        assert exp["total_time_ms"] > 0.0

        # Disclaimer
        assert data["disclaimer"] == RESEARCH_DISCLAIMER

    @pytest.mark.anyio
    async def test_explicit_target_class_by_name(self, client: httpx.AsyncClient) -> None:
        """Verify targeting specific class by name (e.g. COVID19)."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}
        data_payload = {"target_class": "COVID19"}

        response = await client.post(
            "/api/v1/predictions/explain", files=files, data=data_payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["selected_target_class"] == "COVID19"
        assert data["selected_target_label"] == "COVID-19"
        assert data["target_class_index"] == 0

    @pytest.mark.anyio
    async def test_explicit_target_class_by_index(self, client: httpx.AsyncClient) -> None:
        """Verify targeting specific class by integer index string ('3' for TURBERCULOSIS)."""
        image_bytes = create_test_image_bytes(format_name="JPEG")
        files = {"image": ("chest_xray.jpg", image_bytes, "image/jpeg")}
        data_payload = {"target_class": "3"}

        response = await client.post(
            "/api/v1/predictions/explain", files=files, data=data_payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["selected_target_class"] == "TURBERCULOSIS"
        assert data["selected_target_label"] == "Tuberculosis"
        assert data["target_class_index"] == 3

    @pytest.mark.anyio
    async def test_explicit_target_class_via_query_param(self, client: httpx.AsyncClient) -> None:
        """Verify target_class supplied as query parameter is respected."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}

        response = await client.post(
            "/api/v1/predictions/explain?target_class=NORMAL", files=files
        )
        assert response.status_code == 200
        data = response.json()
        assert data["selected_target_class"] == "NORMAL"
        assert data["selected_target_label"] == "Normal"
        assert data["target_class_index"] == 1

    @pytest.mark.anyio
    async def test_custom_alpha_blend(self, client: httpx.AsyncClient) -> None:
        """Verify custom alpha factor is forwarded to visualizer."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}
        data_payload = {"alpha": "0.65"}

        response = await client.post(
            "/api/v1/predictions/explain", files=files, data=data_payload
        )
        assert response.status_code == 200
        data = response.json()
        assert np.isclose(data["explanation"]["alpha"], 0.65)

    @pytest.mark.anyio
    async def test_base64_images_are_valid_decodable_pngs(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify both heatmap_base64 and overlay_base64 decode to valid readable PNG images."""
        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("chest_xray.png", image_bytes, "image/png")}

        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 200
        data = response.json()

        prefix = "data:image/png;base64,"

        # 1. Verify heatmap PNG
        heatmap_uri = data["heatmap_base64"]
        assert heatmap_uri.startswith(prefix)
        heatmap_bytes = base64.b64decode(heatmap_uri[len(prefix) :])
        with Image.open(io.BytesIO(heatmap_bytes)) as h_img:
            assert h_img.format == "PNG"
            assert h_img.mode == "RGB"
            assert h_img.width > 0 and h_img.height > 0

        # 2. Verify overlay PNG
        overlay_uri = data["overlay_base64"]
        assert overlay_uri.startswith(prefix)
        overlay_bytes = base64.b64decode(overlay_uri[len(prefix) :])
        with Image.open(io.BytesIO(overlay_bytes)) as o_img:
            assert o_img.format == "PNG"
            assert o_img.mode == "RGB"
            assert o_img.width > 0 and o_img.height > 0


# ==============================================================================
# 2. Upload Validation & Parameter Checks
# ==============================================================================


class TestUploadValidation:
    """Verify input validation, boundary enforcement, and parameter checks."""

    @pytest.mark.anyio
    async def test_missing_image_field_returns_422(self, client: httpx.AsyncClient) -> None:
        """Verify omitting the required 'image' multipart field returns HTTP 422."""
        files = {"wrong_field": ("xray.png", b"data", "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["error_type"] == "ValidationError"

    @pytest.mark.anyio
    async def test_empty_image_upload_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify empty (0-byte) image upload returns HTTP 400 Bad Request."""
        files = {"image": ("empty.png", b"", "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "empty" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_unsupported_format_returns_415(self, client: httpx.AsyncClient) -> None:
        """Verify uploading unsupported formats (e.g. BMP) returns HTTP 415."""
        bmp_bytes = create_test_image_bytes(format_name="BMP")
        files = {"image": ("sample.bmp", bmp_bytes, "image/bmp")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 415
        data = response.json()
        assert data["status"] == "error"
        assert "unsupported" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_corrupt_image_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify corrupted image bytes return HTTP 400 Bad Request."""
        garbage_bytes = b"\x89PNG\r\n\x1a\n" + b"\xff" * 40  # Truncated PNG
        files = {"image": ("corrupted.png", garbage_bytes, "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"

    @pytest.mark.anyio
    async def test_oversized_upload_returns_413(self, client: httpx.AsyncClient) -> None:
        """Verify upload exceeding MAX_IMAGE_UPLOAD_BYTES returns HTTP 413."""
        settings = get_settings()
        oversized = b"\x00" * (settings.MAX_IMAGE_UPLOAD_BYTES + 10)
        files = {"image": ("oversized.png", oversized, "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 413
        data = response.json()
        assert data["status"] == "error"
        assert "exceeds" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_decompression_bomb_returns_413(self, client: httpx.AsyncClient) -> None:
        """Verify decompression bomb triggers HTTP 413 Request Entity Too Large."""
        class BombPreprocessor:
            def preprocess(self, content: bytes) -> Any:
                raise DecompressionBombError("Decompression bomb: exceeds 25,000,000 pixels.")

        app.dependency_overrides[get_image_preprocessor_dep] = lambda: BombPreprocessor()
        files = {"image": ("bomb.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 413
        data = response.json()
        assert "decompression bomb" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_invalid_target_class_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify unrecognized target class string returns HTTP 400."""
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post(
            "/api/v1/predictions/explain", files=files, data={"target_class": "NON_EXISTENT"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "unrecognized target class" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_target_class_out_of_bounds_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify out-of-range target class integer returns HTTP 400."""
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post(
            "/api/v1/predictions/explain", files=files, data={"target_class": "5"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "out of bounds" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_invalid_alpha_returns_400(self, client: httpx.AsyncClient) -> None:
        """Verify alpha < 0 or > 1 returns HTTP 400."""
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post(
            "/api/v1/predictions/explain", files=files, data={"alpha": "1.5"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "alpha" in data["error"]["message"].lower()


# ==============================================================================
# 3. Model & Explainability Failure Mappings
# ==============================================================================


class TestExplainabilityFailureMappings:
    """Verify internal errors map cleanly to appropriate HTTP status codes without leaking internals."""

    @pytest.mark.anyio
    async def test_model_unavailable_returns_503(self, client: httpx.AsyncClient) -> None:
        """Verify unloaded model returns HTTP 503 Service Unavailable."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=ModelUnavailableError("DenseNet121 model is not loaded.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "temporarily unavailable" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_invalid_model_output_returns_502(self, client: httpx.AsyncClient) -> None:
        """Verify invalid model output returns HTTP 502 Bad Gateway."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=InvalidModelOutputError("Output probabilities contain NaN.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 502
        data = response.json()
        assert data["status"] == "error"
        assert "validation failed" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_gradcam_gradient_error_returns_502(self, client: httpx.AsyncClient) -> None:
        """Verify Grad-CAM gradient failure (NaN/Inf) returns HTTP 502 Bad Gateway."""
        app.dependency_overrides[get_gradcam_service_dep] = lambda: MockGradCAMService(
            raise_exc=GradCAMGradientError("Gradients w.r.t. target layer contain NaN values.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 502
        data = response.json()
        assert data["status"] == "error"
        assert "gradient" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_gradcam_generation_error_returns_500(self, client: httpx.AsyncClient) -> None:
        """Verify Grad-CAM forward/backward crash returns HTTP 500 without leaking stack traces."""
        app.dependency_overrides[get_gradcam_service_dep] = lambda: MockGradCAMService(
            raise_exc=GradCAMGenerationError("Computational graph broken at layer 423.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "unexpected error occurred" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_visualization_failure_returns_500(self, client: httpx.AsyncClient) -> None:
        """Verify overlay rendering failure returns HTTP 500."""
        app.dependency_overrides[get_gradcam_visualizer_dep] = lambda: MockGradCAMVisualizer(
            raise_exc=VisualizerError("Alpha blending buffer overflow.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "rendering" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_inference_execution_error_returns_500_without_leaking_internals(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify inference execution crash returns HTTP 500 without leaking internal pointers."""
        app.dependency_overrides[get_inference_service_dep] = lambda: MockInferenceService(
            raise_exc=InferenceExecutionError("Internal CUDA memory fault at 0x7FFF00.")
        )
        files = {"image": ("xray.png", create_test_image_bytes(), "image/png")}
        response = await client.post("/api/v1/predictions/explain", files=files)

        assert response.status_code == 500
        data = response.json()
        assert "0x7FFF00" not in data["error"]["message"]


# ==============================================================================
# 4. OpenAPI Discovery & Async Concurrency Safety
# ==============================================================================


class TestOpenAPIDiscoveryAndAsyncSafety:
    """Verify OpenAPI documentation and threadpool execution."""

    @pytest.mark.anyio
    async def test_openapi_schema_contains_explain_endpoint(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify POST /api/v1/predictions/explain is documented in OpenAPI JSON."""
        response = await client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        paths = schema.get("paths", {})
        assert "/api/v1/predictions/explain" in paths
        post_op = paths["/api/v1/predictions/explain"].get("post", {})
        assert "Predictions" in post_op.get("tags", [])
        assert "Explainability" in post_op.get("tags", [])
        assert "200" in post_op.get("responses", {})


# ==============================================================================
# 5. Disk Safety & In-Memory Isolation
# ==============================================================================


class TestDiskSafety:
    """Verify explainability operations do not write temporary files to disk."""

    @pytest.mark.anyio
    async def test_no_files_written_to_disk(self, client: httpx.AsyncClient) -> None:
        """Verify processing an upload creates zero temporary files on the filesystem."""
        before_files = set(os.listdir("."))

        image_bytes = create_test_image_bytes(format_name="PNG")
        files = {"image": ("disk_check.png", image_bytes, "image/png")}

        response = await client.post("/api/v1/predictions/explain", files=files)
        assert response.status_code == 200

        after_files = set(os.listdir("."))
        new_files = after_files - before_files
        assert len(new_files) == 0, f"Unexpected files written to disk: {new_files}"


# ==============================================================================
# 6. Real Model End-to-End Explain Smoke Test
# ==============================================================================


class TestRealModelExplainSmoke:
    """End-to-end integration smoke test with the actual loaded DenseNet121 model."""

    @pytest.mark.anyio
    async def test_real_model_explain_smoke(self) -> None:
        """Execute POST /api/v1/predictions/explain using the real model without mocks."""
        loader = get_model_loader()
        try:
            model = loader.load_model()
            assert model is not None

            # Clear any dependency overrides to exercise real services
            app.dependency_overrides.clear()

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as real_client:
                image_bytes = create_test_image_bytes(format_name="PNG", size=(224, 224))
                files = {"image": ("real_xray.png", image_bytes, "image/png")}

                response = await real_client.post("/api/v1/predictions/explain", files=files)
                assert response.status_code == 200

                data = response.json()
                assert data["predicted_class"] in CLASS_INDICES.values()
                assert data["selected_target_class"] == data["predicted_class"]
                assert data["heatmap_base64"].startswith("data:image/png;base64,")
                assert data["overlay_base64"].startswith("data:image/png;base64,")
                assert data["explanation"]["target_layer"] == TARGET_CONV_LAYER
                assert data["explanation"]["differentiation_target"] == "post-softmax probability S_c in [0.0, 1.0]"
                assert data["explanation"]["inference_time_ms"] > 0.0
                assert data["explanation"]["gradcam_time_ms"] > 0.0
                assert data["disclaimer"] == RESEARCH_DISCLAIMER
        finally:
            loader.unload()
            app.dependency_overrides.clear()
