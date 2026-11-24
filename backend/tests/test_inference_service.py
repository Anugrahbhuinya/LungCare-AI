"""Comprehensive unit, integration, and contract test suite for InferenceService.

Validates:
1. Four-class prediction execution and schema contract satisfaction.
2. Index-to-class and display-name mappings (including the intentional TURBERCULOSIS spelling).
3. Defensive output tensor contract validation (shape, finiteness, bounds, sum=1.0).
4. Service integration with ImagePreprocessor and ModelLoader.
5. Deterministic reproducibility across repeated invocations.
6. Monotonic latency telemetry (inference_time_ms, preprocessing_time_ms).
7. End-to-end integration with the real loaded DenseNet121 model.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image
import pytest

from app.services.image_preprocessor import (
    EmptyImageError,
    ImagePreprocessor,
    PreprocessedImageResult,
    UnsupportedFormatError,
)
from app.services.inference_service import (
    DEFAULT_CLASS_MAPPING,
    DEFAULT_DISPLAY_NAMES,
    RESEARCH_DISCLAIMER,
    InferenceError,
    InferenceExecutionError,
    InferenceService,
    InvalidModelOutputError,
    ModelUnavailableError,
    PredictionResult,
    get_inference_service,
)
from app.services.model_loader import ModelLoader, get_model_loader


# ==============================================================================
# Test Doubles (Mocks for Fast, Deterministic Unit Testing)
# ==============================================================================


class MockKerasModel:
    """Controllable mock of a Keras model for unit testing forward pass behaviors."""

    def __init__(
        self,
        output_array: Optional[np.ndarray] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.output_array = (
            output_array
            if output_array is not None
            else np.array([[0.05, 0.10, 0.80, 0.05]], dtype=np.float32)
        )
        self.raise_exc = raise_exc
        self.call_count = 0
        self.name = "MockDenseNet121"

    def __call__(self, tensor: Any, training: bool = False) -> Any:
        self.call_count += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.output_array


class MockModelLoader:
    """Controllable mock of ModelLoader avoiding disk I/O in unit tests."""

    def __init__(
        self,
        model: Optional[Any] = None,
        raise_exc: Optional[Exception] = None,
        config_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._raise_exc = raise_exc
        self.get_model_calls = 0
        self.metadata = None
        self.is_loaded = model is not None

        if config_data:
            from dataclasses import make_dataclass
            from pathlib import Path

            ArtifactContainer = make_dataclass(
                "ArtifactContainer", [("config_data", dict), ("model_dir", Path)]
            )
            self._artifacts = ArtifactContainer(config_data=config_data, model_dir=Path("."))
        else:
            self._artifacts = None

    def get_model(self) -> Any:
        self.get_model_calls += 1
        if self._raise_exc:
            raise self._raise_exc
        return self._model


class MockPreprocessor:
    """Controllable mock of ImagePreprocessor avoiding full image decode cycles."""

    def __init__(
        self,
        tensor: Optional[np.ndarray] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.tensor = (
            tensor
            if tensor is not None
            else np.zeros((1, 224, 224, 3), dtype=np.float32)
        )
        self.raise_exc = raise_exc
        self.preprocess_calls = 0

    def preprocess(self, image_bytes: bytes) -> PreprocessedImageResult:
        self.preprocess_calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return PreprocessedImageResult(
            tensor=self.tensor,
            original_width=640,
            original_height=480,
            processed_width=224,
            processed_height=224,
            original_format="PNG",
            original_mode="RGB",
            final_mode="RGB",
            exif_transposed=False,
            alpha_composited=False,
        )


def create_synthetic_png_bytes(size: tuple[int, int] = (100, 100), color: tuple = (100, 100, 100)) -> bytes:
    """Generate in-memory synthetic PNG image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ==============================================================================
# 1. Successful Inference & Result Contracts
# ==============================================================================


class TestSuccessfulInference:
    """Verify standard four-class prediction lifecycle and contract satisfaction."""

    def test_predict_returns_valid_prediction_result(self) -> None:
        """Verify successful prediction returns a fully populated PredictionResult."""
        # Simulated model output: class 2 (PNEUMONIA) is highest
        mock_output = np.array([[0.05, 0.10, 0.80, 0.05]], dtype=np.float32)
        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        result = service.predict(create_synthetic_png_bytes())

        assert isinstance(result, PredictionResult)
        assert result.predicted_class == "PNEUMONIA"
        assert result.predicted_label == "Pneumonia"
        assert np.isclose(result.confidence, 0.80)
        assert len(result.probabilities) == 4
        assert result.probabilities["COVID19"] == pytest.approx(0.05)
        assert result.probabilities["NORMAL"] == pytest.approx(0.10)
        assert result.probabilities["PNEUMONIA"] == pytest.approx(0.80)
        assert result.probabilities["TURBERCULOSIS"] == pytest.approx(0.05)
        assert result.model_architecture == "DenseNet121"
        assert result.input_width == 640
        assert result.input_height == 480
        assert result.processed_width == 224
        assert result.processed_height == 224
        assert result.inference_time_ms >= 0.0
        assert result.preprocessing_time_ms >= 0.0
        assert result.disclaimer == RESEARCH_DISCLAIMER

    def test_prediction_result_to_dict(self) -> None:
        """Verify result.to_dict() returns a JSON-serializable telemetry payload."""
        mock_output = np.array([[0.70, 0.10, 0.10, 0.10]], dtype=np.float32)
        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        result = service.predict(b"dummy_bytes")
        data = result.to_dict()

        assert data["predicted_class"] == "COVID19"
        assert data["predicted_label"] == "COVID-19"
        assert data["confidence"] == pytest.approx(0.70)
        assert isinstance(data["probabilities"], dict)
        assert set(data["probabilities"].keys()) == {"COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"}
        assert "disclaimer" in data


# ==============================================================================
# 2. Class Mapping Verification
# ==============================================================================


class TestClassMapping:
    """Verify exact index mapping for all four radiological classes."""

    @pytest.mark.parametrize(
        ("argmax_idx", "expected_class", "expected_label"),
        [
            (0, "COVID19", "COVID-19"),
            (1, "NORMAL", "Normal"),
            (2, "PNEUMONIA", "Pneumonia"),
            (3, "TURBERCULOSIS", "Tuberculosis"),
        ],
    )
    def test_each_class_index_maps_correctly(
        self,
        argmax_idx: int,
        expected_class: str,
        expected_label: str,
    ) -> None:
        """Verify each output index (0..3) maps to its correct internal key and display label."""
        probs = [0.05, 0.05, 0.05, 0.05]
        probs[argmax_idx] = 0.85
        mock_output = np.array([probs], dtype=np.float32)

        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        result = service.predict(b"dummy_bytes")

        assert result.predicted_class == expected_class
        assert result.predicted_label == expected_label
        assert np.isclose(result.confidence, 0.85)

    def test_turberculosis_intentional_spelling_preserved(self) -> None:
        """Specifically verify internal TURBERCULOSIS spelling is preserved and maps to Tuberculosis."""
        mock_output = np.array([[0.05, 0.05, 0.05, 0.85]], dtype=np.float32)
        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        result = service.predict(b"dummy_bytes")

        assert result.predicted_class == "TURBERCULOSIS"
        assert result.predicted_label == "Tuberculosis"
        assert "TURBERCULOSIS" in result.probabilities


# ==============================================================================
# 3. Defensive Output Tensor Validation
# ==============================================================================


class TestOutputTensorValidation:
    """Verify strict validation of model forward pass outputs."""

    def test_incorrect_output_shape_3_classes_rejected(self) -> None:
        """Verify (1, 3) shape raises InvalidModelOutputError."""
        invalid_output = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"
        assert "(1, 4)" in str(exc_info.value)

    def test_incorrect_output_shape_5_classes_rejected(self) -> None:
        """Verify (1, 5) shape raises InvalidModelOutputError."""
        invalid_output = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"

    def test_1d_output_shape_rejected(self) -> None:
        """Verify unbatched 1D (4,) shape raises InvalidModelOutputError."""
        invalid_output = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"

    def test_nan_output_rejected(self) -> None:
        """Verify NaN values in output raise InvalidModelOutputError."""
        invalid_output = np.array([[0.1, np.nan, 0.5, 0.4]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"
        assert "non-finite" in str(exc_info.value).lower()

    def test_inf_output_rejected(self) -> None:
        """Verify Infinity values in output raise InvalidModelOutputError."""
        invalid_output = np.array([[0.1, np.inf, 0.5, 0.4]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"

    def test_negative_probabilities_rejected(self) -> None:
        """Verify negative probabilities raise InvalidModelOutputError."""
        invalid_output = np.array([[-0.1, 0.6, 0.3, 0.2]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"
        assert "outside valid range" in str(exc_info.value).lower()

    def test_probability_greater_than_one_rejected(self) -> None:
        """Verify probabilities exceeding 1.0 raise InvalidModelOutputError."""
        invalid_output = np.array([[1.1, 0.0, 0.0, 0.0]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"

    def test_probability_sum_divergence_rejected(self) -> None:
        """Verify probability distribution that fails sum-to-1.0 raises InvalidModelOutputError."""
        # Sum is 0.70 (violates sum=1.0)
        invalid_output = np.array([[0.2, 0.2, 0.2, 0.1]], dtype=np.float32)
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel(output_array=invalid_output)),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InvalidModelOutputError) as exc_info:
            service.predict(b"dummy")
        assert exc_info.value.error_code == "INVALID_MODEL_OUTPUT"
        assert "do not sum to 1.0" in str(exc_info.value)


# ==============================================================================
# 4. Service Integration & Error Propagation
# ==============================================================================


class TestServiceIntegration:
    """Verify integration boundaries between preprocessor, loader, and inference service."""

    def test_preprocessor_called_exactly_once(self) -> None:
        """Verify preprocessor is invoked exactly once per predict() invocation."""
        mock_prep = MockPreprocessor()
        mock_loader = MockModelLoader(model=MockKerasModel())

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        service.predict(b"test_bytes")

        assert mock_prep.preprocess_calls == 1

    def test_model_reused_across_multiple_calls_not_reloaded(self) -> None:
        """Verify the loaded model singleton is reused across requests without reinitialization."""
        mock_model = MockKerasModel()
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore

        service.predict(b"request_1")
        service.predict(b"request_2")
        service.predict(b"request_3")

        # Model forward pass called 3 times on the SAME model instance
        assert mock_model.call_count == 3
        # Loader get_model called 3 times (fast memory pointer resolution, no disk reload)
        assert mock_loader.get_model_calls == 3

    def test_preprocessing_errors_propagated_transparently(self) -> None:
        """Verify ImagePreprocessingErrors from preprocessor bubble up cleanly."""
        mock_prep = MockPreprocessor(raise_exc=EmptyImageError("Payload is empty."))
        service = InferenceService(
            model_loader=MockModelLoader(model=MockKerasModel()),  # type: ignore
            preprocessor=mock_prep,  # type: ignore
        )

        with pytest.raises(EmptyImageError) as exc_info:
            service.predict(b"")

        assert exc_info.value.error_code == "EMPTY_IMAGE"

    def test_model_unavailable_error_when_loader_fails(self) -> None:
        """Verify ModelUnavailableError is raised if ModelLoader fails to provide model."""
        mock_loader = MockModelLoader(raise_exc=RuntimeError("Disk read failure."))
        service = InferenceService(
            model_loader=mock_loader,  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(ModelUnavailableError) as exc_info:
            service.predict(b"valid_bytes")

        assert exc_info.value.error_code == "MODEL_UNAVAILABLE"
        assert "Disk read failure" in str(exc_info.value)

    def test_model_unavailable_error_when_model_is_none(self) -> None:
        """Verify ModelUnavailableError is raised if ModelLoader returns None."""
        mock_loader = MockModelLoader(model=None)
        service = InferenceService(
            model_loader=mock_loader,  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(ModelUnavailableError) as exc_info:
            service.predict(b"valid_bytes")

        assert exc_info.value.error_code == "MODEL_UNAVAILABLE"

    def test_forward_pass_runtime_error_converted_to_inference_execution_error(self) -> None:
        """Verify unhandled forward pass exceptions raise InferenceExecutionError."""
        mock_model = MockKerasModel(raise_exc=RuntimeError("TensorFlow CUDA out of memory."))
        service = InferenceService(
            model_loader=MockModelLoader(model=mock_model),  # type: ignore
            preprocessor=MockPreprocessor(),  # type: ignore
        )

        with pytest.raises(InferenceExecutionError) as exc_info:
            service.predict(b"valid_bytes")

        assert exc_info.value.error_code == "INFERENCE_EXECUTION_ERROR"
        assert "CUDA out of memory" in str(exc_info.value)


# ==============================================================================
# 5. Determinism & Timing Telemetry
# ==============================================================================


class TestDeterminismAndTiming:
    """Verify deterministic outputs and monotonic high-resolution latency tracking."""

    def test_deterministic_output_for_identical_calls(self) -> None:
        """Verify identical inputs through a deterministic model yield bitwise-equal outputs."""
        mock_output = np.array([[0.10, 0.70, 0.15, 0.05]], dtype=np.float32)
        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore

        res1 = service.predict(b"fixed_payload")
        res2 = service.predict(b"fixed_payload")

        assert res1.predicted_class == res2.predicted_class
        assert res1.predicted_label == res2.predicted_label
        assert res1.confidence == res2.confidence
        assert res1.probabilities == res2.probabilities

    def test_latency_telemetry_is_nonnegative_and_plausible(self) -> None:
        """Verify inference_time_ms and preprocessing_time_ms are recorded as positive floats."""
        mock_output = np.array([[0.10, 0.10, 0.70, 0.10]], dtype=np.float32)
        mock_model = MockKerasModel(output_array=mock_output)
        mock_loader = MockModelLoader(model=mock_model)
        mock_prep = MockPreprocessor()

        service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep)  # type: ignore
        result = service.predict(b"dummy")

        assert isinstance(result.inference_time_ms, float)
        assert isinstance(result.preprocessing_time_ms, float)
        assert result.inference_time_ms >= 0.0
        assert result.preprocessing_time_ms >= 0.0

    def test_singleton_accessor(self) -> None:
        """Verify get_inference_service returns a persistent singleton."""
        s1 = get_inference_service()
        s2 = get_inference_service()
        assert s1 is s2


# ==============================================================================
# 6. Real Model End-to-End Integration Smoke Test
# ==============================================================================


class TestRealModelIntegration:
    """Verify end-to-end integration with the real loaded DenseNet121 model artifact."""

    def test_real_model_inference_on_synthetic_xray(self) -> None:
        """Execute full end-to-end pipeline with real ModelLoader and real ImagePreprocessor."""
        # 1. Initialize real application singletons
        real_loader = get_model_loader()
        # Ensure model is initialized
        real_loader.load_model()
        assert real_loader.is_loaded is True

        real_preprocessor = ImagePreprocessor()
        service = InferenceService(model_loader=real_loader, preprocessor=real_preprocessor)

        # 2. Create a synthetic 300x300 chest radiograph grayscale/RGB image
        synthetic_bytes = create_synthetic_png_bytes(size=(300, 300), color=(128, 128, 128))

        # 3. Execute prediction
        result = service.predict(synthetic_bytes)

        # 4. Validate contracts
        assert isinstance(result, PredictionResult)
        assert result.predicted_class in ("COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS")
        assert result.predicted_label in ("COVID-19", "Normal", "Pneumonia", "Tuberculosis")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 4
        assert pytest.approx(sum(result.probabilities.values()), abs=1e-3) == 1.0
        assert result.processed_width == 224
        assert result.processed_height == 224
        assert result.inference_time_ms > 0.0
        assert result.preprocessing_time_ms > 0.0
        assert result.disclaimer == RESEARCH_DISCLAIMER

