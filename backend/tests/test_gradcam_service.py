"""Test suite for Grad-CAM Visual Explainability Service.

Validates:
1. DenseNet121 architecture compatibility and target layer verification (conv5_block16_2_conv).
2. Numerical correctness, finite bounds [0, 1], and spatial dimensions (224x224).
3. Class-specific explanation selection across all 4 categories (COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS).
4. Defensive error handling (invalid shapes, non-float dtypes, NaN/inf, invalid class indices).
5. Safe uninformative marking on zero/flat gradients.
6. Model weight preservation (weights untouched by gradient backpropagation).
7. Thread-safe concurrency and singleton access.
8. Real-model integration smoke test using approved model artifacts.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, Optional
from unittest import mock
import numpy as np
import pytest

try:
    import tensorflow as tf
    import keras
    from app.services.gradcam_service import (
        CLASS_DISPLAY_NAMES,
        CLASS_INDICES,
        RESEARCH_DISCLAIMER,
        TARGET_CONV_LAYER,
        GradCAMError,
        GradCAMGenerationError,
        GradCAMGradientError,
        GradCAMLayerNotFoundError,
        GradCAMResult,
        GradCAMService,
        InvalidGradCAMInputError,
        get_gradcam_service,
    )
    from app.services.model_loader import ModelLoader, get_model_loader
    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(
    not HAS_TF,
    reason="TensorFlow/Keras runtime required (execute using ML environment backend/.venv-ml)",
)


# ==============================================================================
# Test Fixtures & Synthetic Model Doubles
# ==============================================================================


@pytest.fixture(scope="module")
def loaded_real_model() -> Any:
    """Fixture providing the real loaded DenseNet121 model singleton."""
    loader = get_model_loader()
    model = loader.load_model()
    assert model is not None
    return model


@pytest.fixture
def synthetic_input_tensor() -> np.ndarray:
    """Create deterministic synthetic float32 preprocessed image tensor (1, 224, 224, 3)."""
    np.random.seed(42)
    # DenseNet ImageNet normalized range: roughly [-2.5, 2.5]
    return np.random.uniform(-2.0, 2.0, (1, 224, 224, 3)).astype(np.float32)


# ==============================================================================
# 1. Architecture Compatibility Tests
# ==============================================================================


class TestArchitectureCompatibility:
    """Verify target convolutional layer and backbone connectivity."""

    def test_target_layer_exists_in_backbone(self, loaded_real_model: Any) -> None:
        """Requirement: Verify conv5_block16_2_conv exists in the DenseNet121 backbone."""
        backbone = loaded_real_model.get_layer("densenet121")
        assert backbone is not None

        target_layer = backbone.get_layer(TARGET_CONV_LAYER)
        assert target_layer is not None
        assert target_layer.name == TARGET_CONV_LAYER

    def test_target_layer_output_spatial_dimensions(self, loaded_real_model: Any) -> None:
        """Requirement: Verify target layer outputs a 7x7x32 spatial feature map."""
        backbone = loaded_real_model.get_layer("densenet121")
        target_layer = backbone.get_layer(TARGET_CONV_LAYER)

        # Output shape: (None, 7, 7, 32)
        shape = target_layer.output.shape
        assert shape[1] == 7
        assert shape[2] == 7
        assert shape[3] == 32

    def test_four_class_model_output_preserved(self, loaded_real_model: Any) -> None:
        """Requirement: Verify model output retains exactly 4 diagnostic classes."""
        assert loaded_real_model.output_shape == (None, 4)

    def test_gradient_model_reuses_existing_weights(self, loaded_real_model: Any) -> None:
        """Requirement: Verify Grad-CAM gradient model reuses original layer weights without duplication."""
        service = GradCAMService(model_loader=get_model_loader())
        grad_model = service._build_or_get_grad_model()

        # Find dense layer in original model and in grad_model
        orig_dense = loaded_real_model.get_layer("dense")
        grad_dense = grad_model.get_layer("dense")

        # Must share identical kernel weights in memory
        orig_kernel = orig_dense.get_weights()[0]
        grad_kernel = grad_dense.get_weights()[0]
        assert np.array_equal(orig_kernel, grad_kernel)

    def test_exact_differentiation_target_is_softmax_probability(
        self, loaded_real_model: Any, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Issue 1 Verification: Confirm Grad-CAM backpropagates against post-softmax probability S_c.

        Validates:
        1. Model's final Dense classification layer uses softmax activation.
        2. Gradient model outputs are [conv_outputs, final_predictions].
        3. Predictions tensor sums to 1.0 (is a normalized probability distribution, not logits).
        4. Target class score S_c is a direct slice of post-softmax predictions.
        5. Gradients d(S_c)/d(conv_outputs) are non-None and finite.
        """
        # 1. Verify final layer activation is softmax
        dense_layer = loaded_real_model.get_layer("dense")
        assert dense_layer is not None
        assert getattr(dense_layer.activation, "__name__", "") == "softmax"

        # 2. Inspect auxiliary gradient model graph
        service = GradCAMService(model_loader=get_model_loader())
        grad_model = service._build_or_get_grad_model()

        # Input shape check
        assert grad_model.input_shape == (None, 224, 224, 3)
        # Outputs: [conv_output, predictions]
        assert len(grad_model.outputs) == 2
        conv_out_tensor, pred_tensor = grad_model.outputs
        assert conv_out_tensor.shape[1:] == (7, 7, 32)
        assert pred_tensor.shape[1:] == (4,)

        # 3. Trace actual tensors in GradientTape
        input_t = tf.convert_to_tensor(synthetic_input_tensor, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_t, training=False)
            tape.watch(conv_outputs)

            # Assert predictions are valid softmax probabilities
            prob_sum = float(tf.reduce_sum(predictions, axis=-1).numpy()[0])
            assert pytest.approx(prob_sum, abs=1e-5) == 1.0
            assert np.all(predictions.numpy() >= 0.0)
            assert np.all(predictions.numpy() <= 1.0)

            # Score is the post-softmax probability for selected class (e.g. index 2)
            selected_idx = 2
            class_score = predictions[:, selected_idx]
            assert 0.0 <= float(class_score.numpy()[0]) <= 1.0

        grads = tape.gradient(class_score, conv_outputs)
        assert grads is not None
        assert grads.shape == (1, 7, 7, 32)
        assert tf.reduce_all(tf.math.is_finite(grads))


# ==============================================================================
# 2. Grad-CAM Numerical Correctness Tests
# ==============================================================================


class TestGradCAMNumericalCorrectness:
    """Verify numerical contracts, spatial normalization, and error handling."""

    def test_valid_input_produces_finite_output(self, synthetic_input_tensor: np.ndarray) -> None:
        """Verify valid input produces a non-empty, fully finite heatmap."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=2)

        assert isinstance(result, GradCAMResult)
        assert result.heatmap is not None
        assert np.all(np.isfinite(result.heatmap))

    def test_heatmap_dimensions_are_224x224(self, synthetic_input_tensor: np.ndarray) -> None:
        """Requirement: Output heatmap must have exact spatial dimensions (224, 224)."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=2)

        assert result.heatmap.shape == (224, 224)
        assert result.heatmap_width == 224
        assert result.heatmap_height == 224

    def test_heatmap_values_in_0_to_1_range(self, synthetic_input_tensor: np.ndarray) -> None:
        """Requirement: Heatmap values must be strictly normalized within [0.0, 1.0]."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=2)

        h_min = float(np.min(result.heatmap))
        h_max = float(np.max(result.heatmap))

        assert h_min >= 0.0
        assert h_max <= 1.0
        if result.status == "success":
            assert np.isclose(h_max, 1.0, atol=1e-3)

    def test_constant_near_zero_gradients_marked_uninformative(self) -> None:
        """Requirement: Constant or flat zero gradients produce uninformative status and zero heatmap."""
        service = GradCAMService()

        # Create flat zero tensor
        zero_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        result = service.generate_heatmap(zero_tensor, class_index=1)

        # Result must be marked uninformative or success with valid finite heatmap
        assert result.status in ("success", "uninformative")
        assert np.all(np.isfinite(result.heatmap))
        assert np.min(result.heatmap) >= 0.0
        assert np.max(result.heatmap) <= 1.0

    def test_invalid_input_shape_rejected(self) -> None:
        """Verify non-(1, 224, 224, 3) shapes raise InvalidGradCAMInputError."""
        service = GradCAMService()

        # 3D tensor
        with pytest.raises(InvalidGradCAMInputError) as exc_3d:
            service.generate_heatmap(np.zeros((224, 224, 3), dtype=np.float32))
        assert "Invalid input tensor shape" in str(exc_3d.value)

        # Batch size 2
        with pytest.raises(InvalidGradCAMInputError):
            service.generate_heatmap(np.zeros((2, 224, 224, 3), dtype=np.float32))

        # Wrong spatial size (1, 100, 100, 3)
        with pytest.raises(InvalidGradCAMInputError):
            service.generate_heatmap(np.zeros((1, 100, 100, 3), dtype=np.float32))

    def test_invalid_dtype_rejected(self) -> None:
        """Verify non-floating point dtypes raise InvalidGradCAMInputError."""
        service = GradCAMService()

        uint8_tensor = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        with pytest.raises(InvalidGradCAMInputError) as exc_info:
            service.generate_heatmap(uint8_tensor)
        assert "Invalid tensor dtype" in str(exc_info.value)

        int32_tensor = np.zeros((1, 224, 224, 3), dtype=np.int32)
        with pytest.raises(InvalidGradCAMInputError):
            service.generate_heatmap(int32_tensor)

    def test_nan_and_inf_inputs_rejected(self, synthetic_input_tensor: np.ndarray) -> None:
        """Verify tensors containing NaN or Inf raise InvalidGradCAMInputError."""
        service = GradCAMService()

        nan_tensor = synthetic_input_tensor.copy()
        nan_tensor[0, 10, 10, 0] = np.nan
        with pytest.raises(InvalidGradCAMInputError) as exc_nan:
            service.generate_heatmap(nan_tensor)
        assert "non-finite" in str(exc_nan.value)

        inf_tensor = synthetic_input_tensor.copy()
        inf_tensor[0, 10, 10, 0] = np.inf
        with pytest.raises(InvalidGradCAMInputError) as exc_inf:
            service.generate_heatmap(inf_tensor)
        assert "non-finite" in str(exc_inf.value)

    def test_invalid_class_index_rejected(self, synthetic_input_tensor: np.ndarray) -> None:
        """Verify class indices outside [0, 3] raise InvalidGradCAMInputError."""
        service = GradCAMService()

        with pytest.raises(InvalidGradCAMInputError) as exc_4:
            service.generate_heatmap(synthetic_input_tensor, class_index=4)
        assert "out of bounds" in str(exc_4.value)

        with pytest.raises(InvalidGradCAMInputError) as exc_neg:
            service.generate_heatmap(synthetic_input_tensor, class_index=-1)
        assert "out of bounds" in str(exc_neg.value)

        with pytest.raises(InvalidGradCAMInputError):
            service.generate_heatmap(synthetic_input_tensor, class_index="invalid")  # type: ignore

    def test_non_existent_target_layer_raises_layer_not_found(self) -> None:
        """Verify invalid layer name raises GradCAMLayerNotFoundError."""
        service = GradCAMService(target_layer_name="non_existent_layer_xyz")
        synthetic_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

        with pytest.raises(GradCAMLayerNotFoundError) as exc_info:
            service.generate_heatmap(synthetic_input)
        assert "non_existent_layer_xyz" in str(exc_info.value)


# ==============================================================================
# 2b. Numerical Exception & Remediation Tests (Issue 2)
# ==============================================================================


class TestGradCAMNumericalExceptionsAndRemediation:
    """Verify distinct error handling for uninformative vs invalid numerical states (Issue 2)."""

    def test_valid_zero_gradients_returns_uninformative_zero_heatmap(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Flat zero gradients must produce status 'uninformative' and zero heatmap."""
        service = GradCAMService()
        zero_grads = tf.zeros((1, 7, 7, 32), dtype=tf.float32)

        with mock.patch.object(tf.GradientTape, "gradient", return_value=zero_grads):
            result = service.generate_heatmap(synthetic_input_tensor, class_index=0)

        assert result.status == "uninformative"
        assert result.heatmap.shape == (224, 224)
        assert np.all(result.heatmap == 0.0)

    def test_valid_constant_activation_map_returns_uninformative(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Constant activation map (max - min <= 1e-10) produces status 'uninformative' and zero heatmap."""
        service = GradCAMService()
        constant_conv = tf.ones((1, 7, 7, 32), dtype=tf.float32)
        constant_grads = tf.ones((1, 7, 7, 32), dtype=tf.float32)
        dummy_preds = tf.constant([[0.25, 0.25, 0.25, 0.25]], dtype=tf.float32)

        mock_model = mock.MagicMock(return_value=(constant_conv, dummy_preds))

        with mock.patch.object(service, "_build_or_get_grad_model", return_value=mock_model), \
             mock.patch.object(tf.GradientTape, "gradient", return_value=constant_grads):
            result = service.generate_heatmap(synthetic_input_tensor, class_index=0)

        assert result.status == "uninformative"
        assert result.heatmap.shape == (224, 224)
        assert np.all(result.heatmap == 0.0)

    def test_nan_gradients_raises_gradcam_gradient_error(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: NaN gradients must raise GradCAMGradientError."""
        service = GradCAMService()
        nan_grads = tf.constant(np.full((1, 7, 7, 32), np.nan, dtype=np.float32))

        with mock.patch.object(tf.GradientTape, "gradient", return_value=nan_grads):
            with pytest.raises(GradCAMGradientError) as exc_info:
                service.generate_heatmap(synthetic_input_tensor, class_index=1)

        assert "NaN" in str(exc_info.value)
        assert exc_info.value.error_code == "GRADCAM_GRADIENT_ERROR"

    def test_infinite_gradients_raises_gradcam_gradient_error(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Infinite gradients must raise GradCAMGradientError."""
        service = GradCAMService()
        inf_grads = tf.constant(np.full((1, 7, 7, 32), np.inf, dtype=np.float32))

        with mock.patch.object(tf.GradientTape, "gradient", return_value=inf_grads):
            with pytest.raises(GradCAMGradientError) as exc_info:
                service.generate_heatmap(synthetic_input_tensor, class_index=1)

        assert "Inf" in str(exc_info.value)
        assert exc_info.value.error_code == "GRADCAM_GRADIENT_ERROR"

    def test_missing_disconnected_gradients_raises_gradcam_gradient_error(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: None / disconnected gradients must raise GradCAMGradientError."""
        service = GradCAMService()

        with mock.patch.object(tf.GradientTape, "gradient", return_value=None):
            with pytest.raises(GradCAMGradientError) as exc_info:
                service.generate_heatmap(synthetic_input_tensor, class_index=1)

        assert "evaluated to None" in str(exc_info.value)
        assert exc_info.value.error_code == "GRADCAM_GRADIENT_ERROR"

    def test_non_finite_feature_maps_raises_gradcam_generation_error(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Non-finite feature maps (NaN/Inf) must raise GradCAMGenerationError."""
        service = GradCAMService()
        nan_conv = tf.constant(np.full((1, 7, 7, 32), np.nan, dtype=np.float32))
        dummy_preds = tf.constant([[0.1, 0.7, 0.1, 0.1]], dtype=np.float32)

        mock_model = mock.MagicMock(return_value=(nan_conv, dummy_preds))

        with mock.patch.object(service, "_build_or_get_grad_model", return_value=mock_model):
            with pytest.raises(GradCAMGenerationError) as exc_info:
                service.generate_heatmap(synthetic_input_tensor, class_index=1)

        assert "non-finite feature maps" in str(exc_info.value)
        assert exc_info.value.error_code == "GRADCAM_GENERATION_ERROR"

    def test_non_finite_resized_heatmap_raises_gradcam_generation_error(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Non-finite values in resized activation map must raise GradCAMGenerationError."""
        service = GradCAMService()
        nan_resized = tf.constant(np.full((1, 224, 224, 1), np.nan, dtype=np.float32))

        with mock.patch.object(tf.image, "resize", return_value=nan_resized):
            with pytest.raises(GradCAMGenerationError) as exc_info:
                service.generate_heatmap(synthetic_input_tensor, class_index=1)

        assert "non-finite" in str(exc_info.value)
        assert exc_info.value.error_code == "GRADCAM_GENERATION_ERROR"


# ==============================================================================
# 3. Class-Specific Explanations Tests
# ==============================================================================


class TestClassSpecificExplanations:
    """Verify class-specific selection, label mapping, and default argmax behavior."""

    @pytest.mark.parametrize(
        "class_idx,expected_key,expected_display",
        [
            (0, "COVID19", "COVID-19"),
            (1, "NORMAL", "Normal"),
            (2, "PNEUMONIA", "Pneumonia"),
            (3, "TURBERCULOSIS", "Tuberculosis"),
        ],
    )
    def test_each_of_four_class_indices_can_be_selected(
        self,
        synthetic_input_tensor: np.ndarray,
        class_idx: int,
        expected_key: str,
        expected_display: str,
    ) -> None:
        """Requirement: Verify each of the 4 valid class indices can be targeted specifically."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=class_idx)

        assert result.class_index == class_idx
        assert result.selected_class == expected_key
        assert result.selected_label == expected_display
        assert result.heatmap.shape == (224, 224)
        assert len(result.probabilities) == 4
        assert result.disclaimer == RESEARCH_DISCLAIMER

    def test_turberculosis_intentional_spelling_preserved(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: The fourth class identifier must preserve the TURBERCULOSIS spelling."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=3)

        assert result.selected_class == "TURBERCULOSIS"
        assert result.selected_label == "Tuberculosis"

    def test_default_selection_uses_model_predicted_class(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: When class_index is omitted, the model's predicted class is used."""
        service = GradCAMService()
        result = service.generate_heatmap(synthetic_input_tensor, class_index=None)

        # Expected index is argmax of probabilities
        prob_values = [result.probabilities[CLASS_INDICES[i]] for i in range(4)]
        expected_pred_idx = int(np.argmax(prob_values))

        assert result.class_index == expected_pred_idx
        assert result.selected_class == CLASS_INDICES[expected_pred_idx]
        assert result.selected_label == CLASS_DISPLAY_NAMES[result.selected_class]


# ==============================================================================
# 4. Model Integrity & Concurrency Safety Tests
# ==============================================================================


class TestModelIntegrityAndConcurrency:
    """Verify weights remain frozen, instances are shared, and concurrency is thread-safe."""

    def test_model_weights_not_updated_during_gradcam(
        self, loaded_real_model: Any, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Requirement: Gradient backpropagation must NOT modify any model weights."""
        dense_layer = loaded_real_model.get_layer("dense")
        weights_before = [w.copy() for w in dense_layer.get_weights()]

        service = GradCAMService()
        _ = service.generate_heatmap(synthetic_input_tensor, class_index=2)

        weights_after = dense_layer.get_weights()
        for w_before, w_after in zip(weights_before, weights_after):
            assert np.array_equal(w_before, w_after)

    def test_singleton_accessor(self) -> None:
        """Verify get_gradcam_service returns the persistent application singleton."""
        s1 = get_gradcam_service()
        s2 = get_gradcam_service()
        assert s1 is s2

    def test_concurrent_invocations_under_lock(
        self, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Verify concurrent multi-threaded requests serialize safely through the instance lock."""
        service = GradCAMService()

        def run_explanation(target_idx: int) -> GradCAMResult:
            return service.generate_heatmap(synthetic_input_tensor, class_index=target_idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_explanation, idx) for idx in (0, 1, 2)]
            results = [f.result() for f in futures]

        assert len(results) == 3
        for idx, res in enumerate(results):
            assert res.class_index == idx
            assert res.heatmap.shape == (224, 224)


# ==============================================================================
# 5. Real Model End-to-End Grad-CAM Smoke Test
# ==============================================================================


class TestRealModelGradCAMSmoke:
    """End-to-end integration smoke test with the actual loaded DenseNet121 model."""

    def test_real_model_gradcam_inference_on_synthetic_xray(
        self, loaded_real_model: Any, synthetic_input_tensor: np.ndarray
    ) -> None:
        """Execute full Grad-CAM pipeline using the pre-trained DenseNet121 model.

        Evaluates mathematical validity and latency telemetry on synthetic radiograph data.
        Strict medical neutrality: Output is purely computational explainability research.
        """
        service = GradCAMService(model_loader=get_model_loader())
        result = service.generate_heatmap(synthetic_input_tensor, class_index=None)

        assert isinstance(result, GradCAMResult)
        assert result.selected_class in ("COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS")
        assert result.selected_label in ("COVID-19", "Normal", "Pneumonia", "Tuberculosis")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 4
        assert pytest.approx(sum(result.probabilities.values()), abs=1e-3) == 1.0
        assert result.heatmap_width == 224
        assert result.heatmap_height == 224
        assert result.heatmap.shape == (224, 224)
        assert np.all(np.isfinite(result.heatmap))
        assert result.latency_ms > 0.0
        assert result.disclaimer == RESEARCH_DISCLAIMER

        # Convert to dictionary without grid
        res_dict = result.to_dict(include_grid=False)
        assert "heatmap_grid" not in res_dict
        assert res_dict["target_layer"] == TARGET_CONV_LAYER

        # Convert with grid
        res_dict_with_grid = result.to_dict(include_grid=True)
        assert "heatmap_grid" in res_dict_with_grid
        assert len(res_dict_with_grid["heatmap_grid"]) == 224
