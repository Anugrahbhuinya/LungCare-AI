"""Grad-CAM Visual Explainability Service for LungCare AI.

Implements class-specific Gradient-weighted Class Activation Mapping (Grad-CAM)
for the pre-trained DenseNet121 chest radiograph classification model.
Reuses the validated model singleton and execution graph with strict numerical
safety, input validation, and thread synchronization.

Differentiation Target:
    Differentiates against the post-softmax class probability S_c in [0.0, 1.0]
    produced by the model's final classification layer Dense(4, activation='softmax').
    The model architecture fuses the Dense layer with softmax activation, so the
    computational graph outputs post-softmax probabilities directly. The backpropagated
    gradient is d(S_c) / d(A^k), where A^k is the feature map of target convolutional
    layer 'conv5_block16_2_conv'.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.services.model_loader import ModelLoader, get_model_loader

logger = logging.getLogger("lungcare.ml.gradcam")

# Approved Grad-CAM target convolutional layer in the DenseNet121 backbone
TARGET_CONV_LAYER: str = "conv5_block16_2_conv"

# Model output class order and display mapping (strict preservation of TURBERCULOSIS)
CLASS_INDICES: Dict[int, str] = {
    0: "COVID19",
    1: "NORMAL",
    2: "PNEUMONIA",
    3: "TURBERCULOSIS",
}

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "COVID19": "COVID-19",
    "NORMAL": "Normal",
    "PNEUMONIA": "Pneumonia",
    "TURBERCULOSIS": "Tuberculosis",
}

RESEARCH_DISCLAIMER: str = (
    "LungCare AI is an experimental AI prediction model developed solely for research and "
    "educational purposes. It does not provide medical diagnosis, clinical advice, or treatment "
    "recommendations. Diagnostic evaluations must be performed by certified radiologists."
)


@dataclass(frozen=True)
class GradCAMResult:
    """Immutable structured result returned by GradCAMService."""

    selected_class: str
    selected_label: str
    class_index: int
    target_layer: str
    heatmap: np.ndarray  # 2D float32 array of shape (224, 224) in [0.0, 1.0]
    heatmap_width: int
    heatmap_height: int
    status: str  # "success" or "uninformative"
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self, include_grid: bool = False) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        payload: Dict[str, Any] = {
            "selected_class": self.selected_class,
            "selected_label": self.selected_label,
            "class_index": self.class_index,
            "target_layer": self.target_layer,
            "heatmap_width": self.heatmap_width,
            "heatmap_height": self.heatmap_height,
            "status": self.status,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
            "disclaimer": self.disclaimer,
        }
        if include_grid:
            payload["heatmap_grid"] = self.heatmap.tolist()
        return payload


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class GradCAMError(Exception):
    """Base exception for all Grad-CAM explainability failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "GRADCAM_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class InvalidGradCAMInputError(GradCAMError):
    """Raised when the input tensor or class index violates dimensional or numerical contracts."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, error_code="INVALID_GRADCAM_INPUT", details=details)


class GradCAMLayerNotFoundError(GradCAMError):
    """Raised when the specified target convolutional layer cannot be found in the model."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, error_code="GRADCAM_LAYER_NOT_FOUND", details=details)


class GradCAMGradientError(GradCAMError):
    """Raised when gradient computation produces None or disconnected computational paths."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, error_code="GRADCAM_GRADIENT_ERROR", details=details)


class GradCAMGenerationError(GradCAMError):
    """Raised when an unhandled failure occurs during heatmap generation or normalization."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, error_code="GRADCAM_GENERATION_ERROR", details=details)


# ==============================================================================
# Service Implementation
# ==============================================================================


class GradCAMService:
    """Production-grade Grad-CAM Explainability Service for DenseNet121."""

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        target_layer_name: str = TARGET_CONV_LAYER,
        coordinator: Optional[Any] = None,
    ) -> None:
        """Initialize Grad-CAM service with dependency injection support.

        Args:
            model_loader: Optional ModelLoader singleton instance.
            target_layer_name: Convolutional layer name providing spatial feature maps.
            coordinator: Optional MLExecutionCoordinator singleton instance.
        """
        self._model_loader = model_loader or get_model_loader()
        self._target_layer_name = target_layer_name
        if coordinator is not None:
            self._coordinator = coordinator
        else:
            from app.services.ml_coordinator import get_ml_coordinator
            self._coordinator = get_ml_coordinator()
        self._grad_model: Optional[Any] = None

    @property
    def coordinator(self) -> Any:
        """Return the associated MLExecutionCoordinator instance."""
        return self._coordinator

    @property
    def target_layer_name(self) -> str:
        """Return the configured target convolutional layer name."""
        return self._target_layer_name

    def _build_or_get_grad_model(self) -> Any:
        """Construct or retrieve the reusable Functional gradient model.

        Reuses the existing loaded model's weights and layers without duplicating
        parameters or recompiling the model graph. Handles both direct layers and
        nested DenseNet121 backbone architectures.
        """
        if self._grad_model is not None:
            return self._grad_model

        model = self._model_loader.get_model()
        if model is None:
            raise GradCAMGenerationError("Model instance is unavailable in ModelLoader.")

        from app.services.ml_coordinator import ModelLifecycleState
        if getattr(self._model_loader, "is_loaded", False) and self._coordinator.state != ModelLifecycleState.DRAINING:
            if not self._coordinator.is_ready:
                self._coordinator.complete_loading()

        import keras

        # Case A: Check if the target layer exists directly in the primary model
        target_layer = None
        try:
            target_layer = model.get_layer(self._target_layer_name)
        except Exception:
            target_layer = None

        if target_layer is not None:
            try:
                self._grad_model = keras.Model(
                    inputs=model.inputs,
                    outputs=[target_layer.output, model.output],
                    name="GradCAM_Direct_Model",
                )
                logger.info("Constructed direct Grad-CAM gradient model for layer: %s", self._target_layer_name)
                return self._grad_model
            except Exception as exc:
                logger.debug("Direct model construction failed (%s), inspecting sub-models...", exc)

        # Case B: Inspect sub-models (nested DenseNet121 backbone)
        backbone_layer = None
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    target_layer = layer.get_layer(self._target_layer_name)
                    backbone_layer = layer
                    break
                except Exception:
                    continue

        if target_layer is None or backbone_layer is None:
            raise GradCAMLayerNotFoundError(
                f"Target layer '{self._target_layer_name}' not found in model or nested backbone layers.",
                details={"target_layer": self._target_layer_name, "available_layers": [l.name for l in model.layers]},
            )

        try:
            # Construct submodel for backbone: inputs -> [target_layer.output, backbone.output]
            backbone_submodel = keras.Model(
                inputs=backbone_layer.inputs,
                outputs=[target_layer.output, backbone_layer.output],
                name="GradCAM_Backbone_Submodel",
            )

            # Discover subsequent head layers from outer model
            backbone_idx = model.layers.index(backbone_layer)
            subsequent_layers = model.layers[backbone_idx + 1 :]

            # Build unified functional gradient model connecting inputs -> [target_conv, final_predictions]
            grad_inputs = keras.Input(shape=(224, 224, 3), name="gradcam_input")
            conv_out, backbone_out = backbone_submodel(grad_inputs)

            x = backbone_out
            for layer in subsequent_layers:
                # Call subsequent layers preserving existing weights
                x = layer(x)

            self._grad_model = keras.Model(
                inputs=grad_inputs,
                outputs=[conv_out, x],
                name="GradCAM_Nested_Model",
            )
            logger.info(
                "Constructed nested Grad-CAM gradient model for backbone '%s' layer '%s' with %d head layers.",
                backbone_layer.name,
                self._target_layer_name,
                len(subsequent_layers),
            )
            return self._grad_model

        except Exception as exc:
            logger.exception("Failed to build Grad-CAM gradient model graph: %s", exc)
            raise GradCAMGenerationError(
                f"Failed to construct Grad-CAM gradient model: {exc}",
                details={"error": str(exc), "target_layer": self._target_layer_name},
            ) from exc

    def generate_heatmap(
        self,
        image_tensor: Union[np.ndarray, Any],
        class_index: Optional[int] = None,
    ) -> GradCAMResult:
        """Generate class-specific Grad-CAM heatmap for a preprocessed radiograph tensor.

        Differentiation Target:
            Differentiates against the post-softmax class probability S_c in [0.0, 1.0]
            produced by the model's final classification layer Dense(4, activation='softmax').
            The backpropagated gradient is d(S_c) / d(A^k), where A^k is the feature map of
            target convolutional layer 'conv5_block16_2_conv'.

        Args:
            image_tensor: Float32 tensor of shape (1, 224, 224, 3) with DenseNet preprocessing applied.
            class_index: Optional zero-indexed class integer (0 to 3). If None, defaults to predicted class.

        Returns:
            GradCAMResult containing the 224x224 normalized float32 heatmap and classification metadata.

        Raises:
            InvalidGradCAMInputError: If tensor shape, dtype, values, or class_index are invalid.
            GradCAMLayerNotFoundError: If target layer cannot be resolved.
            GradCAMGradientError: If gradient backpropagation produces None.
            GradCAMGenerationError: If execution fails unexpectedly.
        """
        # Step 1: Strict Defensive Input Validation
        self._validate_input_tensor(image_tensor)
        self._validate_class_index(class_index)

        import tensorflow as tf

        # Ensure input is a float32 tf.Tensor
        if not isinstance(image_tensor, tf.Tensor):
            input_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        else:
            input_tensor = tf.cast(image_tensor, dtype=tf.float32)

        # Ensure model is available and sync coordinator state if needed
        model = self._model_loader.get_model()
        if model is None:
            raise GradCAMGenerationError("Model instance is unavailable in ModelLoader.")
        from app.services.ml_coordinator import MLNotReadyError, ModelLifecycleState
        from app.services.inference_service import ModelUnavailableError

        if getattr(self._model_loader, "is_loaded", False) and self._coordinator.state != ModelLifecycleState.DRAINING:
            if not self._coordinator.is_ready:
                self._coordinator.complete_loading()

        # Step 2: Synchronized Gradient Computation under shared coordinator execution lock
        try:
            with self._coordinator.acquire_execution():
                grad_model = self._build_or_get_grad_model()

                t_start = time.perf_counter()
                try:
                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(input_tensor, training=False)
                        tape.watch(conv_outputs)

                        # Determine target class index
                        if class_index is None:
                            selected_idx = int(tf.argmax(predictions[0]))
                        else:
                            selected_idx = int(class_index)

                        # Differentiation target: post-softmax probability S_c = predictions[:, selected_idx]
                        class_score = predictions[:, selected_idx]

                    # Backpropagate gradients of the selected class score w.r.t. target feature map
                    grads = tape.gradient(class_score, conv_outputs)

                except Exception as exc:
                    logger.exception("Grad-CAM GradientTape computation failed: %s", exc)
                    raise GradCAMGenerationError(
                        f"Forward/backward pass during Grad-CAM failed: {exc}",
                        details={"error": str(exc)},
                    ) from exc
        except MLNotReadyError as exc:
            logger.warning("Grad-CAM execution rejected by coordinator: %s", exc)
            raise ModelUnavailableError(
                str(exc),
                details={"lifecycle_state": exc.state.value},
            ) from exc

        # Step 3: Defensive Validation of Convolutional Feature Maps
        if not tf.reduce_all(tf.math.is_finite(conv_outputs)):
            raise GradCAMGenerationError(
                f"Target convolutional layer '{self._target_layer_name}' produced non-finite feature maps (NaN/Inf).",
                details={"target_layer": self._target_layer_name},
            )

        # Step 4: Defensive Validation of Gradients (Issue 2)
        if grads is None:
            raise GradCAMGradientError(
                f"Gradients w.r.t. target layer '{self._target_layer_name}' evaluated to None. "
                "Ensure target layer is topologically connected to the loss score.",
                details={"target_layer": self._target_layer_name},
            )

        if tf.reduce_any(tf.math.is_nan(grads)):
            raise GradCAMGradientError(
                f"Gradients w.r.t. target layer '{self._target_layer_name}' contain NaN values.",
                details={"target_layer": self._target_layer_name},
            )

        if tf.reduce_any(tf.math.is_inf(grads)):
            raise GradCAMGradientError(
                f"Gradients w.r.t. target layer '{self._target_layer_name}' contain Inf values.",
                details={"target_layer": self._target_layer_name},
            )

        # Detect valid zero gradients (where class score is completely insensitive to target layer)
        is_zero_gradients = bool(tf.reduce_all(tf.equal(grads, 0.0)).numpy())

        # Step 5: Channel-wise Importance Weights via Global Average Pooling
        # Shape: (32,) for DenseNet121 block16_2_conv
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Step 6: Weighted Sum of Activation Maps
        conv_outputs_val = conv_outputs[0]  # Shape: (7, 7, 32)
        cam = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)  # Shape: (7, 7)

        # Step 7: Rectified Linear Unit (ReLU) to isolate positive contributions
        cam = tf.maximum(cam, 0.0)

        if not tf.reduce_all(tf.math.is_finite(cam)):
            raise GradCAMGenerationError(
                "Weighted activation map contains non-finite values after pooling and ReLU."
            )

        # Step 8: Safe Spatial Upsampling to 224 x 224
        cam_resized = tf.image.resize(
            cam[None, ..., None], (224, 224), method=tf.image.ResizeMethod.BILINEAR
        )[0, ..., 0]

        if not tf.reduce_all(tf.math.is_finite(cam_resized)):
            raise GradCAMGenerationError("Resized activation map contains non-finite values.")

        # Step 9: Safe Numerical Normalization separating valid uninformative from active heatmaps (Issue 2)
        cam_max = tf.reduce_max(cam_resized)
        cam_min = tf.reduce_min(cam_resized)

        max_val_float = float(cam_max.numpy())
        min_val_float = float(cam_min.numpy())

        if not np.isfinite(max_val_float) or not np.isfinite(min_val_float):
            raise GradCAMGenerationError(
                "Extrema of resized activation map are non-finite.",
                details={"max": max_val_float, "min": min_val_float},
            )

        is_constant_activation = (max_val_float - min_val_float) <= 1e-10 or max_val_float <= 1e-10

        if is_zero_gradients or is_constant_activation:
            # Valid uninformative explanation: flat or zero response
            status = "uninformative"
            normalized_heatmap = np.zeros((224, 224), dtype=np.float32)
            logger.info(
                "Grad-CAM produced valid %s (zero_grads=%s, constant_act=%s). Marked uninformative.",
                "zero gradient" if is_zero_gradients else "constant activation map",
                is_zero_gradients,
                is_constant_activation,
            )
        else:
            status = "success"
            normalized_tensor = (cam_resized - cam_min) / (cam_max - cam_min)
            normalized_heatmap = np.clip(normalized_tensor.numpy(), 0.0, 1.0).astype(np.float32)

        # Verify output finiteness
        if not np.all(np.isfinite(normalized_heatmap)):
            raise GradCAMGenerationError("Normalized Grad-CAM heatmap contains non-finite values.")

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Step 9: Classification Metadata Mapping
        pred_array = predictions[0].numpy()
        confidence = float(pred_array[selected_idx])
        probabilities = {
            CLASS_INDICES[i]: float(pred_array[i])
            for i in range(4)
        }
        selected_class = CLASS_INDICES[selected_idx]
        selected_label = CLASS_DISPLAY_NAMES[selected_class]

        logger.info(
            "Grad-CAM explanation generated: class=%s (%s), status=%s, latency=%.2fms",
            selected_class,
            selected_label,
            status,
            latency_ms,
        )

        return GradCAMResult(
            selected_class=selected_class,
            selected_label=selected_label,
            class_index=selected_idx,
            target_layer=self._target_layer_name,
            heatmap=normalized_heatmap,
            heatmap_width=224,
            heatmap_height=224,
            status=status,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    def _validate_input_tensor(self, tensor: Any) -> None:
        """Validate input tensor shape, dtype, and numerical bounds."""
        if tensor is None:
            raise InvalidGradCAMInputError("Input image tensor cannot be None.")

        # Convert to numpy array for attribute inspections if needed
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            raise InvalidGradCAMInputError(
                f"Unsupported tensor type '{type(tensor).__name__}'. Expected NumPy array or TensorFlow tensor."
            )

        if arr.ndim != 4 or arr.shape != (1, 224, 224, 3):
            raise InvalidGradCAMInputError(
                f"Invalid input tensor shape {arr.shape}. Expected exact shape (1, 224, 224, 3).",
                details={"shape": list(arr.shape), "expected_shape": [1, 224, 224, 3]},
            )

        if not np.issubdtype(arr.dtype, np.floating):
            raise InvalidGradCAMInputError(
                f"Invalid tensor dtype '{arr.dtype}'. Expected floating-point dtype (e.g. float32).",
                details={"dtype": str(arr.dtype)},
            )

        if not np.all(np.isfinite(arr)):
            raise InvalidGradCAMInputError(
                "Input tensor contains non-finite values (NaN or Inf).",
                details={"nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum())},
            )

    def _validate_class_index(self, class_index: Optional[int]) -> None:
        """Validate that class_index is within allowable domain [0, 3]."""
        if class_index is None:
            return

        if not isinstance(class_index, (int, np.integer)):
            raise InvalidGradCAMInputError(
                f"class_index must be an integer, got {type(class_index).__name__}.",
                details={"class_index": class_index},
            )

        if class_index < 0 or class_index > 3:
            raise InvalidGradCAMInputError(
                f"class_index {class_index} is out of bounds. Expected integer in range [0, 3].",
                details={"class_index": class_index, "valid_range": [0, 3]},
            )


# ==============================================================================
# Thread-Safe Singleton Accessor
# ==============================================================================

_gradcam_service_instance: Optional[GradCAMService] = None
_gradcam_service_lock = threading.Lock()


def get_gradcam_service(
    model_loader: Optional[ModelLoader] = None,
    target_layer_name: str = TARGET_CONV_LAYER,
) -> GradCAMService:
    """Return the application-wide singleton instance of GradCAMService."""
    global _gradcam_service_instance
    if _gradcam_service_instance is None or model_loader is not None:
        with _gradcam_service_lock:
            if _gradcam_service_instance is None or model_loader is not None:
                _gradcam_service_instance = GradCAMService(
                    model_loader=model_loader,
                    target_layer_name=target_layer_name,
                )
    return _gradcam_service_instance
