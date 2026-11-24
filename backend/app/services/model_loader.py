"""DenseNet121 Model Loading Lifecycle Service.

Provides a thread-safe, deterministic, and observable lifecycle manager for
loading, validating, and caching the pre-trained DenseNet121 Keras model.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from app.utils.artifact_validator import (
    ArtifactValidationError,
    ConfigurationValidationError,
    EXPECTED_CLASS_MAPPING,
    ValidatedArtifacts,
    validate_artifacts,
)

logger = logging.getLogger("lungcare.model_loader")


class ModelContractError(Exception):
    """Raised when loaded model input/output architecture violates expected contracts."""
    pass


class ModelNotLoadedError(Exception):
    """Raised when model inference is attempted before successful initialization."""
    pass


@dataclass(frozen=True)
class ModelContractMetadata:
    """Detailed architectural and telemetry metadata for the validated model."""
    model_name: str
    input_shape: Tuple[Optional[int], ...]
    output_shape: Tuple[Optional[int], ...]
    layer_count: int
    total_params: int
    trainable_params: int
    non_trainable_params: int
    input_dtype: str
    output_dtype: str
    final_activation: str
    load_duration_ms: float
    smoke_test_latency_ms: float
    smoke_test_probabilities: Dict[str, float]


class ModelLoader:
    """Thread-safe lifecycle service managing the DenseNet121 Keras model instance."""

    def __init__(self, coordinator: Optional[Any] = None) -> None:
        self._lock = threading.Lock()
        self._model: Optional[Any] = None
        self._metadata: Optional[ModelContractMetadata] = None
        self._artifacts: Optional[ValidatedArtifacts] = None
        self._is_loaded: bool = False
        self._load_error: Optional[Exception] = None
        if coordinator is not None:
            self._coordinator = coordinator
        else:
            from app.services.ml_coordinator import get_ml_coordinator
            self._coordinator = get_ml_coordinator()

    @property
    def coordinator(self) -> Any:
        """Return the associated MLExecutionCoordinator instance."""
        return self._coordinator

    @property
    def is_loaded(self) -> bool:
        """Return True if model has been successfully validated and loaded into memory."""
        return self._is_loaded

    @property
    def metadata(self) -> Optional[ModelContractMetadata]:
        """Return architectural and telemetry metadata if model is loaded."""
        return self._metadata

    @property
    def load_error(self) -> Optional[Exception]:
        """Return the exception encountered during the last load attempt, if any."""
        return self._load_error

    def load_model(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> Any:
        """Load and validate the DenseNet121 .keras model instance.
        
        Thread-safe: uses double-checked locking to avoid repeated disk reads.
        Compiles the model with compile=False for optimal inference performance.
        Executes a controlled synthetic inference smoke test before exposing the model.
        """
        # Fast non-blocking check
        if self._is_loaded and not force_reload and self._model is not None:
            if not self._coordinator.is_ready and not self._coordinator.is_draining:
                self._coordinator.complete_loading()
            return self._model

        with self._lock:
            # Re-check under lock
            if self._is_loaded and not force_reload and self._model is not None:
                if not self._coordinator.is_ready and not self._coordinator.is_draining:
                    self._coordinator.complete_loading()
                return self._model

            logger.info("Initializing DenseNet121 model loading lifecycle...")
            t_start = time.perf_counter()

            # Signal coordinator that loading has started
            self._coordinator.start_loading()

            try:
                # 1. Validate artifact discovery, readability, and configuration schema
                artifacts = validate_artifacts(model_dir=model_dir, verify_hashes=True)
                self._artifacts = artifacts

                # 2. Import Keras (scoped import avoids overhead during basic API startup)
                import keras
                from tensorflow.keras.applications.densenet import preprocess_input

                # 3. Load primary .keras artifact (explicitly compile=False for inference)
                primary_path = artifacts.keras_path
                logger.info("Loading primary Keras model from: %s", primary_path)
                
                t_load_0 = time.perf_counter()
                model = keras.models.load_model(primary_path, compile=False)
                t_load_1 = time.perf_counter()
                load_duration_ms = (t_load_1 - t_load_0) * 1000.0

                # 4. Architecture and Contract Validation
                contract_meta = self._validate_model_contract(
                    model=model,
                    artifacts=artifacts,
                    load_duration_ms=load_duration_ms,
                    preprocess_fn=preprocess_input,
                )

                # 5. Commit state
                self._model = model
                self._metadata = contract_meta
                self._is_loaded = True
                self._load_error = None

                # Signal coordinator that model is READY
                self._coordinator.complete_loading()

                logger.info(
                    "DenseNet121 model loaded successfully in %.2f ms [Total params: %s, Classes: %d]",
                    load_duration_ms,
                    f"{contract_meta.total_params:,}",
                    contract_meta.output_shape[-1],
                )
                return self._model

            except Exception as exc:
                self._is_loaded = False
                self._model = None
                self._metadata = None
                self._load_error = exc
                self._coordinator.record_loading_error(exc)
                logger.exception("Failed to load DenseNet121 model: %s", exc)
                raise

    def get_model(self) -> Any:
        """Retrieve the validated model instance, initializing it if necessary."""
        if not self._is_loaded or self._model is None:
            return self.load_model()
        return self._model

    def unload(self, drain_timeout: Optional[float] = None) -> None:
        """Reset internal state and release model reference (useful for testing and shutdown).
        
        Guarantees that model memory is not deallocated while active operations are running.
        """
        with self._lock:
            if self._coordinator.active_operations > 0:
                logger.warning(
                    "ModelLoader.unload() invoked with %d active operations. Waiting up to %s...",
                    self._coordinator.active_operations,
                    drain_timeout,
                )
                drained = self._coordinator.wait_for_active_operations(timeout=drain_timeout)
                if not drained:
                    logger.error(
                        "Cannot safely unload model: %d active operations still running. Aborting deallocation.",
                        self._coordinator.active_operations,
                    )
                    return

            self._coordinator.complete_unloading()
            self._model = None
            self._metadata = None
            self._artifacts = None
            self._is_loaded = False
            self._load_error = None
            logger.info("DenseNet121 model unloaded from memory.")

    def _validate_model_contract(
        self,
        model: Any,
        artifacts: ValidatedArtifacts,
        load_duration_ms: float,
        preprocess_fn: Any,
    ) -> ModelContractMetadata:
        """Validate input shape, output shape, layer activations, and execute synthetic smoke test."""
        # Input shape check: expected (None, 224, 224, 3)
        input_shape = getattr(model, "input_shape", None)
        if input_shape != (None, 224, 224, 3):
            raise ModelContractError(
                f"Model input shape contract violated: expected (None, 224, 224, 3), found {input_shape}"
            )

        # Output shape check: expected (None, 4)
        output_shape = getattr(model, "output_shape", None)
        if output_shape != (None, 4):
            raise ModelContractError(
                f"Model output shape contract violated: expected (None, 4), found {output_shape}"
            )

        # Output class count consistency with config mapping
        class_mapping = artifacts.config_data.get("class_mapping", {})
        if output_shape[-1] != len(class_mapping):
            raise ModelContractError(
                f"Model output dimension ({output_shape[-1]}) does not match config class count ({len(class_mapping)})"
            )

        # Final layer activation inspection
        final_layer = model.layers[-1]
        final_activation = getattr(final_layer, "activation", None)
        activation_name = final_activation.__name__ if final_activation else "unknown"
        if activation_name.lower() != "softmax":
            raise ModelContractError(
                f"Expected final layer activation 'softmax', but found '{activation_name}' on layer '{final_layer.name}'"
            )

        # Parameter counts
        total_params = int(model.count_params())
        trainable_params = int(sum(p.numpy().size for p in model.trainable_variables))
        non_trainable_params = int(sum(p.numpy().size for p in model.non_trainable_variables))

        # Dtypes
        input_dtype = str(getattr(model, "input_dtype", None) or model.inputs[0].dtype)
        output_dtype = str(model.outputs[0].dtype)

        # ----------------------------------------------------------------------
        # Synthetic Inference Smoke Test
        # ----------------------------------------------------------------------
        # Create deterministic synthetic RGB image array in [0, 255] range
        np.random.seed(1337)
        synthetic_raw = np.random.uniform(0.0, 255.0, (1, 224, 224, 3)).astype(np.float32)
        
        # Apply standard DenseNet preprocessing
        synthetic_preprocessed = preprocess_fn(synthetic_raw.copy())

        # Forward pass (training=False ensures deterministic BatchNorm and Dropout behaviour)
        t_infer_0 = time.perf_counter()
        raw_output = model(synthetic_preprocessed, training=False).numpy()
        t_infer_1 = time.perf_counter()
        smoke_latency_ms = (t_infer_1 - t_infer_0) * 1000.0

        if raw_output.shape != (1, 4):
            raise ModelContractError(
                f"Synthetic smoke test output shape mismatch: expected (1, 4), got {raw_output.shape}"
            )

        if not np.all(np.isfinite(raw_output)):
            raise ModelContractError("Synthetic smoke test produced non-finite (NaN or Inf) probabilities.")

        if np.any(raw_output < 0.0) or np.any(raw_output > 1.0):
            raise ModelContractError("Synthetic smoke test output contains probabilities outside [0.0, 1.0].")

        prob_sum = float(np.sum(raw_output[0]))
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            raise ModelContractError(
                f"Synthetic smoke test probabilities do not sum to 1.0 (sum={prob_sum:.6f})"
            )

        # Map probabilities using configured class keys
        probs_dict = {
            EXPECTED_CLASS_MAPPING[str(i)]: float(raw_output[0, i])
            for i in range(4)
        }

        return ModelContractMetadata(
            model_name=str(getattr(model, "name", "DenseNet121")),
            input_shape=input_shape,
            output_shape=output_shape,
            layer_count=len(model.layers),
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            final_activation=activation_name,
            load_duration_ms=load_duration_ms,
            smoke_test_latency_ms=smoke_latency_ms,
            smoke_test_probabilities=probs_dict,
        )


# Singleton factory for dependency injection
_global_loader: Optional[ModelLoader] = None
_global_lock = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Return the application-wide singleton ModelLoader instance."""
    global _global_loader
    if _global_loader is None:
        with _global_lock:
            if _global_loader is None:
                _global_loader = ModelLoader()
    return _global_loader
