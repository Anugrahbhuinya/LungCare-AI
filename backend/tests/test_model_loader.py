"""Focused test suite for DenseNet121 model artifact validation and loading lifecycle.

Covers:
1. Valid artifact path resolution.
2. Missing model artifact.
3. Missing configuration.
4. Invalid JSON configuration.
5. Invalid class mapping.
6. Incorrect configured input dimensions.
7. Successful model loading.
8. Input/output contract validation.
9. Synthetic inference output validation.
10. Repeated access returns the same model instance.
11. Concurrent initialization does not create duplicate model instances.
12. Model loading errors are reported clearly.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import pytest

from app.utils.artifact_validator import (
    APPROVED_SHA256_BASELINES,
    ArtifactValidationError,
    ConfigurationValidationError,
    EXPECTED_CLASS_MAPPING,
    resolve_model_directory,
    validate_artifacts,
    validate_model_config,
)

# Detect if ML runtime (TensorFlow/Keras) is available in current interpreter
try:
    import tensorflow as tf
    import keras
    HAS_ML_RUNTIME = True
except ImportError:
    HAS_ML_RUNTIME = False

skip_if_no_ml = pytest.mark.skipif(
    not HAS_ML_RUNTIME,
    reason="TensorFlow/Keras runtime required (execute using backend/.venv-ml)",
)


# ------------------------------------------------------------------------------
# 1. Artifact Path & Integrity Resolution
# ------------------------------------------------------------------------------

def test_valid_artifact_path_resolution() -> None:
    """Requirement 1: Valid artifact path resolution and baseline validation."""
    artifacts = validate_artifacts(verify_hashes=True)
    assert artifacts.model_dir.is_dir()
    assert artifacts.keras_path.is_file()
    assert artifacts.h5_path.is_file()
    assert artifacts.config_path.is_file()
    assert artifacts.evaluation_path.is_file()

    # Verify SHA-256 integrity matches approved baselines
    for filename, expected_hash in APPROVED_SHA256_BASELINES.items():
        assert artifacts.sha256_hashes[filename] == expected_hash


def test_missing_model_artifact_raises_error(tmp_path: Path) -> None:
    """Requirement 2: Missing model artifact raises ArtifactValidationError."""
    # Create incomplete model directory missing .keras
    (tmp_path / "lungcare_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lungcare_evaluation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lungcare_densenet121.h5").write_text("fake_h5", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_artifacts(model_dir=tmp_path, verify_hashes=False)
    assert "lungcare_densenet121.keras" in str(exc_info.value)


def test_missing_configuration_raises_error(tmp_path: Path) -> None:
    """Requirement 3: Missing configuration file raises ArtifactValidationError."""
    (tmp_path / "lungcare_densenet121.keras").write_text("fake_keras", encoding="utf-8")
    (tmp_path / "lungcare_densenet121.h5").write_text("fake_h5", encoding="utf-8")
    (tmp_path / "lungcare_evaluation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_artifacts(model_dir=tmp_path, verify_hashes=False)
    assert "lungcare_config.json" in str(exc_info.value)


def test_invalid_json_configuration_raises_error(tmp_path: Path) -> None:
    """Requirement 4: Malformed/corrupted JSON configuration raises ConfigurationValidationError."""
    (tmp_path / "lungcare_densenet121.keras").write_text("fake_keras", encoding="utf-8")
    (tmp_path / "lungcare_densenet121.h5").write_text("fake_h5", encoding="utf-8")
    (tmp_path / "lungcare_evaluation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lungcare_config.json").write_text("{ unclosed invalid json ...", encoding="utf-8")

    with pytest.raises(ConfigurationValidationError) as exc_info:
        validate_artifacts(model_dir=tmp_path, verify_hashes=False)
    assert "Failed to parse" in str(exc_info.value)


def test_invalid_class_mapping_raises_error(tmp_path: Path) -> None:
    """Requirement 5: Invalid or non-contiguous class mapping raises ConfigurationValidationError."""
    valid_dir = resolve_model_directory()
    with open(valid_dir / "lungcare_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Corrupt class mapping to only 3 classes
    config["class_mapping"] = {"0": "COVID19", "1": "NORMAL", "2": "PNEUMONIA"}
    with pytest.raises(ConfigurationValidationError) as exc_info:
        validate_model_config(config, valid_dir)
    assert "Expected exactly 4 classes" in str(exc_info.value)

    # Corrupt class keys to non-contiguous
    config["class_mapping"] = {"0": "COVID19", "1": "NORMAL", "2": "PNEUMONIA", "4": "TURBERCULOSIS"}
    with pytest.raises(ConfigurationValidationError) as exc_info:
        validate_model_config(config, valid_dir)
    assert "contiguous" in str(exc_info.value).lower()


def test_incorrect_configured_input_dimensions_raises_error() -> None:
    """Requirement 6: Incorrect configured input dimensions raise ConfigurationValidationError."""
    valid_dir = resolve_model_directory()
    with open(valid_dir / "lungcare_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    config["input_size"] = [256, 256]
    with pytest.raises(ConfigurationValidationError) as exc_info:
        validate_model_config(config, valid_dir)
    assert "Expected input_size [224, 224]" in str(exc_info.value)


# ------------------------------------------------------------------------------
# 2. Model Loading Lifecycle & Contract Tests (Require ML Runtime)
# ------------------------------------------------------------------------------

@skip_if_no_ml
def test_successful_model_loading() -> None:
    """Requirement 7: Primary .keras artifact loads successfully."""
    from app.services.model_loader import get_model_loader
    loader = get_model_loader()
    model = loader.load_model()

    assert loader.is_loaded is True
    assert model is not None
    assert loader.load_error is None


@skip_if_no_ml
def test_input_output_contract_validation() -> None:
    """Requirement 8: Input/output shapes and softmax activation contracts."""
    from app.services.model_loader import get_model_loader
    loader = get_model_loader()
    loader.load_model()
    meta = loader.metadata

    assert meta is not None
    assert meta.input_shape == (None, 224, 224, 3)
    assert meta.output_shape == (None, 4)
    assert meta.final_activation.lower() == "softmax"
    assert meta.total_params == 7045700
    assert meta.trainable_params == 6148
    assert meta.non_trainable_params == 7039554
    assert "float" in meta.input_dtype.lower()
    assert "float" in meta.output_dtype.lower()


@skip_if_no_ml
def test_synthetic_inference_output_validation() -> None:
    """Requirement 9: Synthetic inference smoke test produces valid probabilities."""
    from app.services.model_loader import get_model_loader
    loader = get_model_loader()
    loader.load_model()
    meta = loader.metadata

    assert meta is not None
    probs = meta.smoke_test_probabilities
    assert len(probs) == 4
    for key in EXPECTED_CLASS_MAPPING.values():
        assert key in probs
        assert 0.0 <= probs[key] <= 1.0

    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-4


@skip_if_no_ml
def test_repeated_access_returns_same_model_instance() -> None:
    """Requirement 10: Repeated access returns the cached singleton model instance."""
    from app.services.model_loader import get_model_loader
    loader = get_model_loader()
    model_a = loader.get_model()
    model_b = loader.get_model()
    model_c = loader.load_model()

    assert model_a is model_b
    assert model_b is model_c


@skip_if_no_ml
def test_concurrent_initialization_does_not_create_duplicates() -> None:
    """Requirement 11: Concurrent access safely resolves to the identical model instance."""
    from app.services.model_loader import ModelLoader
    # Use dedicated loader instance to test concurrency
    loader = ModelLoader()

    def worker_load():
        return loader.load_model()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_load) for _ in range(8)]
        results = [f.result() for f in futures]

    first_instance = results[0]
    for inst in results[1:]:
        assert inst is first_instance
    assert loader.is_loaded is True


def test_model_loading_errors_reported_clearly(tmp_path: Path) -> None:
    """Requirement 12: Errors during artifact discovery/validation are reported clearly."""
    from app.services.model_loader import ModelLoader
    loader = ModelLoader()

    # Attempt loading from empty directory
    with pytest.raises(ArtifactValidationError) as exc_info:
        loader.load_model(model_dir=tmp_path)

    assert loader.is_loaded is False
    assert loader.load_error is not None
    assert "missing" in str(exc_info.value).lower() or "not exist" in str(exc_info.value).lower()

