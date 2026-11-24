"""Model artifact discovery and integrity validation utility.

Deterministically locates, inspects, and cryptographically verifies the
DenseNet121 model artifacts and configuration schema without mutating files.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Approved cryptographic integrity baselines
APPROVED_SHA256_BASELINES: Dict[str, str] = {
    "lungcare_config.json": "6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456",
    "lungcare_densenet121.h5": "AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B",
    "lungcare_densenet121.keras": "B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249",
    "lungcare_evaluation.json": "5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392",
}

EXPECTED_CLASS_MAPPING: Dict[str, str] = {
    "0": "COVID19",
    "1": "NORMAL",
    "2": "PNEUMONIA",
    "3": "TURBERCULOSIS",
}


class ArtifactValidationError(Exception):
    """Raised when model artifacts are missing, unreadable, or corrupted."""
    pass


class ConfigurationValidationError(ArtifactValidationError):
    """Raised when model configuration metadata violates contracts."""
    pass


@dataclass(frozen=True)
class ValidatedArtifacts:
    """Immutable value object holding validated artifact paths and configurations."""
    model_dir: Path
    keras_path: Path
    h5_path: Path
    config_path: Path
    evaluation_path: Path
    config_data: Dict[str, Any]
    evaluation_data: Dict[str, Any]
    sha256_hashes: Dict[str, str]


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 cryptographic checksum of a file in streaming chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise ArtifactValidationError(f"Cannot compute hash: file does not exist at '{path}'")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def resolve_model_directory(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Deterministically resolve the models/LungCare_AI_Model directory.
    
    Order of precedence:
    1. Explicit custom_path argument.
    2. LUNGCARE_MODEL_DIR environment variable.
    3. Traversal relative to backend package root.
    """
    if custom_path:
        candidate = Path(custom_path).resolve()
        if candidate.is_dir():
            return candidate
        raise ArtifactValidationError(f"Specified model directory does not exist: '{candidate}'")

    env_dir = os.environ.get("LUNGCARE_MODEL_DIR")
    if env_dir:
        candidate = Path(env_dir).resolve()
        if candidate.is_dir():
            return candidate
        raise ArtifactValidationError(f"LUNGCARE_MODEL_DIR target does not exist: '{candidate}'")

    # Resolve relative to backend/app/utils (3 levels up is project root)
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]  # backend/app/utils -> backend/app -> backend -> LungCare-AI
    default_dir = (project_root / "models" / "LungCare_AI_Model").resolve()

    if default_dir.is_dir():
        return default_dir

    # Fallback: check working directory and parents
    cwd = Path.cwd().resolve()
    for base in [cwd, cwd.parent]:
        candidate = (base / "models" / "LungCare_AI_Model").resolve()
        if candidate.is_dir():
            return candidate

    raise ArtifactValidationError(
        f"Unable to locate model directory. Searched '{default_dir}' from '{current_file}'."
    )


def validate_artifacts(
    model_dir: Optional[Union[str, Path]] = None,
    verify_hashes: bool = True,
) -> ValidatedArtifacts:
    """Locate, validate readability, parse JSONs, and verify integrity of all 4 artifacts."""
    resolved_dir = resolve_model_directory(model_dir)

    expected_files = {
        "keras": resolved_dir / "lungcare_densenet121.keras",
        "h5": resolved_dir / "lungcare_densenet121.h5",
        "config": resolved_dir / "lungcare_config.json",
        "evaluation": resolved_dir / "lungcare_evaluation.json",
    }

    # 1. Existence, readability, and non-zero size checks
    for key, path in expected_files.items():
        if not path.exists():
            raise ArtifactValidationError(f"Required model artifact '{path.name}' is missing at '{path}'")
        if not path.is_file():
            raise ArtifactValidationError(f"Model artifact path '{path}' is not a regular file")
        try:
            size = path.stat().st_size
            if size == 0:
                raise ArtifactValidationError(f"Model artifact '{path.name}' is empty (0 bytes)")
        except OSError as exc:
            raise ArtifactValidationError(f"Cannot read artifact '{path.name}': {exc}")

    # 2. JSON parsing
    try:
        with open(expected_files["config"], "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as exc:
        raise ConfigurationValidationError(f"Failed to parse '{expected_files['config'].name}': {exc}")

    try:
        with open(expected_files["evaluation"], "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)
    except Exception as exc:
        raise ArtifactValidationError(f"Failed to parse '{expected_files['evaluation'].name}': {exc}")

    # 3. SHA-256 integrity verification
    calculated_hashes = {}
    for key, path in expected_files.items():
        calculated_hash = compute_sha256(path)
        calculated_hashes[path.name] = calculated_hash

        if verify_hashes and path.name in APPROVED_SHA256_BASELINES:
            expected_hash = APPROVED_SHA256_BASELINES[path.name]
            if calculated_hash != expected_hash:
                raise ArtifactValidationError(
                    f"Integrity check failed for '{path.name}'. "
                    f"Expected {expected_hash}, computed {calculated_hash}"
                )

    # 4. Configuration structure validation
    validate_model_config(config_data, resolved_dir)

    return ValidatedArtifacts(
        model_dir=resolved_dir,
        keras_path=expected_files["keras"],
        h5_path=expected_files["h5"],
        config_path=expected_files["config"],
        evaluation_path=expected_files["evaluation"],
        config_data=config_data,
        evaluation_data=evaluation_data,
        sha256_hashes=calculated_hashes,
    )


def validate_model_config(config: Dict[str, Any], model_dir: Path) -> Path:
    """Validate lungcare_config.json against expected DenseNet121 platform contracts.
    
    Returns the resolved primary .keras model artifact path.
    """
    required_fields = [
        "model_name",
        "architecture",
        "input_size",
        "input_channels",
        "preprocessing",
        "color_mode",
        "class_mapping",
        "display_names",
        "model_file",
    ]

    for field in required_fields:
        if field not in config:
            raise ConfigurationValidationError(f"Configuration is missing required field: '{field}'")

    # Architecture validation
    arch = str(config.get("architecture", "")).strip()
    if arch.lower() != "densenet121":
        raise ConfigurationValidationError(f"Expected architecture 'DenseNet121', found '{arch}'")

    # Input dimensions
    input_size = config.get("input_size")
    if input_size != [224, 224] and input_size != (224, 224):
        raise ConfigurationValidationError(f"Expected input_size [224, 224], found '{input_size}'")

    # Channels & color mode
    channels = config.get("input_channels")
    if channels != 3:
        raise ConfigurationValidationError(f"Expected input_channels 3, found '{channels}'")

    color_mode = str(config.get("color_mode", "")).strip().upper()
    if color_mode != "RGB":
        raise ConfigurationValidationError(f"Expected color_mode 'RGB', found '{color_mode}'")

    # Preprocessing function
    prep = str(config.get("preprocessing", "")).strip()
    if prep != "tf.keras.applications.densenet.preprocess_input":
        raise ConfigurationValidationError(
            f"Expected preprocessing 'tf.keras.applications.densenet.preprocess_input', found '{prep}'"
        )

    # Class mapping validation: contiguous 0..3 indices
    class_mapping = config.get("class_mapping")
    if not isinstance(class_mapping, dict):
        raise ConfigurationValidationError("Field 'class_mapping' must be a JSON object")

    if len(class_mapping) != 4:
        raise ConfigurationValidationError(f"Expected exactly 4 classes in class_mapping, found {len(class_mapping)}")

    indices = sorted([str(k) for k in class_mapping.keys()])
    if indices != ["0", "1", "2", "3"]:
        raise ConfigurationValidationError(f"Class mapping keys must be contiguous '0', '1', '2', '3'. Found: {indices}")

    # Preserve internal class key: TURBERCULOSIS
    for idx, expected_key in EXPECTED_CLASS_MAPPING.items():
        actual_key = class_mapping.get(idx)
        if actual_key != expected_key:
            raise ConfigurationValidationError(
                f"Class index '{idx}' mapping mismatch: expected '{expected_key}', found '{actual_key}'"
            )

    # Display names validation
    display_names = config.get("display_names")
    if not isinstance(display_names, dict):
        raise ConfigurationValidationError("Field 'display_names' must be a JSON object")

    for internal_key in EXPECTED_CLASS_MAPPING.values():
        if internal_key not in display_names:
            raise ConfigurationValidationError(f"Missing display name for internal class '{internal_key}'")

    # Primary model resolution: .keras artifact
    # The config references 'lungcare_densenet121.h5' or 'lungcare_densenet121.keras'
    # We resolve the primary modern Keras 3 artifact 'lungcare_densenet121.keras'
    primary_keras_path = (model_dir / "lungcare_densenet121.keras").resolve()
    if not primary_keras_path.is_file():
        raise ConfigurationValidationError(
            f"Primary Keras 3 model artifact 'lungcare_densenet121.keras' does not exist at '{primary_keras_path}'"
        )

    return primary_keras_path

