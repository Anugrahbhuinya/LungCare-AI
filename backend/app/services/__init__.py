"""Service layer for LungCare AI.

Encapsulates core business logic, domain workflows, and integrations:
- Model loading lifecycle & contract validation (Phase 2.2.2)
- Chest X-ray image preprocessing & validation pipeline (Phase 2.2.3)
- Four-class DenseNet121 inference service (Phase 2.2.4)
- Grad-CAM explainability (Phase 2.2.5+)
- Medical information RAG retrieval (Phase 2.3)
- Multi-agent orchestration (Phase 2.4)
- Report generation & audit history (Phase 2.5)

Endpoints delegate all processing logic to services in this package.
"""

from app.services.image_preprocessor import (
    CorruptImageError,
    DecompressionBombError,
    EmptyImageError,
    ImagePreprocessingError,
    ImagePreprocessor,
    ImagePreprocessorConfig,
    ImageTooLargeError,
    InvalidImageModeError,
    PreprocessedImageResult,
    TensorValidationError,
    UnsupportedFormatError,
    get_image_preprocessor,
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
from app.services.model_loader import (
    ModelContractError,
    ModelContractMetadata,
    ModelLoader,
    ModelNotLoadedError,
    get_model_loader,
)
from app.services.gradcam_service import (
    GradCAMError,
    GradCAMGenerationError,
    GradCAMGradientError,
    GradCAMLayerNotFoundError,
    GradCAMService,
    InvalidGradCAMInputError,
    get_gradcam_service,
)
from app.services.gradcam_visualizer import (
    GradCAMVisualizer,
    InvalidVisualizerInputError,
    VisualizerError,
    VisualizerResult,
    get_gradcam_visualizer,
)

__all__ = [
    # Model Loading Lifecycle
    "ModelLoader",
    "get_model_loader",
    "ModelContractMetadata",
    "ModelContractError",
    "ModelNotLoadedError",
    # Image Preprocessing & Input Validation
    "ImagePreprocessor",
    "get_image_preprocessor",
    "ImagePreprocessorConfig",
    "PreprocessedImageResult",
    "ImagePreprocessingError",
    "EmptyImageError",
    "ImageTooLargeError",
    "UnsupportedFormatError",
    "CorruptImageError",
    "DecompressionBombError",
    "InvalidImageModeError",
    "TensorValidationError",
    # Inference Service & Prediction Pipeline
    "InferenceService",
    "get_inference_service",
    "PredictionResult",
    "InferenceError",
    "ModelUnavailableError",
    "InvalidModelOutputError",
    "InferenceExecutionError",
    "DEFAULT_CLASS_MAPPING",
    "DEFAULT_DISPLAY_NAMES",
    "RESEARCH_DISCLAIMER",
    # Grad-CAM Explainability Pipeline (Phase 2.2.6)
    "GradCAMService",
    "get_gradcam_service",
    "GradCAMError",
    "InvalidGradCAMInputError",
    "GradCAMLayerNotFoundError",
    "GradCAMGradientError",
    "GradCAMGenerationError",
    # Grad-CAM Visualization & Overlay (Phase 2.2.6)
    "GradCAMVisualizer",
    "get_gradcam_visualizer",
    "VisualizerResult",
    "VisualizerError",
    "InvalidVisualizerInputError",
]
