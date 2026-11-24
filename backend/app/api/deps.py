"""FastAPI application-wide dependency providers.

Centralizes dependency injection providers for service singletons and runtime configuration,
decoupling endpoint route definitions from service instantiation and lifecycle management.
Follows lazy evaluation to prevent TensorFlow or model initialization during module import.
"""

from __future__ import annotations

from typing import Any
from app.core.config import Settings, get_settings
from app.services.image_preprocessor import ImagePreprocessor, get_image_preprocessor
from app.services.inference_service import InferenceService, get_inference_service
from app.services.gradcam_service import GradCAMService, get_gradcam_service
from app.services.gradcam_visualizer import GradCAMVisualizer, get_gradcam_visualizer
from app.services.ml_coordinator import MLExecutionCoordinator, get_ml_coordinator

__all__ = [
    "get_settings_dep",
    "get_ml_coordinator_dep",
    "get_ml_coordinator",
    "get_inference_service_dep",
    "get_gradcam_service_dep",
    "get_gradcam_visualizer_dep",
    "get_image_preprocessor_dep",
]


def get_settings_dep() -> Settings:
    """Dependency provider returning application runtime configuration."""
    return get_settings()


def get_ml_coordinator_dep() -> MLExecutionCoordinator:
    """Dependency provider returning application MLExecutionCoordinator singleton instance."""
    return get_ml_coordinator()


def get_inference_service_dep() -> InferenceService:
    """Dependency provider returning application InferenceService singleton instance."""
    return get_inference_service()


def get_gradcam_service_dep() -> GradCAMService:
    """Dependency provider returning application GradCAMService singleton instance."""
    return get_gradcam_service()


def get_gradcam_visualizer_dep() -> GradCAMVisualizer:
    """Dependency provider returning application GradCAMVisualizer singleton instance."""
    return get_gradcam_visualizer()


def get_image_preprocessor_dep() -> ImagePreprocessor:
    """Dependency provider returning application ImagePreprocessor singleton instance."""
    return get_image_preprocessor()
