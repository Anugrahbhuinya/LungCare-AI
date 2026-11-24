"""Centralized Domain Error Handling and Exception Handlers for LungCare AI Backend.

Provides a unified, machine-readable error architecture:
1. Shared error response builder (create_error_response) guaranteeing consistent
   Pydantic v2 ErrorResponse serialization and X-Request-ID header correlation.
2. Domain exception handlers preserving machine-readable error codes (e.g. EMPTY_IMAGE,
   UNSUPPORTED_FORMAT, MODEL_UNAVAILABLE, INVALID_MODEL_OUTPUT, GRADCAM_GRADIENT_ERROR).
3. Production security sanitization ensuring zero internal stack trace or memory leak
   when DEBUG=False.
4. Backward-compatible handlers for StarletteHTTPException, RequestValidationError,
   and unhandled generic Exceptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Type
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.request_context import get_request_id, reset_request_id, set_request_id
from app.schemas.error import ErrorDetail, ErrorResponse
from app.services.image_preprocessor import (
    CorruptImageError,
    DecompressionBombError,
    EmptyImageError,
    ImagePreprocessingError,
    ImageTooLargeError,
    InvalidImageModeError,
    TensorValidationError,
    UnsupportedFormatError,
)
from app.services.inference_service import (
    InferenceError,
    InferenceExecutionError,
    InvalidModelOutputError,
    ModelUnavailableError,
)
from app.services.gradcam_service import (
    GradCAMError,
    GradCAMGenerationError,
    GradCAMGradientError,
    GradCAMLayerNotFoundError,
    InvalidGradCAMInputError,
)
from app.services.gradcam_visualizer import (
    InvalidVisualizerInputError,
    VisualizerError,
)
from app.services.ml_coordinator import (
    MLCoordinatorError,
    MLNotReadyError,
)
from app.services.model_loader import (
    ModelContractError,
    ModelNotLoadedError,
)

logger = logging.getLogger("lungcare.errors")


def create_error_response(
    request: Request,
    status_code: int,
    message: str,
    error_type: str,
    details: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Build and serialize a standardized ErrorResponse with matching X-Request-ID."""
    req_id = get_request_id(request)
    error_response = ErrorResponse(
        status="error",
        error=ErrorDetail(
            message=message,
            error_type=error_type,
            details=details,
        ),
        request_id=req_id,
    )
    response_headers = {"X-Request-ID": req_id}
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json"),
        headers=response_headers,
    )


# ------------------------------------------------------------------------------
# Domain Exception Resolvers
# ------------------------------------------------------------------------------

def resolve_image_preprocessing_error(
    exc: ImagePreprocessingError, debug: bool = False
) -> Tuple[int, str, str, Optional[Any]]:
    """Resolve (status_code, error_type, message, details) for ImagePreprocessingError."""
    if isinstance(exc, EmptyImageError):
        return status.HTTP_400_BAD_REQUEST, "EMPTY_IMAGE", exc.message, exc.details or None
    elif isinstance(exc, UnsupportedFormatError):
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "UNSUPPORTED_FORMAT", exc.message, exc.details or None
    elif isinstance(exc, DecompressionBombError):
        return status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "DECOMPRESSION_BOMB", exc.message, exc.details or None
    elif isinstance(exc, ImageTooLargeError):
        return status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "IMAGE_TOO_LARGE", exc.message, exc.details or None
    elif isinstance(exc, CorruptImageError):
        return status.HTTP_400_BAD_REQUEST, "CORRUPT_IMAGE", exc.message, exc.details or None
    elif isinstance(exc, InvalidImageModeError):
        return status.HTTP_400_BAD_REQUEST, "INVALID_IMAGE_MODE", exc.message, exc.details or None
    elif isinstance(exc, TensorValidationError):
        msg = f"Tensor validation failure: {exc.message}" if debug else "Image tensor validation failed."
        details = {"error_code": "TENSOR_VALIDATION_ERROR"} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "TENSOR_VALIDATION_ERROR", msg, details
    else:
        err_code = getattr(exc, "error_code", "IMAGE_PREPROCESSING_ERROR")
        return status.HTTP_400_BAD_REQUEST, err_code, exc.message, exc.details or None


def resolve_inference_error(
    exc: InferenceError, debug: bool = False
) -> Tuple[int, str, str, Optional[Any]]:
    """Resolve (status_code, error_type, message, details) for InferenceError."""
    if isinstance(exc, ModelUnavailableError):
        msg = "DenseNet121 model is not loaded or temporarily unavailable. Please retry later."
        return status.HTTP_503_SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE", msg, exc.details or None
    elif isinstance(exc, InvalidModelOutputError):
        msg = "Model output validation failed. Output violated integrity contracts."
        return status.HTTP_502_BAD_GATEWAY, "INVALID_MODEL_OUTPUT", msg, exc.details or None
    elif isinstance(exc, InferenceExecutionError):
        msg = f"Inference execution failure: {exc.message}" if debug else "An unexpected error occurred during model inference."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "INFERENCE_EXECUTION_ERROR", msg, details
    else:
        err_code = getattr(exc, "error_code", "INFERENCE_ERROR")
        msg = f"Inference failure: {exc.message}" if debug else "An unexpected error occurred during model inference."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, err_code, msg, details


def resolve_gradcam_error(
    exc: GradCAMError, debug: bool = False
) -> Tuple[int, str, str, Optional[Any]]:
    """Resolve (status_code, error_type, message, details) for GradCAMError."""
    if isinstance(exc, InvalidGradCAMInputError):
        return status.HTTP_400_BAD_REQUEST, "INVALID_GRADCAM_INPUT", exc.message, exc.details or None
    elif isinstance(exc, GradCAMGradientError):
        msg = f"Grad-CAM gradient backpropagation failed: {exc.message}"
        return status.HTTP_502_BAD_GATEWAY, "GRADCAM_GRADIENT_ERROR", msg, exc.details or None
    elif isinstance(exc, GradCAMLayerNotFoundError):
        msg = f"Target layer failure: {exc.message}" if debug else "An unexpected error occurred during Grad-CAM explainability generation."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "GRADCAM_LAYER_NOT_FOUND", msg, details
    elif isinstance(exc, GradCAMGenerationError):
        msg = f"Grad-CAM generation failure: {exc.message}" if debug else "An unexpected error occurred during Grad-CAM explainability generation."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "GRADCAM_GENERATION_ERROR", msg, details
    else:
        err_code = getattr(exc, "error_code", "GRADCAM_ERROR")
        msg = f"Grad-CAM failure: {exc.message}" if debug else "An unexpected error occurred during Grad-CAM explainability generation."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, err_code, msg, details


def resolve_visualizer_error(
    exc: VisualizerError, debug: bool = False
) -> Tuple[int, str, str, Optional[Any]]:
    """Resolve (status_code, error_type, message, details) for VisualizerError."""
    if isinstance(exc, InvalidVisualizerInputError):
        return status.HTTP_400_BAD_REQUEST, "INVALID_VISUALIZER_INPUT", exc.message, exc.details or None
    else:
        err_code = getattr(exc, "error_code", "VISUALIZER_ERROR")
        msg = f"Overlay rendering failure: {exc.message}" if debug else "An error occurred while rendering the visual explainability overlay."
        details = {"exception_type": exc.__class__.__name__} if debug else None
        return status.HTTP_500_INTERNAL_SERVER_ERROR, err_code, msg, details


def resolve_coordinator_error(
    exc: MLCoordinatorError, debug: bool = False
) -> Tuple[int, str, str, Optional[Any]]:
    """Resolve (status_code, error_type, message, details) for MLCoordinatorError."""
    if isinstance(exc, MLNotReadyError):
        msg = "DenseNet121 model is not ready for execution. Please retry later."
        return status.HTTP_503_SERVICE_UNAVAILABLE, "MODEL_NOT_READY", msg, exc.details or None
    err_code = getattr(exc, "error_code", "ML_COORDINATOR_ERROR")
    return status.HTTP_503_SERVICE_UNAVAILABLE, err_code, "DenseNet121 model coordinator error. Please retry later.", exc.details or None


# ------------------------------------------------------------------------------
# Centralized Handlers
# ------------------------------------------------------------------------------

async def image_preprocessing_exception_handler(request: Request, exc: ImagePreprocessingError) -> JSONResponse:
    """Handle ImagePreprocessingError and its subclasses."""
    settings = get_settings()
    status_code, error_type, message, details = resolve_image_preprocessing_error(exc, debug=settings.DEBUG)
    if status_code >= 500:
        logger.exception("Image preprocessing internal failure [%s]: %s", error_type, exc)
    else:
        logger.warning("Image preprocessing validation error [%s]: %s (path: %s)", error_type, exc, request.url.path)
    return create_error_response(request, status_code, message, error_type, details)


async def inference_exception_handler(request: Request, exc: InferenceError) -> JSONResponse:
    """Handle InferenceError and its subclasses."""
    settings = get_settings()
    status_code, error_type, message, details = resolve_inference_error(exc, debug=settings.DEBUG)
    if status_code >= 500 and status_code not in (502, 503):
        logger.exception("Inference execution failure [%s]: %s", error_type, exc)
    else:
        logger.warning("Inference service rejection [%s]: %s (path: %s)", error_type, exc, request.url.path)
    return create_error_response(request, status_code, message, error_type, details)


async def gradcam_exception_handler(request: Request, exc: GradCAMError) -> JSONResponse:
    """Handle GradCAMError and its subclasses."""
    settings = get_settings()
    status_code, error_type, message, details = resolve_gradcam_error(exc, debug=settings.DEBUG)
    if status_code >= 500 and status_code != 502:
        logger.exception("Grad-CAM explainability internal failure [%s]: %s", error_type, exc)
    else:
        logger.warning("Grad-CAM explainability warning [%s]: %s (path: %s)", error_type, exc, request.url.path)
    return create_error_response(request, status_code, message, error_type, details)


async def visualizer_exception_handler(request: Request, exc: VisualizerError) -> JSONResponse:
    """Handle VisualizerError and its subclasses."""
    settings = get_settings()
    status_code, error_type, message, details = resolve_visualizer_error(exc, debug=settings.DEBUG)
    if status_code >= 500:
        logger.exception("Visualizer rendering failure [%s]: %s", error_type, exc)
    else:
        logger.warning("Visualizer input error [%s]: %s (path: %s)", error_type, exc, request.url.path)
    return create_error_response(request, status_code, message, error_type, details)


async def coordinator_exception_handler(request: Request, exc: MLCoordinatorError) -> JSONResponse:
    """Handle MLCoordinatorError and its subclasses."""
    settings = get_settings()
    status_code, error_type, message, details = resolve_coordinator_error(exc, debug=settings.DEBUG)
    logger.warning("ML coordinator rejection [%s]: %s (path: %s)", error_type, exc, request.url.path)
    return create_error_response(request, status_code, message, error_type, details)


async def model_loader_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle ModelNotLoadedError and ModelContractError."""
    if isinstance(exc, ModelNotLoadedError):
        logger.warning("ModelLoader access before loaded: %s", exc)
        return create_error_response(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "DenseNet121 model is not loaded or temporarily unavailable. Please retry later.",
            "MODEL_UNAVAILABLE",
        )
    elif isinstance(exc, ModelContractError):
        logger.error("ModelContractError violation: %s", exc)
        return create_error_response(
            request,
            status.HTTP_502_BAD_GATEWAY,
            "Model architecture contract validation failed.",
            "MODEL_CONTRACT_ERROR",
        )
    logger.exception("Unexpected model loader failure: %s", exc)
    return create_error_response(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An unexpected model error occurred.",
        "MODEL_LOADER_ERROR",
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions with consistent error schema."""
    if isinstance(exc.detail, dict):
        msg = exc.detail.get("message", str(exc.detail))
        err_type = exc.detail.get("error_type") or exc.detail.get("error_code") or "HTTPException"
        details = exc.detail.get("details", {"status_code": exc.status_code})
    else:
        msg = str(exc.detail)
        err_type = "HTTPException"
        details = {"status_code": exc.status_code}

    return create_error_response(
        request=request,
        status_code=exc.status_code,
        message=msg,
        error_type=err_type,
        details=details,
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request schema validation errors."""
    sanitized_errors = [
        {
            "field": ".".join(str(loc) for loc in err.get("loc", [])),
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        }
        for err in exc.errors()
    ]
    return create_error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Request validation failed.",
        error_type="ValidationError",
        details=sanitized_errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught server errors without leaking internals in production."""
    settings = get_settings()
    req_id = get_request_id(request)
    token = set_request_id(req_id)
    try:
        logger.exception("Unhandled server exception processing %s: %s", request.url.path, exc)
        if settings.DEBUG:
            message = f"Internal server error: {str(exc)}"
            details = {"exception_type": exc.__class__.__name__}
        else:
            message = "An unexpected server error occurred. Please contact system support."
            details = None

        return create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_type="InternalServerError",
            details=details,
        )
    finally:
        reset_request_id(token)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers onto the FastAPI application instance."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ImagePreprocessingError, image_preprocessing_exception_handler)
    app.add_exception_handler(InferenceError, inference_exception_handler)
    app.add_exception_handler(GradCAMError, gradcam_exception_handler)
    app.add_exception_handler(VisualizerError, visualizer_exception_handler)
    app.add_exception_handler(MLCoordinatorError, coordinator_exception_handler)
    app.add_exception_handler(ModelNotLoadedError, model_loader_exception_handler)
    app.add_exception_handler(ModelContractError, model_loader_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
