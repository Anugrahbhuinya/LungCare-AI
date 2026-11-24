"""Pydantic schemas for standardized API error responses."""

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error metadata."""

    message: str = Field(
        description="Human-readable description of the error.",
        examples=["The requested resource was not found."],
    )
    error_type: str = Field(
        description="Machine-readable classification of the error.",
        examples=["NotFoundError"],
    )
    details: Optional[Any] = Field(
        default=None,
        description="Optional additional context or field validation details.",
    )


class ErrorResponse(BaseModel):
    """Standardized top-level error response envelope."""

    status: str = Field(
        default="error",
        description="Status flag indicating an error response.",
        examples=["error"],
    )
    error: ErrorDetail = Field(
        description="Error details payload.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the error occurred.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request correlation identifier (matches X-Request-ID response header).",
        examples=["f81d4fae-7dec-11d0-a765-00a0c91e6bf6"],
    )


# ==============================================================================
# Reusable OpenAPI Error Response Documentation Dictionaries
# ==============================================================================

PREDICTION_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": "Validation or client input error (empty upload, corrupt bytes, invalid mode, or out-of-bounds parameters).",
    },
    413: {
        "model": ErrorResponse,
        "description": "Payload or image dimensional limits exceeded (file size or pixel count threshold).",
    },
    415: {
        "model": ErrorResponse,
        "description": "Unsupported media format (allowed formats: JPEG, PNG, WEBP).",
    },
    422: {
        "model": ErrorResponse,
        "description": "Request schema validation failed (returns standardized ErrorResponse envelope).",
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal server or neural network execution error (sanitized in production).",
    },
    502: {
        "model": ErrorResponse,
        "description": "Model output contract violation, gradient backpropagation failure, or architecture mismatch.",
    },
    503: {
        "model": ErrorResponse,
        "description": "DenseNet121 model unavailable, lifecycle draining active, or coordinator not ready.",
    },
}

PROBE_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    500: {
        "model": ErrorResponse,
        "description": "Internal server failure during probe execution.",
    },
}

READY_PROBE_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    500: {
        "model": ErrorResponse,
        "description": "Internal server failure during readiness probe execution.",
    },
    503: {
        "model": ErrorResponse,
        "description": "Service or ML model subsystem not ready to accept traffic (model initialization error or coordinator draining active).",
    },
}
