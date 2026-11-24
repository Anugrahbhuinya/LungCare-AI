"""Pydantic schemas for health and readiness probes."""

from datetime import datetime, timezone
from typing import Any, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for basic liveness probe."""

    status: str = Field(
        default="healthy",
        description="Liveness status of the API service.",
        examples=["healthy"],
    )
    service: str = Field(
        default="lungcare-api",
        description="Unique identifier for the service.",
        examples=["lungcare-api"],
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the application.",
        examples=["1.0.0"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the probe evaluation.",
    )


class ReadinessResponse(BaseModel):
    """Schema for readiness probe confirming dependency initialization."""

    status: str = Field(
        default="ready",
        description="Readiness status of the API service to accept incoming traffic.",
        examples=["ready"],
    )
    service: str = Field(
        default="lungcare-api",
        description="Unique identifier for the service.",
        examples=["lungcare-api"],
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the application.",
        examples=["1.0.0"],
    )
    model_loaded: bool = Field(
        default=False,
        description="Flag indicating if the DenseNet121 ML model is loaded into memory.",
        examples=[False],
    )
    environment: str = Field(
        description="Current runtime environment (e.g., development, production).",
        examples=["development"],
    )
    dependencies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Status of subsystem dependencies.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the readiness evaluation.",
    )

