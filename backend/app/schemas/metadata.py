"""Pydantic schemas for API root and versioned discovery metadata."""

from typing import Dict
from pydantic import BaseModel, Field


class APIMetadataResponse(BaseModel):
    """Schema for versioned API discovery metadata."""

    app_name: str = Field(
        description="Name of the application service.",
        examples=["LungCare AI API"],
    )
    version: str = Field(
        description="Semantic version of the application.",
        examples=["1.0.0"],
    )
    environment: str = Field(
        description="Current runtime environment (e.g. development, production).",
        examples=["development"],
    )
    api_version: str = Field(
        default="v1",
        description="API major version identifier.",
        examples=["v1"],
    )
    docs_url: str = Field(
        description="Path to interactive Swagger UI documentation.",
        examples=["/docs"],
    )
    openapi_url: str = Field(
        description="Path to OpenAPI JSON schema.",
        examples=["/api/v1/openapi.json"],
    )
    health_url: str = Field(
        description="Path to health check probe.",
        examples=["/api/v1/health"],
    )
    ready_url: str = Field(
        description="Path to readiness check probe.",
        examples=["/api/v1/ready"],
    )
    endpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of primary endpoint routes.",
    )


class RootMetadataResponse(BaseModel):
    """Schema for top-level system metadata."""

    app_name: str = Field(
        description="Name of the application service.",
        examples=["LungCare AI API"],
    )
    version: str = Field(
        description="Semantic version of the application.",
        examples=["1.0.0"],
    )
    environment: str = Field(
        description="Current runtime environment.",
        examples=["development"],
    )
    docs_url: str = Field(
        description="Path to interactive Swagger UI documentation.",
        examples=["/docs"],
    )
    openapi_url: str = Field(
        description="Path to primary OpenAPI JSON schema.",
        examples=["/api/v1/openapi.json"],
    )
    openapi_alias_url: str = Field(
        default="/openapi.json",
        description="Path to root alias OpenAPI JSON schema.",
        examples=["/openapi.json"],
    )
    health_url: str = Field(
        default="/health",
        description="Path to root health check probe.",
        examples=["/health"],
    )
    ready_url: str = Field(
        default="/ready",
        description="Path to root readiness check probe.",
        examples=["/ready"],
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="Base path for version 1 API routes.",
        examples=["/api/v1"],
    )

