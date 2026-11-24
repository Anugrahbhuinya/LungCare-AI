"""API Version 1 centralized router aggregator."""

from fastapi import APIRouter, Depends, status
from app.api.v1.endpoints import health, predictions
from app.core.config import Settings, get_settings
from app.schemas.error import PROBE_ERROR_RESPONSES
from app.schemas.metadata import APIMetadataResponse

api_router = APIRouter()

# Register endpoint modules
api_router.include_router(health.router)
api_router.include_router(predictions.router)


@api_router.get(
    "/",
    response_model=APIMetadataResponse,
    status_code=status.HTTP_200_OK,
    responses=PROBE_ERROR_RESPONSES,
    summary="API v1 Discovery & Metadata",
    description=(
        "Returns version 1 API metadata, documentation references, and probe links."
    ),
    tags=["System"],
)
async def get_v1_metadata(
    settings: Settings = Depends(get_settings),
) -> APIMetadataResponse:
    """Return discovery metadata for API Version 1."""
    docs_enabled = settings.is_docs_enabled
    return APIMetadataResponse(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        api_version="v1",
        docs_url="/docs" if docs_enabled else "disabled_in_production",
        openapi_url=(
            f"{settings.API_V1_PREFIX}/openapi.json"
            if docs_enabled
            else "disabled_in_production"
        ),
        health_url=f"{settings.API_V1_PREFIX}/health",
        ready_url=f"{settings.API_V1_PREFIX}/ready",
        endpoints={
            "health": f"{settings.API_V1_PREFIX}/health",
            "ready": f"{settings.API_V1_PREFIX}/ready",
            "predictions": f"{settings.API_V1_PREFIX}/predictions",
            "explain": f"{settings.API_V1_PREFIX}/predictions/explain",
        },
    )
