"""Health and readiness probe endpoints."""

from typing import Any, Union
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from app.core.config import Settings, get_settings
from app.core.request_context import get_request_id
from app.schemas.error import ErrorDetail, ErrorResponse, PROBE_ERROR_RESPONSES, READY_PROBE_ERROR_RESPONSES
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses=PROBE_ERROR_RESPONSES,
    summary="Service Health Check",
    description="Liveness probe to confirm that the API service process is up and running.",
    tags=["Health & Probes"],
)
async def check_health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return basic service health information."""
    return HealthResponse(
        status="healthy",
        service="lungcare-api",
        version=settings.APP_VERSION,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    responses=READY_PROBE_ERROR_RESPONSES,
    summary="Service Readiness Check",
    description=(
        "Readiness probe to confirm the service is ready to accept incoming traffic. "
        "Reports subsystem initialization status and DenseNet121 model availability. "
        "Returns HTTP 200 when ready or pending initialization, and HTTP 503 when in ERROR or DRAINING."
    ),
    tags=["Health & Probes"],
)
async def check_readiness(
    settings: Settings = Depends(get_settings),
) -> Union[ReadinessResponse, JSONResponse]:
    """Check readiness status of the application and its subsystems.
    
    Lifecycle mapping:
    - READY: HTTP 200 OK with ReadinessResponse (model_loaded=True)
    - ERROR: HTTP 503 Service Unavailable with ErrorResponse (status='error', error_type='MODEL_NOT_READY')
    - DRAINING: HTTP 503 Service Unavailable with ErrorResponse (status='error', error_type='SERVICE_DRAINING')
    - UNLOADED / Pending: HTTP 200 OK with ReadinessResponse (model_loaded=False), preserving intended deployment behavior
    """
    from app.services.model_loader import get_model_loader
    from app.services.ml_coordinator import ModelLifecycleState

    loader = get_model_loader()
    coordinator = loader.coordinator
    req_id = get_request_id()

    # 1. DRAINING: Server is shutting down and draining operations -> HTTP 503
    if coordinator is not None and (
        coordinator.is_draining or coordinator.state == ModelLifecycleState.DRAINING
    ):
        active_count = coordinator.active_operations
        error_resp = ErrorResponse(
            status="error",
            error=ErrorDetail(
                message="Server is shutting down and draining active operations; service is not ready to accept traffic.",
                error_type="SERVICE_DRAINING",
                details={
                    "lifecycle_state": "DRAINING",
                    "dependencies": {
                        "ml_model": f"draining (active_operations={active_count})",
                        "database": "not_configured",
                    },
                },
            ),
            request_id=req_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_resp.model_dump(mode="json"),
        )

    # 2. ERROR: Model initialization failed or coordinator in ERROR -> HTTP 503
    if loader.load_error is not None or (
        coordinator is not None and coordinator.state == ModelLifecycleState.ERROR
    ):
        error_resp = ErrorResponse(
            status="error",
            error=ErrorDetail(
                message="DenseNet121 model failed to initialize or encountered an error and is not ready to accept traffic.",
                error_type="MODEL_NOT_READY",
                details={
                    "lifecycle_state": "ERROR",
                    "dependencies": {
                        "ml_model": "unloaded (failed to load: model unavailable)",
                        "database": "not_configured",
                    },
                },
            ),
            request_id=req_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_resp.model_dump(mode="json"),
        )

    # 3. READY: Model is successfully loaded into memory and coordinator is ready -> HTTP 200
    if loader.is_loaded:
        return ReadinessResponse(
            status="ready",
            service="lungcare-api",
            version=settings.APP_VERSION,
            model_loaded=True,
            environment=settings.ENVIRONMENT,
            dependencies={
                "ml_model": "loaded",
                "database": "not_configured",
            },
        )

    # 4. UNLOADED / Pending initialization or disabled: HTTP 200
    # Preserves explicitly intended deployment behavior for pre-warm probes
    return ReadinessResponse(
        status="ready",
        service="lungcare-api",
        version=settings.APP_VERSION,
        model_loaded=False,
        environment=settings.ENVIRONMENT,
        dependencies={
            "ml_model": "unloaded (model initialization pending or disabled)",
            "database": "not_configured",
        },
    )
