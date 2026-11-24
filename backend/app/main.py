"""Main application entry point for LungCare AI Backend API.

Initializes FastAPI with enterprise middleware, structured logging, centralized
routing, CORS security, and consistent exception handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.request_context import get_request_id, reset_request_id, set_request_id
from app.middleware.access_logging import AccessLoggingMiddleware
from app.middleware.request_correlation import RequestCorrelationMiddleware
from app.schemas.error import ErrorDetail, ErrorResponse, PROBE_ERROR_RESPONSES
from app.schemas.metadata import RootMetadataResponse


logger = get_logger("lungcare.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL)
    logger.info(
        "Starting %s [version: %s, env: %s, debug: %s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )

    # Initialize ML model lifecycle and coordinator during startup
    try:
        from app.services.ml_coordinator import get_ml_coordinator
        from app.services.model_loader import get_model_loader

        coordinator = get_ml_coordinator()
        loader = get_model_loader()
        loader.load_model()
        logger.info(
            "DenseNet121 model and coordinator initialized successfully (state: %s).",
            coordinator.state.value,
        )
    except Exception as exc:
        # Support degraded startup state: application remains alive for health/liveness probes
        logger.warning(
            "Model initialization failed during startup (degraded state active): %s",
            exc,
        )

    yield

    # Clean shutdown: graceful draining and release of model resources
    try:
        from app.services.ml_coordinator import get_ml_coordinator
        from app.services.model_loader import get_model_loader

        coordinator = get_ml_coordinator()
        loader = get_model_loader()

        # Step 1: Transition coordinator to DRAINING to reject new incoming ML requests
        coordinator.start_draining()

        # Step 2: Offload synchronous draining wait to worker threadpool if operations are active
        if coordinator.active_operations > 0:
            from starlette.concurrency import run_in_threadpool
            drain_timeout = 5.0
            drained = await run_in_threadpool(coordinator.wait_for_active_operations, timeout=drain_timeout)
            if not drained:
                logger.error(
                    "Shutdown drain timed out after %.1f seconds with %d active operations remaining. "
                    "Retaining model memory reference to prevent segmentation faults during in-flight execution.",
                    drain_timeout,
                    coordinator.active_operations,
                )
                return

        # Step 3: Unload model only after active operations reach zero
        loader.unload()
        logger.info("DenseNet121 model resources released successfully during application shutdown.")
    except Exception as exc:
        logger.warning(
            "Failed to cleanly release model resources during application shutdown: %s",
            exc,
        )
    finally:
        try:
            from app.services.ml_coordinator import get_ml_coordinator
            get_ml_coordinator().complete_unloading()
        except Exception:
            pass

    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    """Factory function to build and configure the FastAPI application instance."""
    settings = get_settings()

    # Determine OpenAPI docs exposure based on explicit configuration
    docs_enabled = settings.is_docs_enabled
    docs_url = "/docs" if docs_enabled else None
    redoc_url = "/redoc" if docs_enabled else None
    openapi_url = f"{settings.API_V1_PREFIX}/openapi.json" if docs_enabled else None

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise backend API for LungCare AI — an extensible AI-powered "
            "chest X-ray research and medical information assistance platform. "
            "Supports DenseNet121 multi-class classification, Grad-CAM visual "
            "explainability, and medical information assistance. "
            "(Research prototype: not approved for primary diagnostic use.)"
        ),
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # --------------------------------------------------------------------------
    # CORS Configuration
    # --------------------------------------------------------------------------
    # Strictly bind explicit origins and disallow wildcards when credentials enabled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )
    # Access logging middleware: records structured access events within correlation context
    app.add_middleware(AccessLoggingMiddleware)
    # Request correlation middleware: extracts/generates X-Request-ID, binds context, injects header
    app.add_middleware(RequestCorrelationMiddleware)

    # --------------------------------------------------------------------------
    # Centralized Exception Handlers (Phase 2.3.4.3)
    # --------------------------------------------------------------------------
    register_exception_handlers(app)

    # --------------------------------------------------------------------------
    # Route Registration
    # --------------------------------------------------------------------------
    # Root-level health probes (standard for Kubernetes, ALB, container runners)
    app.include_router(health_router, prefix="", tags=["Root Probes"])

    # Versioned API routes under /api/v1
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Root OpenAPI schema alias (supports tooling expecting /openapi.json at root)
    if docs_enabled:
        @app.get(
            "/openapi.json",
            include_in_schema=False,
            summary="Root OpenAPI Schema Alias",
        )
        async def root_openapi_schema() -> dict:
            """Root alias endpoint returning the identical OpenAPI schema as the versioned URL."""
            return app.openapi()

    # Root welcome / metadata route
    @app.get(
        "/",
        response_model=RootMetadataResponse,
        responses=PROBE_ERROR_RESPONSES,
        summary="API Root Information",
        description="Root entry point displaying service name, version, and documentation link.",
        tags=["System"],
    )
    async def root() -> RootMetadataResponse:
        return RootMetadataResponse(
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            docs_url="/docs" if docs_enabled else "disabled_in_production",
            openapi_url=(
                f"{settings.API_V1_PREFIX}/openapi.json"
                if docs_enabled
                else "disabled_in_production"
            ),
            openapi_alias_url="/openapi.json" if docs_enabled else "disabled_in_production",
            health_url="/health",
            ready_url="/ready",
            api_v1_prefix=settings.API_V1_PREFIX,
        )

    return app


# Application entry point for ASGI servers (uvicorn app.main:app)
app = create_application()
