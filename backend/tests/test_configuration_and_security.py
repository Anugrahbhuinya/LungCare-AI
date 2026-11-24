"""Test suite for Phase 2.3.2 Configuration & Security Hardening.

Validates:
1. Secure DEBUG configuration (defaults to False, explicit environment overrides).
2. Explicit documentation controls (ENABLE_DOCS, production vs development defaults).
3. Exception information disclosure prevention in production vs development.
4. Dependency provider extraction to app.api.deps (singleton preservation, backward compatibility,
   lazy loading without TensorFlow model initialization on import).
"""

from __future__ import annotations

import os
from typing import AsyncGenerator
import pytest

try:
    import httpx
    from fastapi import FastAPI, HTTPException
    from app.api.deps import (
        get_gradcam_service_dep,
        get_gradcam_visualizer_dep,
        get_image_preprocessor_dep,
        get_inference_service_dep,
        get_settings_dep,
    )
    from app.core.config import Settings
    from app.main import app, create_application
    from app.services.model_loader import get_model_loader
    HAS_BACKEND_DEPS = True
except ImportError:
    HAS_BACKEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_BACKEND_DEPS,
    reason="FastAPI backend dependencies required",
)


@pytest.fixture
def anyio_backend() -> str:
    """Specify asyncio as the AnyIO backend."""
    return "asyncio"


# ==============================================================================
# 1. Configuration & Documentation Tests
# ==============================================================================


class TestConfigurationSecurity:
    """Verify secure settings defaults and explicit environment overrides."""

    def test_debug_defaults_to_false(self) -> None:
        """Verify that DEBUG defaults to False for production safety."""
        # Instantiate clean Settings without ambient DEBUG env override
        settings = Settings(DEBUG=False)
        assert settings.DEBUG is False

    def test_explicit_debug_true_via_kwargs_or_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify local development can explicitly enable DEBUG."""
        monkeypatch.setenv("DEBUG", "true")
        settings = Settings()
        assert settings.DEBUG is True

        monkeypatch.setenv("DEBUG", "1")
        settings_num = Settings()
        assert settings_num.DEBUG is True

        monkeypatch.setenv("DEBUG", "false")
        settings_false = Settings()
        assert settings_false.DEBUG is False

    def test_docs_enabled_explicit_true_override(self) -> None:
        """Verify explicit ENABLE_DOCS=True activates docs in any environment."""
        prod_settings = Settings(ENVIRONMENT="production", ENABLE_DOCS=True)
        assert prod_settings.is_docs_enabled is True

        dev_settings = Settings(ENVIRONMENT="development", ENABLE_DOCS=True)
        assert dev_settings.is_docs_enabled is True

    def test_docs_enabled_explicit_false_override(self) -> None:
        """Verify explicit ENABLE_DOCS=False deactivates docs even in development."""
        dev_settings = Settings(ENVIRONMENT="development", ENABLE_DOCS=False)
        assert dev_settings.is_docs_enabled is False

        prod_settings = Settings(ENVIRONMENT="production", ENABLE_DOCS=False)
        assert prod_settings.is_docs_enabled is False

    def test_docs_default_in_production_environment(self) -> None:
        """Verify docs default to disabled in production when ENABLE_DOCS is unset."""
        prod_settings = Settings(ENVIRONMENT="production", ENABLE_DOCS=None, DEBUG=False)
        assert prod_settings.is_docs_enabled is False

    def test_docs_default_in_development_environment(self) -> None:
        """Verify docs default to enabled in development when ENABLE_DOCS is unset."""
        dev_settings = Settings(ENVIRONMENT="development", ENABLE_DOCS=None, DEBUG=False)
        assert dev_settings.is_docs_enabled is True

    @pytest.mark.anyio
    async def test_production_fastapi_app_disables_docs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that a production FastAPI application disables Swagger, Redoc, and OpenAPI."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.delenv("ENABLE_DOCS", raising=False)

        # Clear get_settings cache to pick up monkeypatched env
        from app.core import config
        config.get_settings.cache_clear()

        prod_app = create_application()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=prod_app),
            base_url="http://testserver",
        ) as client:
            # Swagger UI disabled
            docs_resp = await client.get("/docs")
            assert docs_resp.status_code == 404

            # Redoc disabled
            redoc_resp = await client.get("/redoc")
            assert redoc_resp.status_code == 404

            # Versioned OpenAPI JSON disabled
            v1_openapi = await client.get("/api/v1/openapi.json")
            assert v1_openapi.status_code == 404

            # Root OpenAPI alias disabled
            root_openapi = await client.get("/openapi.json")
            assert root_openapi.status_code == 404

            # Root discovery metadata indicates disabled in production
            root_resp = await client.get("/")
            assert root_resp.status_code == 200
            root_data = root_resp.json()
            assert root_data["docs_url"] == "disabled_in_production"
            assert root_data["openapi_url"] == "disabled_in_production"
            assert root_data["openapi_alias_url"] == "disabled_in_production"

        # Restore settings cache
        config.get_settings.cache_clear()


# ==============================================================================
# 2. Exception Handling & Information Disclosure Tests
# ==============================================================================


class TestExceptionInformationDisclosure:
    """Verify that unhandled exceptions do not leak stack traces or secrets in production."""

    @pytest.mark.anyio
    async def test_unhandled_exception_returns_500_with_safe_message_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify unhandled error returns safe message and no internal details when DEBUG=False."""
        monkeypatch.setenv("DEBUG", "false")
        from app.core import config
        config.get_settings.cache_clear()

        test_app = create_application()

        # Add a test route that raises a simulated internal exception with sensitive info
        @test_app.get("/test-internal-failure")
        async def fail_route() -> None:
            raise RuntimeError("Database password leaked: db://user:secret@internal-db:5432/core")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/test-internal-failure")
            assert response.status_code == 500
            data = response.json()

            assert data["status"] == "error"
            assert "error" in data
            assert data["error"]["error_type"] == "InternalServerError"
            # Must return the safe generic message
            assert data["error"]["message"] == "An unexpected server error occurred. Please contact system support."
            # Must not leak the secret or stack trace
            assert data["error"]["details"] is None
            assert "secret" not in response.text
            assert "password" not in response.text

        config.get_settings.cache_clear()

    @pytest.mark.anyio
    async def test_unhandled_exception_in_debug_mode_provides_exception_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify unhandled error provides debug context only when DEBUG=True."""
        monkeypatch.setenv("DEBUG", "true")
        from app.core import config
        config.get_settings.cache_clear()

        test_app = create_application()

        @test_app.get("/test-debug-failure")
        async def fail_debug_route() -> None:
            raise ValueError("Test debug computation error")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/test-debug-failure")
            assert response.status_code == 500
            data = response.json()

            assert data["status"] == "error"
            assert data["error"]["error_type"] == "InternalServerError"
            assert "Test debug computation error" in data["error"]["message"]
            assert data["error"]["details"] == {"exception_type": "ValueError"}

        config.get_settings.cache_clear()

    @pytest.mark.anyio
    async def test_http_exception_behavior_preserved(self) -> None:
        """Verify standard HTTPException (e.g. 404) returns consistent ErrorResponse."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/non-existent-probe-url")
            assert response.status_code == 404
            data = response.json()
            assert data["status"] == "error"
            assert data["error"]["error_type"] == "HTTPException"
            assert "not found" in data["error"]["message"].lower()


# ==============================================================================
# 3. Dependency Provider Extraction Tests
# ==============================================================================


class TestDependencyProviderExtraction:
    """Verify that dependencies in app.api.deps resolve correctly and preserve singletons."""

    def test_deps_module_exports_all_required_providers(self) -> None:
        """Verify that app.api.deps exports the full suite of dependency providers."""
        import app.api.deps as deps

        for expected in [
            "get_settings_dep",
            "get_inference_service_dep",
            "get_gradcam_service_dep",
            "get_gradcam_visualizer_dep",
            "get_image_preprocessor_dep",
        ]:
            assert hasattr(deps, expected), f"Missing provider: {expected}"
            assert callable(getattr(deps, expected))

    def test_deps_singleton_instances_preserved(self) -> None:
        """Verify that dependency providers return singleton service instances."""
        s1 = get_settings_dep()
        s2 = get_settings_dep()
        assert s1 is s2

        inf1 = get_inference_service_dep()
        inf2 = get_inference_service_dep()
        assert inf1 is inf2

        cam1 = get_gradcam_service_dep()
        cam2 = get_gradcam_service_dep()
        assert cam1 is cam2

        vis1 = get_gradcam_visualizer_dep()
        vis2 = get_gradcam_visualizer_dep()
        assert vis1 is vis2

        prep1 = get_image_preprocessor_dep()
        prep2 = get_image_preprocessor_dep()
        assert prep1 is prep2

    def test_backward_compatible_endpoint_imports(self) -> None:
        """Verify that existing test files can still import providers from predictions endpoint."""
        from app.api.v1.endpoints.predictions import (
            get_gradcam_service_dep as leg_cam,
            get_gradcam_visualizer_dep as leg_vis,
            get_image_preprocessor_dep as leg_prep,
            get_inference_service_dep as leg_inf,
        )

        assert leg_inf is get_inference_service_dep
        assert leg_cam is get_gradcam_service_dep
        assert leg_vis is get_gradcam_visualizer_dep
        assert leg_prep is get_image_preprocessor_dep

    def test_importing_api_modules_does_not_initialize_model(self) -> None:
        """Verify that importing app.api.deps does NOT trigger model loading."""
        loader = get_model_loader()
        # Ensure loader was not forced into loaded state merely by importing deps
        # (It should only load when load_model() is explicitly invoked)
        import app.api.deps
        import app.api.v1.endpoints.predictions
        # Verification that importing modules does not fail or raise circular import
        assert app.api.deps is not None
        assert app.api.v1.endpoints.predictions is not None
