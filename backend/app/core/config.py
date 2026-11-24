"""Application configuration module using Pydantic Settings.

Manages runtime settings loaded from environment variables and .env files
with strict validation, type safety, and security guardrails.
"""

from functools import lru_cache
import json
from typing import Any, List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings with environment variable overrides."""

    # Application Metadata
    APP_NAME: str = "LungCare AI API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    ENABLE_DOCS: Optional[bool] = None
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Image Preprocessing & Upload Settings (Phase 2.2.3)
    MAX_IMAGE_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_PIXELS: int = 25_000_000             # 25 MP safe decompression limit
    ALLOWED_IMAGE_FORMATS: List[str] = ["JPEG", "PNG", "WEBP"]
    IMAGE_TARGET_SIZE: int = 224

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        """Parse boolean debug setting safely handling non-boolean env var values."""
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in ("true", "1", "yes", "on", "t"):
                return True
            if val_lower in ("false", "0", "no", "off", "f"):
                return False
            return False
        return bool(value)

    @field_validator("ENABLE_DOCS", mode="before")
    @classmethod
    def parse_enable_docs(cls, value: Any) -> Optional[bool]:
        """Parse boolean documentation flag safely handling string values."""
        if value is None:
            return None
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in ("true", "1", "yes", "on", "t"):
                return True
            if val_lower in ("false", "0", "no", "off", "f"):
                return False
            return False
        return bool(value)

    @property
    def is_docs_enabled(self) -> bool:
        """Determine whether OpenAPI documentation endpoints should be active.

        Precedence rules:
        1. Explicit ENABLE_DOCS setting (True/False) is authoritative across all environments.
        2. If ENABLE_DOCS is None, documentation is enabled ONLY in the 'development' environment
           OR when DEBUG is True.
        3. In 'production' or any other non-development environment where ENABLE_DOCS is unset
           and DEBUG is False, documentation endpoints are strictly disabled.
        """
        if self.ENABLE_DOCS is not None:
            return self.ENABLE_DOCS
        return self.DEBUG or self.ENVIRONMENT.lower() == "development"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, List[str], Any]) -> List[str]:
        """Parse CORS origins from a list, JSON array string, or comma-separated string."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except json.JSONDecodeError:
                    pass
            # Split comma-separated string
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        elif isinstance(value, (list, tuple, set)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return []

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_security(cls, origins: List[str]) -> List[str]:
        """Disallow wildcard CORS when credentials and production environments are targeted."""
        if "*" in origins:
            raise ValueError(
                "Wildcard '*' origin is prohibited when credentials are enabled. "
                "Specify explicit allowed origins."
            )
        return origins

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, level: str) -> str:
        """Ensure standard logging levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_level = level.upper()
        if upper_level not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL '{level}'. Allowed: {valid_levels}")
        return upper_level

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()

