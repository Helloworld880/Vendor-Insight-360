from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return value


def _optional_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Vendor Insight 360 API")
        self.app_env = os.getenv("APP_ENV", "development")
        self.api_prefix = os.getenv("API_PREFIX", "/api/v1")
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
        self.database_url_override = os.getenv("DATABASE_URL")
        self.admin_database_url_override = os.getenv("ADMIN_DATABASE_URL")
        self.local_sqlite_path = Path(os.getenv("LOCAL_SQLITE_PATH", "data/vendor_insight_local.db"))
        self.postgres_user = os.getenv("POSTGRES_USER")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD")
        self.postgres_db = os.getenv("POSTGRES_DB")
        self.postgres_test_db = os.getenv("POSTGRES_TEST_DB", self.postgres_db or "vendor_test_db")
        self.postgres_admin_db = os.getenv("POSTGRES_ADMIN_DB", "postgres")
        self.postgres_host = os.getenv("POSTGRES_HOST")
        self.postgres_port = _optional_int("POSTGRES_PORT")
        if self.should_use_postgres:
            self.postgres_user = self.postgres_user or _require_env("POSTGRES_USER")
            self.postgres_password = self.postgres_password or _require_env("POSTGRES_PASSWORD")
            self.postgres_db = self.postgres_db or _require_env("POSTGRES_DB")
            self.postgres_test_db = os.getenv("POSTGRES_TEST_DB", _require_env("POSTGRES_TEST_DB"))
            self.postgres_admin_db = os.getenv("POSTGRES_ADMIN_DB", _require_env("POSTGRES_ADMIN_DB"))
            self.postgres_host = self.postgres_host or _require_env("POSTGRES_HOST")
            self.postgres_port = self.postgres_port or int(_require_env("POSTGRES_PORT"))
        self.redis_url_override = os.getenv("REDIS_URL")
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_port = _optional_int("REDIS_PORT")
        self.redis_db = _optional_int("REDIS_DB", 0) or 0
        self.redis_test_db = _optional_int("REDIS_TEST_DB", 1) or 1
        if self.should_use_external_redis:
            self.redis_host = self.redis_host or _require_env("REDIS_HOST")
            self.redis_port = self.redis_port or int(_require_env("REDIS_PORT"))
            self.redis_db = int(os.getenv("REDIS_DB", str(self.redis_db)))
            self.redis_test_db = int(os.getenv("REDIS_TEST_DB", str(self.redis_test_db)))
        self.secret_key = os.getenv("SECRET_KEY", "replace-with-strong-secret")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
        self.refresh_token_expire_minutes = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_dir = Path(os.getenv("LOG_DIR", "logs"))
        self.log_file_name = os.getenv("LOG_FILE_NAME", "vendor_analytics.jsonl")
        self.log_max_bytes = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
        self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
            if origin.strip()
        ]
        self.rate_limit_per_window = int(os.getenv("RATE_LIMIT_PER_WINDOW", "120"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "120"))
        self.admin_username = os.getenv("ADMIN_USERNAME", "admin")
        self.admin_password = os.getenv("ADMIN_PASSWORD", "StrongAdminPass123")
        self.admin_role = os.getenv("ADMIN_ROLE", "admin")
        self.model_registry_path = Path(os.getenv("MODEL_REGISTRY_PATH", "artifacts/model_registry"))
        self.default_model_name = os.getenv("DEFAULT_MODEL_NAME", "vendor_risk")
        self.training_dataset_path = Path(
            os.getenv("TRAINING_DATASET_PATH", "data/vendor_risk_training.csv")
        )
        self.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / self.log_file_name

    @property
    def is_test(self) -> bool:
        return self.app_env.lower() in {"test", "testing"}

    @property
    def should_use_postgres(self) -> bool:
        if self.database_url_override:
            return self.database_url_override.startswith("postgresql")
        return self.app_env.lower() in {"test", "testing", "production"}

    @property
    def should_use_external_redis(self) -> bool:
        if self.redis_url_override:
            return not self.redis_url_override.startswith("memory://")
        return self.app_env.lower() in {"test", "testing", "production"}

    @property
    def active_postgres_db(self) -> str:
        return self.postgres_test_db if self.is_test else self.postgres_db

    @property
    def active_redis_db(self) -> int:
        return self.redis_test_db if self.is_test else self.redis_db

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        if not self.should_use_postgres:
            self.local_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.local_sqlite_path.as_posix()}"
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.active_postgres_db}"
        )

    @property
    def admin_database_url(self) -> str:
        if self.admin_database_url_override:
            return self.admin_database_url_override
        if not self.should_use_postgres:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_admin_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_url_override:
            return self.redis_url_override
        if not self.should_use_external_redis:
            return "memory://local"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.active_redis_db}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
