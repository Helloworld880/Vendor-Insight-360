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


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Vendor Insight 360 API")
        self.app_env = os.getenv("APP_ENV", "development")
        self.api_prefix = os.getenv("API_PREFIX", "/api/v1")
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.postgres_user = _require_env("POSTGRES_USER")
        self.postgres_password = _require_env("POSTGRES_PASSWORD")
        self.postgres_db = _require_env("POSTGRES_DB")
        self.postgres_test_db = _require_env("POSTGRES_TEST_DB")
        self.postgres_admin_db = _require_env("POSTGRES_ADMIN_DB")
        self.postgres_host = _require_env("POSTGRES_HOST")
        self.postgres_port = int(_require_env("POSTGRES_PORT"))
        self.redis_host = _require_env("REDIS_HOST")
        self.redis_port = int(_require_env("REDIS_PORT"))
        self.redis_db = int(_require_env("REDIS_DB"))
        self.redis_test_db = int(_require_env("REDIS_TEST_DB"))
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
    def active_postgres_db(self) -> str:
        return self.postgres_test_db if self.is_test else self.postgres_db

    @property
    def active_redis_db(self) -> int:
        return self.redis_test_db if self.is_test else self.redis_db

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.active_postgres_db}"
        )

    @property
    def admin_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_admin_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.active_redis_db}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
