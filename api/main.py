from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from jose.exceptions import JWTError
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.db import database_ready, get_db_session, initialize_database, wait_for_database
from database.queries import get_user_by_username
from services.model_registry import LoadedModel, ModelRegistry, ModelRegistryError
from services.vendor_service import VendorService
from utils.logging_setup import setup_logging
from utils.redis_client import redis_client
from utils.request_context import request_id_var, user_id_var
from utils.security import verify_password


settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/login")


class AppMetrics:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = Lock()
        self._request_count = 0

    def increment(self) -> None:
        with self._lock:
            self._request_count += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "request_count": self._request_count,
            }


metrics = AppMetrics()
vendor_service = VendorService()
model_registry = ModelRegistry()


class VendorBase(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=100)
    status: str = Field(default="active", min_length=1, max_length=50)
    delivery_rate: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    cost_efficiency: float = Field(ge=0, le=100)
    on_time_rate: float = Field(ge=0, le=100)
    cost_variance: float
    reliability: float = Field(ge=0, le=100)
    performance_score: float = Field(ge=0, le=100)
    risk_score: float = Field(ge=0, le=100)


class VendorCreate(VendorBase):
    pass


class VendorUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    status: str | None = Field(default=None, min_length=1, max_length=50)
    delivery_rate: float | None = Field(default=None, ge=0, le=100)
    quality_score: float | None = Field(default=None, ge=0, le=100)
    cost_efficiency: float | None = Field(default=None, ge=0, le=100)
    on_time_rate: float | None = Field(default=None, ge=0, le=100)
    cost_variance: float | None = None
    reliability: float | None = Field(default=None, ge=0, le=100)
    performance_score: float | None = Field(default=None, ge=0, le=100)
    risk_score: float | None = Field(default=None, ge=0, le=100)


class ModelPredictRequest(BaseModel):
    delivery_rate: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    cost_efficiency: float = Field(ge=0, le=100)
    on_time_rate: float = Field(ge=0, le=100)
    cost_variance: float
    reliability: float = Field(ge=0, le=100)
    performance_score: float = Field(ge=0, le=100)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class AppError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseOperationError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "DATABASE_OPERATION_FAILED"


class ModelNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "MODEL_NOT_FOUND"


class VendorNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "VENDOR_NOT_FOUND"


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "AUTHENTICATION_FAILED"


class AuthorizationError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "FORBIDDEN"


class RateLimitExceededError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "RATE_LIMIT_EXCEEDED"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "RESOURCE_CONFLICT"


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", request_id_var.get("-"))
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "details": details or {},
            }
        },
    )


def safe_headers(headers: Any) -> dict[str, Any]:
    allowed = {}
    for header_name in ("user-agent", "content-type", "x-request-id", "x-forwarded-for"):
        if header_name in headers:
            allowed[header_name] = headers.get(header_name)
    allowed["authorization_present"] = "authorization" in headers
    return allowed


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        wait_for_database()
        initialize_database()
        redis_client.ping()
        model_registry.ensure_model(settings.default_model_name)
        logger.info("application.startup.complete", extra={"event": "application.startup.complete"})
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        request_id_token = request_id_var.set(request_id)
        user_id_token = user_id_var.set(_extract_user_id_from_headers(request))
        started = time.perf_counter()
        metrics.increment()
        logger.info(
            "request.started",
            extra={
                "event": "request.started",
                "method": request.method,
                "path": request.url.path,
                "headers": safe_headers(request.headers),
            },
        )
        try:
            _enforce_rate_limit(request)
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request.failed",
                extra={
                    "event": "request.failed",
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        else:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request.completed",
                extra={
                    "event": "request.completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
            return response
        finally:
            request_id_var.reset(request_id_token)
            user_id_var.reset(user_id_token)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.error(
            "application.error",
            extra={
                "event": "application.error",
                "error_code": exc.code,
                "details": exc.details,
            },
        )
        return error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "VALIDATION_ERROR",
            "Request validation failed.",
            {"issues": exc.errors()},
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("database.exception", extra={"event": "database.exception"})
        return error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DATABASE_OPERATION_FAILED",
            "Database operation failed.",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled.exception", extra={"event": "unhandled.exception"})
        return error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_SERVER_ERROR",
            "Internal server error.",
        )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": f"{settings.app_name} is running"}

    @app.post(f"{settings.api_prefix}/login")
    def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
        # 🔥 TEMP LOGIN (bypass DB)
        if form_data.username == settings.admin_username and form_data.password == settings.admin_password:
            return {
                "access_token": _create_token("1", "admin", "admin", "access"),
                "refresh_token": _create_token("1", "admin", "admin", "refresh"),
                "token_type": "bearer",
            }
        raise AuthenticationError("Incorrect username or password.")

    @app.post(f"{settings.api_prefix}/refresh")
    def refresh_token(payload: RefreshTokenRequest) -> dict[str, str]:
        decoded = _decode_token(payload.refresh_token)
        if decoded.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token.")
        return {
            "access_token": _create_token(
                str(decoded["user_id"]),
                str(decoded["sub"]),
                str(decoded["role"]),
                "access",
            ),
            "token_type": "bearer",
        }

    @app.get(f"{settings.api_prefix}/vendors")
    def list_vendors_endpoint(
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        try:
            vendors = vendor_service.list_vendors(session)
            return {"data": vendors, "count": len(vendors)}
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc

    @app.post(f"{settings.api_prefix}/vendors", status_code=status.HTTP_201_CREATED)
    def create_vendor_endpoint(
        payload: VendorCreate,
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        try:
            vendor = vendor_service.create_vendor(session, payload.model_dump())
            invalidate_vendor_cache()
            return {"data": vendor}
        except ValueError as exc:
            raise ConflictError(str(exc), {"field": "name"}) from exc
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc

    @app.get(f"{settings.api_prefix}/vendors/{{vendor_id}}")
    def get_vendor_endpoint(
        vendor_id: int,
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        try:
            vendor = vendor_service.get_vendor(session, vendor_id)
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc
        if vendor is None:
            raise VendorNotFoundError(f"Vendor '{vendor_id}' not found.", {"vendor_id": vendor_id})
        return {"data": vendor}

    @app.put(f"{settings.api_prefix}/vendors/{{vendor_id}}")
    def update_vendor_endpoint(
        vendor_id: int,
        payload: VendorUpdate,
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        changes = payload.model_dump(exclude_none=True)
        if not changes:
            raise AppError("At least one vendor field must be provided.")
        try:
            vendor = vendor_service.update_vendor(session, vendor_id, changes)
        except ValueError as exc:
            raise ConflictError(str(exc), {"field": "name"}) from exc
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc
        if vendor is None:
            raise VendorNotFoundError(f"Vendor '{vendor_id}' not found.", {"vendor_id": vendor_id})
        invalidate_vendor_cache()
        return {"data": vendor}

    @app.delete(f"{settings.api_prefix}/vendors/{{vendor_id}}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_vendor_endpoint(
        vendor_id: int,
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> Response:
        try:
            deleted = vendor_service.delete_vendor(session, vendor_id)
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc
        if not deleted:
            raise VendorNotFoundError(f"Vendor '{vendor_id}' not found.", {"vendor_id": vendor_id})
        invalidate_vendor_cache()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(f"{settings.api_prefix}/vendors/performance")
    def vendor_performance_endpoint(
        _: dict[str, Any] = Depends(get_current_user),
        session: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        cache_key = "vendors:performance"
        try:
            redis = redis_client.get_client()
            cached = redis.get(cache_key)
        except Exception as exc:
            logger.exception("redis.cache.read.failed", extra={"event": "redis.cache.read.failed"})
            raise AppError("Redis cache is unavailable.") from exc
        if cached:
            return {"data": vendor_service.decode_cache_payload(cached), "cache": "hit"}
        try:
            leaderboard = vendor_service.performance_leaderboard(session)
        except RuntimeError as exc:
            raise DatabaseOperationError(str(exc)) from exc
        try:
            redis.setex(cache_key, settings.cache_ttl_seconds, vendor_service.encode_cache_payload(leaderboard))
        except Exception as exc:
            logger.exception("redis.cache.write.failed", extra={"event": "redis.cache.write.failed"})
            raise AppError("Redis cache is unavailable.") from exc
        return {"data": leaderboard, "cache": "miss"}

    @app.get(f"{settings.api_prefix}/models/{{model_name}}/versions")
    def model_versions_endpoint(
        model_name: str,
        _: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        versions = model_registry.list_versions(model_name)
        if not versions:
            raise ModelNotFoundError(f"No versions found for model '{model_name}'.", {"model_name": model_name})
        return {"model_name": model_name, "versions": versions}

    @app.post(f"{settings.api_prefix}/models/{{model_name}}/predict")
    def model_predict_endpoint(
        model_name: str,
        payload: ModelPredictRequest,
        _: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        try:
            loaded = model_registry.load_latest_model(model_name)
        except ModelRegistryError as exc:
            raise ModelNotFoundError(str(exc), {"model_name": model_name}) from exc
        try:
            prediction = _predict(loaded, payload)
        except Exception as exc:
            raise AppError("Model inference failed.", {"model_name": model_name}) from exc
        return {
            "model_name": model_name,
            "version": loaded.version,
            "prediction": prediction,
        }

    @app.get("/health")
    def health(response: Response) -> dict[str, Any]:
        db_ok = False
        redis_ok = False
        try:
            db_ok = database_ready()
        except Exception:
            logger.exception("healthcheck.database.failed", extra={"event": "healthcheck.database.failed"})
        try:
            redis_ok = redis_client.ping()
        except Exception:
            logger.exception("healthcheck.redis.failed", extra={"event": "healthcheck.redis.failed"})
        if not (db_ok and redis_ok):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ok" if db_ok and redis_ok else "degraded",
            "database": "connected" if db_ok else "disconnected",
            "redis": "connected" if redis_ok else "disconnected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/metrics")
    def metrics_endpoint() -> dict[str, Any]:
        health_status = {"database": "connected", "redis": "connected"}
        try:
            database_ready()
        except Exception:
            health_status["database"] = "disconnected"
        try:
            redis_client.ping()
        except Exception:
            health_status["redis"] = "disconnected"
        snapshot = metrics.snapshot()
        snapshot["status"] = health_status
        return snapshot

    return app


def _predict(loaded_model: LoadedModel, payload: ModelPredictRequest) -> str:
    features = loaded_model.metadata["features"]
    frame = pd.DataFrame([payload.model_dump()], columns=features)
    return str(loaded_model.model.predict(frame)[0])


def _create_token(user_id: str, username: str, role: str, token_type: str) -> str:
    issued_at = datetime.now(timezone.utc)
    if token_type == "access":
        expires_at = issued_at + timedelta(minutes=settings.access_token_expire_minutes)
    else:
        expires_at = issued_at + timedelta(minutes=settings.refresh_token_expire_minutes)
    payload = {
        "user_id": user_id,
        "sub": username,
        "role": role,
        "type": token_type,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthenticationError("Invalid or expired token.") from exc


def _extract_user_id_from_headers(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        return str(_decode_token(token).get("user_id"))
    except AuthenticationError:
        return None


def _enforce_rate_limit(request: Request) -> None:
    if request.url.path in {"/health", "/metrics", "/"}:
        return
    client_ip = request.client.host if request.client else "unknown"
    redis = redis_client.get_client()
    key = f"rate_limit:{client_ip}"
    pipeline = redis.pipeline()
    pipeline.incr(key)
    pipeline.expire(key, settings.rate_limit_window_seconds, nx=True)
    current_requests, _ = pipeline.execute()
    if int(current_requests) > settings.rate_limit_per_window:
        raise RateLimitExceededError(
            "Rate limit exceeded.",
            {
                "limit": settings.rate_limit_per_window,
                "window_seconds": settings.rate_limit_window_seconds,
            },
        )


def invalidate_vendor_cache() -> None:
    try:
        redis_client.get_client().delete("vendors:performance")
    except Exception:
        logger.exception("redis.cache.invalidate.failed", extra={"event": "redis.cache.invalidate.failed"})


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, Any]:
    payload = _decode_token(token)
    if payload.get("type") != "access":
        raise AuthenticationError("Invalid access token.")
    return {
        "user_id": str(payload["user_id"]),
        "username": str(payload["sub"]),
        "role": str(payload["role"]),
    }


# This is the only line that should be at module level after all definitions
app = create_app()
