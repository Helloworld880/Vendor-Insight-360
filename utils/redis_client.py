from __future__ import annotations

import threading
import time
from threading import Lock

import redis

from config.settings import get_settings


settings = get_settings()


class InMemoryPipeline:
    def __init__(self, client: "InMemoryRedis") -> None:
        self.client = client
        self.operations: list[tuple[str, tuple, dict]] = []

    def incr(self, key: str) -> "InMemoryPipeline":
        self.operations.append(("incr", (key,), {}))
        return self

    def expire(self, key: str, seconds: int, nx: bool = False) -> "InMemoryPipeline":
        self.operations.append(("expire", (key, seconds), {"nx": nx}))
        return self

    def execute(self) -> list[object]:
        results = []
        for operation, args, kwargs in self.operations:
            results.append(getattr(self.client, operation)(*args, **kwargs))
        self.operations.clear()
        return results


class InMemoryRedis:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, object] = {}
        self._expiry: dict[str, float] = {}

    def _purge_if_expired(self, key: str) -> None:
        expires_at = self._expiry.get(key)
        if expires_at is not None and expires_at <= time.time():
            self._store.pop(key, None)
            self._expiry.pop(key, None)

    def get(self, key: str) -> object | None:
        with self._lock:
            self._purge_if_expired(key)
            return self._store.get(key)

    def setex(self, key: str, seconds: int, value: object) -> bool:
        with self._lock:
            self._store[key] = value
            self._expiry[key] = time.time() + seconds
            return True

    def delete(self, key: str) -> int:
        with self._lock:
            self._purge_if_expired(key)
            removed = 1 if key in self._store else 0
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return removed

    def incr(self, key: str) -> int:
        with self._lock:
            self._purge_if_expired(key)
            value = int(self._store.get(key, 0)) + 1
            self._store[key] = value
            return value

    def expire(self, key: str, seconds: int, nx: bool = False) -> bool:
        with self._lock:
            self._purge_if_expired(key)
            if key not in self._store:
                return False
            if nx and key in self._expiry:
                return False
            self._expiry[key] = time.time() + seconds
            return True

    def flushdb(self) -> bool:
        with self._lock:
            self._store.clear()
            self._expiry.clear()
            return True

    def ping(self) -> bool:
        return True

    def pipeline(self) -> InMemoryPipeline:
        return InMemoryPipeline(self)


class RedisClient:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
        return cls._instance

    def get_client(self) -> redis.Redis:
        if self._client is None:
            if settings.redis_url.startswith("memory://"):
                self._client = InMemoryRedis()
            else:
                self._client = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    health_check_interval=30,
                )
        return self._client

    def ping(self, max_attempts: int = 10, delay_seconds: int = 1) -> bool:
        if settings.redis_url.startswith("memory://"):
            return True
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return bool(self.get_client().ping())
            except redis.RedisError as exc:
                last_error = exc
                self._client = None
                if attempt == max_attempts:
                    break
                time.sleep(delay_seconds)
        raise RuntimeError("Redis did not become ready in time.") from last_error


redis_client = RedisClient()
