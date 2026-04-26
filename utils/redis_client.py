from __future__ import annotations

import time
from threading import Lock

import redis

from config.settings import get_settings


settings = get_settings()


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
            self._client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
        return self._client

    def ping(self, max_attempts: int = 10, delay_seconds: int = 1) -> bool:
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
