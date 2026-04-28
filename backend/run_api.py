#!/usr/bin/env python3
from __future__ import annotations

import uvicorn

from api.main import app
from config.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
