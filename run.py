#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent


def _streamlit_command(host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]


def _api_command() -> list[str]:
    return [sys.executable, str(BASE_DIR / "run_api.py")]


def _wait_for_api(base_url: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return True
        except URLError:
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return False


def run_api() -> int:
    return subprocess.call(_api_command(), cwd=BASE_DIR)


def run_frontend(host: str, port: int) -> int:
    return subprocess.call(_streamlit_command(host, port), cwd=BASE_DIR)


def run_local(host: str, frontend_port: int) -> int:
    env = os.environ.copy()
    api_base_url = env.get("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"Starting API server at {api_base_url}...")
    api_process = subprocess.Popen(_api_command(), cwd=BASE_DIR, env=env)
    try:
        print("Waiting for API health check...")
        if not _wait_for_api(api_base_url):
            print("API did not become ready in time. Stop here and run `python run_api.py` to inspect backend logs.")
            return 1
        print(f"API is ready. Starting Streamlit frontend at http://{host}:{frontend_port} ...")
        return subprocess.call(_streamlit_command(host, frontend_port), cwd=BASE_DIR, env=env)
    finally:
        if api_process.poll() is None:
            print("Stopping API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vendor Insight 360 local launcher")
    parser.add_argument("--mode", choices=["local", "api", "frontend"], default="local")
    parser.add_argument("--host", default="127.0.0.1", help="Frontend bind host")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Frontend port")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "api":
        return run_api()
    if args.mode == "frontend":
        return run_frontend(args.host, args.frontend_port)
    return run_local(args.host, args.frontend_port)


if __name__ == "__main__":
    raise SystemExit(main())
