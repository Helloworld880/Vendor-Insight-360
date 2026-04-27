import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
REQUEST_TIMEOUT_SECONDS = int(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SECONDS", "20"))

API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("STREAMLIT_API_BASE_URL")
    or "http://localhost:8000"
)

# REQUEST_TIMEOUT_SECONDS = int(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SECONDS", "20"))
# DEFAULT_API_BASE_URL = "http://localhost:8000"

# load_dotenv(dotenv_path=".env", override=True)

def _is_local_api_url(url: str) -> bool:
    lowered = url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None

def _get_api_base_url() -> str:
    configured = (
        os.getenv("API_BASE_URL")
        or os.getenv("STREAMLIT_API_BASE_URL")
        or _get_secret("API_BASE_URL")
        or _get_secret("STREAMLIT_API_BASE_URL")
        or "http://localhost:8000"
    )

    st.session_ state.api_base_url = configured.rstrip("/")
    return st.session_state.api_base_url
def _api_get(path: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{_get_api_base_url()}{path}", headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _login(username: str, password: str) -> tuple[str | None, str | None]:
    api_base_url = _get_api_base_url()
    try:
        response = requests.post(
            f"{api_base_url}/api/v1/login",
            data={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return None, f"Unable to reach the backend API at {api_base_url}: {exc}"

    if response.status_code != 200:
        return None, "Invalid credentials."
    return response.json().get("access_token"), None


def _logout() -> None:
    st.session_state.token = None
    st.rerun()


def _render_connection_banner() -> None:
    api_base_url = _get_api_base_url()
    st.caption(f"API base URL: `{api_base_url}`")
    if _is_local_api_url(api_base_url):
        st.warning(
            "This frontend is still pointed at a local API URL. On Streamlit Cloud, set "
            "`STREAMLIT_API_BASE_URL` or `API_BASE_URL` to your deployed backend URL."
        )


def _render_login() -> None:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        token, error_message = _login(username, password)
        if token:
            st.session_state.token = token
            st.success("Login successful.")
            st.rerun()
        st.error(error_message or "Invalid credentials or API unavailable.")
        st.info("Set `STREAMLIT_API_BASE_URL` in the app environment to your deployed backend URL.")


def _to_frame(payload: list[dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


def _render_vendors(token: str) -> None:
    st.subheader("Vendors")
    response = _api_get("/api/v1/vendors", token)
    frame = _to_frame(response.get("data", []))
    if frame.empty:
        st.info("No vendors returned by the API.")
        return
    st.dataframe(frame, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vendors", len(frame))
    if "performance_score" in frame.columns:
        col2.metric("Average Performance", f"{frame['performance_score'].mean():.1f}")
    if "risk_score" in frame.columns:
        col3.metric("Average Risk", f"{frame['risk_score'].mean():.1f}")


def _render_performance(token: str) -> None:
    st.subheader("Performance Leaderboard")
    response = _api_get("/api/v1/vendors/performance", token)
    frame = _to_frame(response.get("data", []))
    if frame.empty:
        st.info("No performance data returned by the API.")
        return
    st.dataframe(frame, use_container_width=True)
    if "rank" in frame.columns and "name" in frame.columns:
        best = frame.sort_values("rank").iloc[0]
        st.success(f"Top vendor: {best['name']} (rank {int(best['rank'])})")


def _render_model_versions(token: str) -> None:
    st.subheader("Model Versions")
    response = _api_get("/api/v1/models/vendor_risk/versions", token)
    frame = _to_frame(response.get("versions", []))
    if frame.empty:
        st.info("No model versions returned by the API.")
        return
    st.dataframe(frame, use_container_width=True)


def _render_health(token: str) -> None:
    st.subheader("Platform Health")
    response = requests.get(f"{_get_api_base_url()}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", payload.get("status", "unknown"))
    col2.metric("Database", payload.get("database", "unknown"))
    col3.metric("Redis", payload.get("redis", "unknown"))


def main() -> None:
    st.set_page_config(page_title="Vendor Insight 360", page_icon="📊", layout="wide")
    st.title("Vendor Insight 360")
    st.write("Production Streamlit frontend for the FastAPI backend.")
    _render_connection_banner()

    if "token" not in st.session_state:
        st.session_state.token = None

    if not st.session_state.token:
        _render_login()
        return

    with st.sidebar:
        st.success("Authenticated")
        if st.button("Logout", use_container_width=True):
            _logout()

    tabs = st.tabs(["Vendors", "Performance", "Models", "Health"])

    try:
        with tabs[0]:
            _render_vendors(st.session_state.token)
        with tabs[1]:
            _render_performance(st.session_state.token)
        with tabs[2]:
            _render_model_versions(st.session_state.token)
        with tabs[3]:
            _render_health(st.session_state.token)
    except requests.HTTPError as exc:
        st.error(f"API request failed: {exc}")
        if exc.response is not None:
            st.code(exc.response.text)
    except requests.RequestException as exc:
        st.error(f"Unable to reach the backend API: {exc}")


if __name__ == "__main__":
    main()
