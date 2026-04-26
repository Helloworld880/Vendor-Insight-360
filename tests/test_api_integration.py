from __future__ import annotations

from fastapi.testclient import TestClient


def login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/login",
        data={"username": "admin", "password": "StrongAdminPass123"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    return {"Authorization": f"Bearer {payload['access_token']}"}


def test_vendor_crud_and_cache(client: TestClient) -> None:
    headers = login(client)
    payload = {
        "name": "Atlas Components",
        "category": "electronics",
        "status": "active",
        "delivery_rate": 91,
        "quality_score": 88,
        "cost_efficiency": 84,
        "on_time_rate": 93,
        "cost_variance": 2,
        "reliability": 91,
        "performance_score": 89,
        "risk_score": 27,
    }
    create_response = client.post("/api/v1/vendors", json=payload, headers=headers)
    assert create_response.status_code == 201, create_response.text

    vendors_response = client.get("/api/v1/vendors", headers=headers)
    assert vendors_response.status_code == 200, vendors_response.text
    vendors = vendors_response.json()["data"]
    assert len(vendors) == 1
    assert vendors[0]["name"] == "Atlas Components"

    performance_first = client.get("/api/v1/vendors/performance", headers=headers)
    performance_second = client.get("/api/v1/vendors/performance", headers=headers)
    assert performance_first.status_code == 200
    assert performance_first.json()["cache"] == "miss"
    assert performance_second.status_code == 200
    assert performance_second.json()["cache"] == "hit"


def test_model_versions_and_prediction(client: TestClient) -> None:
    headers = login(client)
    versions_response = client.get("/api/v1/models/vendor_risk/versions", headers=headers)
    assert versions_response.status_code == 200, versions_response.text
    versions = versions_response.json()["versions"]
    assert versions
    assert versions[0]["accuracy"] > 0

    predict_response = client.post(
        "/api/v1/models/vendor_risk/predict",
        json={
            "delivery_rate": 95,
            "quality_score": 93,
            "cost_efficiency": 89,
            "on_time_rate": 96,
            "cost_variance": -2,
            "reliability": 94,
            "performance_score": 93,
        },
        headers=headers,
    )
    assert predict_response.status_code == 200, predict_response.text
    prediction = predict_response.json()["prediction"]
    assert prediction in {"low", "medium", "high"}
