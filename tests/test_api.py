"""
Basic API tests for the medicinal plant detector.
Mocks are set up globally in conftest.py.
"""

import pytest
from fastapi.testclient import TestClient


def get_client():
    """Import app after conftest mocks are applied."""
    from main import app
    return TestClient(app)


def test_health_endpoint():
    """Test health endpoint returns success."""
    client = get_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"


def test_root_endpoint():
    """Test root endpoint serves the frontend."""
    client = get_client()
    response = client.get("/")
    assert response.status_code == 200


def test_identify_missing_file():
    """Test identify endpoint rejects request without a file."""
    client = get_client()
    response = client.post("/identify")
    assert response.status_code == 422  # FastAPI validation error


def test_identify_invalid_file_type():
    """Test identify endpoint rejects non-image files."""
    client = get_client()
    response = client.post(
        "/identify",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
