"""
Basic API tests for the medicinal plant detector.
Mocks are set up globally in conftest.py.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock


def test_health_endpoint():
    """Test health endpoint returns success"""
    # Import after mocking
    from main import app
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"


def test_root_endpoint():
    """Test root endpoint serves HTML"""
    from main import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200


def test_identify_missing_file():
    """Test identify endpoint rejects request without file"""
    from main import app
    client = TestClient(app)
    
    response = client.post("/identify")
    assert response.status_code == 422  # FastAPI validation error


def test_identify_invalid_file_type():
    """Test identify endpoint rejects non-image files"""
    from main import app
    client = TestClient(app)
    
    response = client.post(
        "/identify",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
