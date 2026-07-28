"""
Basic API tests for the medicinal plant detector.
Mocks heavy dependencies to keep tests fast.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import MagicMock, patch

# Set up test environment before importing main
os.environ['GROQ_API_KEY'] = 'test_mock_key'

# Mock heavy dependencies
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['groq'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['faiss'] = MagicMock()


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
