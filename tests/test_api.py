"""Tests for EcoSort API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

from ecosort.api.main import app
from ecosort.api.dependencies import _predictor


@pytest.fixture
def mock_predictor():
    """Create mock predictor for dependency injection."""
    mock = MagicMock()
    mock.predict.return_value = {
        "class_id": 0,
        "class_name": "blue_bin",
        "display_name": "Blue Bin (Recyclables)",
        "confidence": 0.95,
        "icon": "♻️",
        "color": "#2563EB",
        "description": "Cardboard, paper, plastic, metal, glass",
        "all_probabilities": {
            "blue_bin": 0.95,
            "green_bin": 0.03,
            "garbage": 0.01,
            "hazardous": 0.005,
            "e_waste": 0.003,
            "yard_waste": 0.002,
        },
    }
    mock.predict_top_k.return_value = [
        {"class_name": "blue_bin", "display_name": "Blue Bin", "confidence": 0.95, "icon": "♻️"},
        {"class_name": "green_bin", "display_name": "Green Bin", "confidence": 0.03, "icon": "🌿"},
        {"class_name": "garbage", "display_name": "Garbage", "confidence": 0.01, "icon": "🗑️"},
    ]
    return mock


@pytest.fixture
def client(mock_predictor):
    """Create a test client with mocked predictor."""
    with patch('ecosort.api.dependencies._predictor', mock_predictor):
        with patch('ecosort.api.dependencies.get_predictor', return_value=mock_predictor):
            yield TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data


class TestClassesEndpoint:
    """Tests for classes listing endpoint."""

    def test_classes_returns_categories(self, client):
        """Test classes endpoint returns all categories."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) == 6

    def test_classes_have_required_fields(self, client):
        """Test each category has required fields."""
        response = client.get("/classes")
        data = response.json()
        for cat in data["categories"]:
            assert "id" in cat
            assert "name" in cat
            assert "display_name" in cat
            assert "color" in cat
            assert "icon" in cat
            assert "description" in cat


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def create_image_file(self):
        """Create a valid image file for upload."""
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return ("test.jpg", img_bytes, "image/jpeg")

    def test_predict_returns_result(self, client, mock_predictor):
        """Test prediction endpoint returns result."""
        files = {"file": self.create_image_file()}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "class_name" in data
        assert "confidence" in data
        assert "display_name" in data

    def test_predict_rejects_invalid_content_type(self, client):
        """Test prediction rejects non-image files."""
        files = {"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_top_k_returns_results(self, client, mock_predictor):
        """Test top-k prediction endpoint."""
        files = {"file": self.create_image_file()}
        response = client.post("/predict/top-k?k=3", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) <= 3


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_message(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
