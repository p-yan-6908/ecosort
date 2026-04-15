"""Pytest configuration and fixtures for EcoSort tests."""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import numpy as np
import io


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing file uploads."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for model input."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing API."""
    predictor = MagicMock()
    predictor.predict.return_value = {
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
    predictor.predict_top_k.return_value = [
        {"class_name": "blue_bin", "display_name": "Blue Bin", "confidence": 0.95, "icon": "♻️"},
        {"class_name": "green_bin", "display_name": "Green Bin", "confidence": 0.03, "icon": "🌿"},
        {"class_name": "garbage", "display_name": "Garbage", "confidence": 0.01, "icon": "🗑️"},
    ]
    return predictor


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure for testing."""
    data_dir = tmp_path / "data"
    
    for split in ["train", "val", "test"]:
        for class_name in ["blue_bin", "green_bin", "garbage"]:
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = Image.new('RGB', (224, 224), color='red')
            img.save(class_dir / "test_image.jpg", 'JPEG')
    
    return data_dir
