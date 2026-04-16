"""Tests for WastePredictor class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import torch
import numpy as np

from ecosort.inference.predictor import WastePredictor
from ecosort.models.classifier import WasteClassifier


class TestWastePredictor:
    """Tests for WastePredictor inference class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.eval.return_value = model
        model.return_value = torch.tensor([[-1.0, 2.0, 0.5, -0.5, 0.0, 1.0]])
        return model

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return Image.new('RGB', (224, 224), color='blue')

    def test_predictor_initialization(self, mock_model):
        """Test predictor can be created with valid model path."""
        with patch.object(WasteClassifier, 'from_checkpoint') as mock_load:
            mock_load.return_value = mock_model
            predictor = WastePredictor(Path("dummy.pth"))
            assert predictor is not None
            assert predictor.model is not None

    def test_predict_returns_dict(self, mock_model, sample_image):
        """Test predict returns a dictionary with expected keys."""
        with patch('ecosort.inference.predictor.WasteClassifier') as MockClassifier:
            MockClassifier.from_checkpoint.return_value = mock_model
            with patch('ecosort.inference.predictor.get_inference_transforms') as mock_transform:
                mock_transform.return_value = lambda image: {"image": torch.randn(3, 224, 224)}
                predictor = WastePredictor(Path("dummy.pth"))
                
                # Need to properly set up the mock
                predictor.model = mock_model
                predictor.categories = {i: MagicMock(
                    id=i,
                    name=f"cat_{i}",
                    display=f"Category {i}",
                    icon="♻️",
                    color="#000000",
                    description="Test category"
                ) for i in range(6)}
                
                result = predictor.predict(sample_image)
                
                assert isinstance(result, dict)
                assert "class_id" in result
                assert "class_name" in result
                assert "confidence" in result
                assert "all_probabilities" in result

    def test_predict_top_k_returns_list(self, mock_model, sample_image):
        """Test predict_top_k returns a list of predictions."""
        with patch('ecosort.inference.predictor.WasteClassifier') as MockClassifier:
            MockClassifier.from_checkpoint.return_value = mock_model
            with patch('ecosort.inference.predictor.get_inference_transforms') as mock_transform:
                mock_transform.return_value = lambda image: {"image": torch.randn(3, 224, 224)}
                predictor = WastePredictor(Path("dummy.pth"))
                
                predictor.model = mock_model
                predictor.categories = {i: MagicMock(
                    name=f"cat_{i}",
                    display=f"Category {i}",
                    icon="♻️"
                ) for i in range(6)}
                
                results = predictor.predict_top_k(sample_image, k=3)
                
                assert isinstance(results, list)
                assert len(results) <= 3

    def test_predict_confidence_range(self, mock_model, sample_image):
        """Test prediction confidence is between 0 and 1."""
        with patch('ecosort.inference.predictor.WasteClassifier') as MockClassifier:
            MockClassifier.from_checkpoint.return_value = mock_model
            with patch('ecosort.inference.predictor.get_inference_transforms') as mock_transform:
                mock_transform.return_value = lambda image: {"image": torch.randn(3, 224, 224)}
                predictor = WastePredictor(Path("dummy.pth"))
                
                predictor.model = mock_model
                predictor.categories = {i: MagicMock(
                    id=i,
                    name=f"cat_{i}",
                    display=f"Category {i}",
                    icon="♻️",
                    color="#000000",
                    description="Test"
                ) for i in range(6)}
                
                result = predictor.predict(sample_image)
                
                assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_converts_to_rgb(self, mock_model):
        """Test predictor converts grayscale images to RGB."""
        gray_image = Image.new('L', (224, 224))  # Grayscale
        
        with patch('ecosort.inference.predictor.WasteClassifier') as MockClassifier:
            MockClassifier.from_checkpoint.return_value = mock_model
            with patch('ecosort.inference.predictor.get_inference_transforms') as mock_transform:
                mock_transform.return_value = lambda image: {"image": torch.randn(3, 224, 224)}
                predictor = WastePredictor(Path("dummy.pth"))
                
                predictor.model = mock_model
                predictor.categories = {i: MagicMock(
                    id=i,
                    name=f"cat_{i}",
                    display=f"Category {i}",
                    icon="♻️",
                    color="#000000",
                    description="Test"
                ) for i in range(6)}
                
                result = predictor.predict(gray_image)
                assert result is not None
