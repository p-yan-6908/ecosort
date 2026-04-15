"""Tests for EcoSort model components."""

import pytest
import torch

from ecosort.models.classifier import WasteClassifier
from ecosort.models.layers import ClassifierHead


class TestClassifierHead:
    """Tests for ClassifierHead module."""

    def test_head_creation(self):
        """Test basic head creation."""
        head = ClassifierHead(in_features=576, num_classes=6, dropout=0.2)
        assert head is not None

    def test_head_forward(self):
        """Test forward pass through head."""
        head = ClassifierHead(in_features=576, num_classes=6, dropout=0.2)
        head.eval()
        x = torch.randn(2, 576)
        output = head(x)
        assert output.shape == (2, 6)

    def test_head_output_range(self):
        """Test head produces valid logits."""
        head = ClassifierHead(in_features=576, num_classes=6, dropout=0.2)
        head.eval()
        x = torch.randn(1, 576)
        output = head(x)
        assert torch.isfinite(output).all()


class TestWasteClassifier:
    """Tests for WasteClassifier model."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        assert model is not None
        assert model.num_classes == 6

    def test_model_forward(self):
        """Test forward pass."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 6)

    def test_model_batch_forward(self):
        """Test forward pass with batch."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, 6)

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.features.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self):
        """Test backbone unfreezing."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.features.parameters():
            assert param.requires_grad

    def test_model_output_valid(self):
        """Test that model output is valid."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total_params < 5_000_000

    def test_from_checkpoint_raises_on_invalid(self):
        """Test that from_checkpoint raises on invalid path."""
        with pytest.raises(FileNotFoundError):
            WasteClassifier.from_checkpoint("nonexistent_checkpoint.pth")

    def test_eval_deterministic(self):
        """Test model is deterministic in eval mode."""
        model = WasteClassifier(num_classes=6, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        assert torch.allclose(output1, output2)
