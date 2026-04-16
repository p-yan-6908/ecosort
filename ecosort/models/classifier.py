"""Waste Classifier Model Architecture."""

from pathlib import Path
from typing import Literal
import torch
import torch.nn as nn
from torchvision import models

from ecosort.models.layers import ClassifierHead, ClassifierHeadWithSE, ClassifierHeadWithECA


class WasteClassifier(nn.Module):
    """Waste classification model with configurable backbone and attention head."""

    def __init__(
        self,
        num_classes: int = 6,
        head_type: Literal["default", "se", "eca"] = "default",
        backbone: Literal["mobilenet_v3_small", "mobilenet_v3_large"] = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type
        self.backbone_name = backbone

        # Load backbone
        if backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
            backbone_features = 576
        elif backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)
            backbone_features = 960
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Create classifier head (will be assigned to backbone.classifier)
        if head_type == "default":
            classifier = ClassifierHead(backbone_features, num_classes, dropout)
        elif head_type == "se":
            classifier = ClassifierHeadWithSE(backbone_features, num_classes, dropout)
        elif head_type == "eca":
            classifier = ClassifierHeadWithECA(backbone_features, num_classes, dropout)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Replace classifier in backbone (so keys match: backbone.classifier.*)
        base_model.classifier = classifier
        self.backbone = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_trainable_params(self, backbone: bool = False):
        if backbone:
            return self.parameters()
        return self.backbone.classifier.parameters()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        num_classes: int = 6,
        head_type: str = "eca",
        backbone: str = "mobilenet_v3_small",
        device: str = "cpu"
    ):
        """Load model from checkpoint."""
        model = cls(
            num_classes=num_classes,
            head_type=head_type,
            backbone=backbone,
            pretrained=False
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
