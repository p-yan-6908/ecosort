"""Waste Classifiers with different backbones."""

import torch
import torch.nn as nn
from torchvision import models

from ecosort.models.layers import ClassifierHead, ClassifierHeadWithSE, ClassifierHeadWithECA


class WasteClassifier(nn.Module):
    """MobileNetV3 based waste classifier."""

    def __init__(
        self, num_classes: int = 6, dropout: float = 0.2, pretrained: bool = True,
        head_type: str = "default", backbone: str = "mobilenet_v3_small"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        # Select backbone
        if backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
        elif backbone == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
        else:  # mobilenet_v3_small
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features

        # Select classifier head
        if head_type == "se":
            head = ClassifierHeadWithSE(in_features, num_classes, dropout)
        elif head_type == "eca":
            head = ClassifierHeadWithECA(in_features, num_classes, dropout)
        else:
            head = ClassifierHead(in_features, num_classes, dropout)
        
        # Replace classifier
        if backbone == "efficientnet_v2_s":
            self.backbone.classifier[1] = head
        else:
            self.backbone.classifier = head

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
            return self.backbone.parameters()
        return self.backbone.classifier.parameters()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, num_classes: int = 6, device: str = "cpu"):
        model = cls(num_classes=num_classes, pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
