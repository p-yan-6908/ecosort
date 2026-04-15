"""Custom Classifier Head for Waste Classification.

This module provides a custom classifier head with batch normalization
and dropout for better generalization on waste classification tasks.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Custom classifier head with batch normalization and dropout.
    
    This head uses a two-layer architecture with BatchNorm, Hardswish activation,
    and dropout for improved generalization.
    
    Attributes:
        classifier: Sequential module containing the classification layers.
    
    Example:
        >>> head = ClassifierHead(in_features=576, num_classes=6, dropout=0.2)
        >>> x = torch.randn(32, 576)
        >>> output = head(x)  # Shape: (32, 6)
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2) -> None:
        """Initialize the classifier head.
        
        Args:
            in_features: Number of input features from the backbone.
            num_classes: Number of output classes for classification.
            dropout: Dropout probability for regularization. Defaults to 0.2.
        """
        super().__init__()

        hidden_features = in_features // 2

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier head.
        
        Args:
            x: Input tensor of shape (batch_size, in_features).
        
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        return self.classifier(x)
