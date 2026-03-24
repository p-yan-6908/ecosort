"""Custom Classifier Head"""

import torch.nn as nn


class ClassifierHead(nn.Module):
    """Custom classifier head with batch normalization and dropout."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        hidden_features = in_features // 2

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
