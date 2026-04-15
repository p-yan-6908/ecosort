"""Custom Layers for Waste Classification."""

import torch
import torch.nn as nn


class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention module."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class ECAAttention(nn.Module):
    """Efficient Channel Attention."""
    
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.unsqueeze(-1)  # (B, C, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = torch.sigmoid(y).squeeze(-1)
        return x * y


class ClassifierHead(nn.Module):
    """Standard classifier head."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = in_features // 2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ClassifierHeadWithSE(nn.Module):
    """Classifier head with SE attention."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        hidden = in_features // 2
        self.bn = nn.BatchNorm1d(in_features)
        self.se = SEAttention(in_features, reduction=4)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.se(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class ClassifierHeadWithECA(nn.Module):
    """Classifier head with ECA attention."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        hidden = in_features // 2
        self.bn = nn.BatchNorm1d(in_features)
        self.eca = ECAAttention(in_features, k_size=3)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.eca(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)
