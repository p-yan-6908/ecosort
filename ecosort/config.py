"""Configuration management for EcoSort.

This module provides a configuration dataclass that loads settings
from YAML files for easy experiment management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    """Configuration settings for EcoSort training and inference.
    
    This dataclass holds all configuration parameters for model architecture,
    training hyperparameters, data processing, and paths.
    
    Attributes:
        model_architecture: Name of the model architecture (e.g., 'mobilenet_v3_small').
        num_classes: Number of classification categories.
        dropout: Dropout rate for the classifier head.
        pretrained: Whether to use pretrained weights.
        batch_size: Batch size for training and inference.
        image_size: Input image size (assumes square images).
        num_workers: Number of data loading workers.
        checkpoints_dir: Directory to save model checkpoints.
    
    Example:
        >>> config = Config.from_yaml("config.yaml")
        >>> print(config.num_classes)
        3
    """

    model_architecture: str
    num_classes: int
    dropout: float
    pretrained: bool
    batch_size: int
    image_size: int
    num_workers: int
    checkpoints_dir: str

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
        
        Returns:
            Config instance with loaded settings.
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            KeyError: If required fields are missing from the YAML file.
        
        Example:
            >>> config = Config.from_yaml("config.yaml")
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model_architecture=data["model"]["architecture"],
            num_classes=data["model"]["num_classes"],
            dropout=data["model"]["dropout"],
            pretrained=data["model"]["pretrained"],
            batch_size=data["training"]["batch_size"],
            image_size=data["data"]["image_size"],
            num_workers=data["data"]["num_workers"],
            checkpoints_dir=data["paths"]["checkpoints_dir"],
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "model": {
                "architecture": self.model_architecture,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "pretrained": self.pretrained,
            },
            "training": {
                "batch_size": self.batch_size,
            },
            "data": {
                "image_size": self.image_size,
                "num_workers": self.num_workers,
            },
            "paths": {
                "checkpoints_dir": self.checkpoints_dir,
            },
        }
