"""Configuration Validation for EcoSort.

This module provides validation functions for configuration files
to ensure all required settings are present and valid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    
    field: str
    message: str
    severity: str = "error"  # error, warning


class ConfigValidator:
    """Validates EcoSort configuration files.
    
    Example:
        >>> validator = ConfigValidator()
        >>> errors = validator.validate("config.yaml")
        >>> if errors:
        ...     print("Validation failed")
    """
    
    REQUIRED_MODEL_FIELDS = ["architecture", "num_classes", "dropout", "pretrained"]
    REQUIRED_TRAINING_FIELDS = ["batch_size"]
    REQUIRED_DATA_FIELDS = ["image_size", "num_workers"]
    REQUIRED_PATH_FIELDS = ["checkpoints_dir"]
    
    VALID_ARCHITECTURES = ["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "resnet50"]
    
    def __init__(self):
        """Initialize the validator."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def validate(self, config_path: str) -> List[ValidationError]:
        """Validate a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
        
        Returns:
            List of validation errors. Empty list if valid.
        
        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        self.errors = []
        self.warnings = []
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                return [ValidationError("yaml", f"Invalid YAML syntax: {e}")]
        
        # Validate model section
        self._validate_section(data, "model", self.REQUIRED_MODEL_FIELDS)
        if "model" in data:
            self._validate_model_config(data["model"])
        
        # Validate training section
        self._validate_section(data, "training", self.REQUIRED_TRAINING_FIELDS)
        if "training" in data:
            self._validate_training_config(data["training"])
        
        # Validate data section
        self._validate_section(data, "data", self.REQUIRED_DATA_FIELDS)
        if "data" in data:
            self._validate_data_config(data["data"])
        
        # Validate paths section
        self._validate_section(data, "paths", self.REQUIRED_PATH_FIELDS)
        
        return self.errors
    
    def _validate_section(self, data: dict, section: str, required_fields: List[str]) -> None:
        """Validate a configuration section exists and has required fields."""
        if section not in data:
            self.errors.append(ValidationError(section, f"Missing required section: {section}"))
            return
        
        for field in required_fields:
            if field not in data[section]:
                self.errors.append(ValidationError(
                    f"{section}.{field}",
                    f"Missing required field: {section}.{field}"
                ))
    
    def _validate_model_config(self, model: dict) -> None:
        """Validate model configuration values."""
        # Check architecture
        if "architecture" in model:
            arch = model["architecture"]
            if arch not in self.VALID_ARCHITECTURES:
                self.warnings.append(ValidationError(
                    "model.architecture",
                    f"Unknown architecture: {arch}. Valid: {self.VALID_ARCHITECTURES}",
                    "warning"
                ))
        
        # Check num_classes
        if "num_classes" in model:
            num = model["num_classes"]
            if not isinstance(num, int) or num < 2:
                self.errors.append(ValidationError(
                    "model.num_classes",
                    f"num_classes must be an integer >= 2, got {num}"
                ))
        
        # Check dropout
        if "dropout" in model:
            dropout = model["dropout"]
            if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
                self.errors.append(ValidationError(
                    "model.dropout",
                    f"dropout must be a number between 0 and 1, got {dropout}"
                ))
    
    def _validate_training_config(self, training: dict) -> None:
        """Validate training configuration values."""
        # Check batch_size
        if "batch_size" in training:
            bs = training["batch_size"]
            if not isinstance(bs, int) or bs < 1:
                self.errors.append(ValidationError(
                    "training.batch_size",
                    f"batch_size must be a positive integer, got {bs}"
                ))
        
        # Check epochs
        if "epochs" in training:
            epochs = training["epochs"]
            if not isinstance(epochs, int) or epochs < 1:
                self.errors.append(ValidationError(
                    "training.epochs",
                    f"epochs must be a positive integer, got {epochs}"
                ))
    
    def _validate_data_config(self, data: dict) -> None:
        """Validate data configuration values."""
        # Check image_size
        if "image_size" in data:
            size = data["image_size"]
            if not isinstance(size, int) or size < 32:
                self.errors.append(ValidationError(
                    "data.image_size",
                    f"image_size must be an integer >= 32, got {size}"
                ))
        
        # Check num_workers
        if "num_workers" in data:
            workers = data["num_workers"]
            if not isinstance(workers, int) or workers < 0:
                self.errors.append(ValidationError(
                    "data.num_workers",
                    f"num_workers must be a non-negative integer, got {workers}"
                ))
    
    def get_warnings(self) -> List[ValidationError]:
        """Get validation warnings.
        
        Returns:
            List of warnings (non-fatal validation issues).
        """
        return self.warnings


def validate_config(config_path: str) -> tuple[bool, List[str]]:
    """Validate a configuration file and return status.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        Tuple of (is_valid, error_messages).
    
    Example:
        >>> is_valid, errors = validate_config("config.yaml")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """
    validator = ConfigValidator()
    try:
        errors = validator.validate(config_path)
        messages = [f"{e.field}: {e.message}" for e in errors]
        return len(errors) == 0, messages
    except FileNotFoundError as e:
        return False, [str(e)]
