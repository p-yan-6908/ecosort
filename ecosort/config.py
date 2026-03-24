from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    model_architecture: str
    num_classes: int
    dropout: float
    pretrained: bool
    batch_size: int
    image_size: int
    num_workers: int
    checkpoints_dir: str

    @classmethod
    def from_yaml(cls, path: str):
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
