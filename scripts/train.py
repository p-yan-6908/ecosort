#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import logging

from ecosort.config import Config
from ecosort.models.classifier import WasteClassifier
from ecosort.data.dataset import create_dataloaders
from ecosort.data.transforms import get_train_transforms, get_val_transforms
from ecosort.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train EcoSort waste classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data-dir", type=str, default="data/processed/ontario")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--phase", type=int, choices=[1, 2])
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    transform_train = get_train_transforms(config.image_size)
    transform_val = get_val_transforms(config.image_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        Path(args.data_dir),
        transform_train,
        transform_val,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    model = WasteClassifier(
        num_classes=config.num_classes,
        dropout=config.dropout,
        pretrained=config.pretrained,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    if args.phase is None or args.phase == 1:
        logger.info("=" * 50)
        logger.info("PHASE 1: Training classifier head (frozen backbone)")
        logger.info("=" * 50)
        trainer.train_phase1(epochs=10, lr=0.01)

    if args.phase is None or args.phase == 2:
        if args.phase == 2:
            phase1_path = Path(args.checkpoint_dir) / "phase1_best.pth"
            if phase1_path.exists():
                model.load_state_dict(torch.load(phase1_path, map_location=device))
                logger.info(f"Loaded Phase 1 model from {phase1_path}")

        logger.info("=" * 50)
        logger.info("PHASE 2: Fine-tuning entire network")
        logger.info("=" * 50)
        trainer.train_phase2(epochs=20, lr=0.0001, warmup_epochs=2)

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")


if __name__ == "__main__":
    main()
