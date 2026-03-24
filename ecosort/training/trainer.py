"""Two-Phase Training for Waste Classifier"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from ecosort.training.scheduler import get_cosine_schedule_with_warmup
from ecosort.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Two-phase training for waste classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        checkpoint_dir: Path = Path("models/checkpoints"),
        gradient_accum_steps: int = 2,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accum_steps = gradient_accum_steps
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_phase1(self, epochs: int, lr: float = 0.01):
        """Phase 1: Train classifier head with frozen backbone."""
        logger.info("Starting Phase 1: Training classifier head")
        self.model.freeze_backbone()
        optimizer = optim.AdamW(
            self.model.backbone.classifier.parameters(), lr=lr, weight_decay=0.01
        )
        return self._train_loop(epochs, optimizer, phase_name="phase1")

    def train_phase2(self, epochs: int, lr: float = 0.0001, warmup_epochs: int = 2):
        """Phase 2: Fine-tune entire network."""
        logger.info("Starting Phase 2: Fine-tuning entire network")
        self.model.unfreeze_backbone()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_epochs * len(self.train_loader),
            num_training_steps=epochs * len(self.train_loader),
        )
        return self._train_loop(
            epochs, optimizer, scheduler=scheduler, phase_name="phase2"
        )

    def _train_loop(
        self,
        epochs: int,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        phase_name: str = "train",
    ) -> Dict:
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / self.gradient_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler:
                        scheduler.step()

                train_loss += loss.item() * self.gradient_accum_steps
                pbar.set_postfix(
                    {"loss": f"{loss.item() * self.gradient_accum_steps:.4f}"}
                )

            avg_train_loss = train_loss / len(self.train_loader)
            val_loss, val_acc, val_f1 = self._validate()

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(phase_name, epoch, val_acc)

        return self.history

    def _validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds)
        return val_loss / len(self.val_loader), metrics["accuracy"], metrics["f1_macro"]

    def _save_checkpoint(self, phase: str, epoch: int, val_acc: float):
        checkpoint_path = self.checkpoint_dir / f"{phase}_best.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Saved best model to {checkpoint_path} (val_acc={val_acc:.4f})")
