#!/usr/bin/env python3
"""Full training script for the best model configuration."""
import sys, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ecosort.models.classifier import WasteClassifier

# Best configuration from autoresearch
NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
HEAD_TYPE = "eca"
BACKBONE = "mobilenet_v3_small"
DATA_DIR = Path("/Users/pyan/ecosort/data/processed/ontario")
CHECKPOINT_DIR = Path("/Users/pyan/ecosort/models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class WasteDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.samples = []
        self.root = Path(root_dir) / split
        classes = ['blue_bin', 'green_bin', 'garbage', 'hazardous', 'e_waste', 'yard_waste']
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            class_dir = self.root / c
            if class_dir.exists():
                for p in class_dir.iterdir():
                    if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((p, self.class_to_idx[c]))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img) if self.transform else img, label


def get_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = WasteDataset(DATA_DIR, "train", transform=train_transform)
    val_ds = WasteDataset(DATA_DIR, "val", transform=val_transform)
    test_ds = WasteDataset(DATA_DIR, "test", transform=val_transform)
    
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    )


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / len(loader), correct / total


def main():
    print("=" * 70)
    print("ECOSORT FULL MODEL TRAINING")
    print("Best Configuration: MobileNetV3-Small + ECA + Label Smoothing 0.1")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    
    # Create model
    model = WasteClassifier(
        num_classes=NUM_CLASSES,
        head_type=HEAD_TYPE,
        backbone=BACKBONE,
        pretrained=True
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Phase 1: Train classifier head (frozen backbone)
    print("\n" + "=" * 70)
    print("PHASE 1: Training classifier head (frozen backbone)")
    print("=" * 70)
    
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.get_trainable_params(backbone=False), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_DIR / "phase1_best.pth")
    
    print(f"\nPhase 1 Best: val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    
    # Phase 2: Fine-tune entire network
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-tuning entire network")
    print("=" * 70)
    
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "phase1_best.pth", weights_only=True))
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_val_acc_phase2 = 0
    best_epoch_phase2 = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(1, 31):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            best_epoch_phase2 = epoch
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\nPhase 2 Best: val_acc={best_val_acc_phase2:.4f} at epoch {best_epoch_phase2}")
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save final model info
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc_phase2:.4f} ({best_val_acc_phase2*100:.2f}%)")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")
    
    return test_acc


if __name__ == "__main__":
    main()
