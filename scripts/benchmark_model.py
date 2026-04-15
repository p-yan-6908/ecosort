#!/usr/bin/env python3
"""Benchmark script for model evaluation."""
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from ecosort.models.classifier import WasteClassifier

BATCH_SIZE = 32
NUM_CLASSES = 6
IMAGE_SIZE = 224
DATA_DIR = Path("/Users/pyan/ecosort/data/processed/ontario")
TIME_BUDGET = 60

class SimpleWasteDataset(Dataset):
    """Simple dataset for benchmarking."""
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.samples = []
        self.root = Path(root_dir) / split
        
        classes = ['blue_bin', 'green_bin', 'garbage', 'hazardous', 'e_waste', 'yard_waste']
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for class_name in classes:
            class_dir = self.root / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_loaders():
    train_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = SimpleWasteDataset(DATA_DIR, "train", transform=train_t)
    val_ds = SimpleWasteDataset(DATA_DIR, "val", transform=val_t)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    )

def train_epoch(model, loader, opt, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        opt.zero_grad()
        loss = criterion(model(img), lbl)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss += criterion(out, lbl).item()
            correct += (out.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
    return correct / total, loss / len(loader)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_loader, val_loader = get_loaders()
    model = WasteClassifier(num_classes=NUM_CLASSES, dropout=0.2, pretrained=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {num_params:,}")
    
    start = time.time()
    best_acc = 0
    
    model.freeze_backbone()
    opt = torch.optim.Adam(model.get_trainable_params(backbone=False), lr=0.01)
    
    epoch = 0
    while time.time() - start < TIME_BUDGET:
        epoch += 1
        train_loss = train_epoch(model, train_loader, opt, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        
        if epoch == 5:
            model.unfreeze_backbone()
            opt = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        if time.time() - start >= TIME_BUDGET:
            break
    
    elapsed = time.time() - start
    final_acc, final_loss = evaluate(model, val_loader, device)
    
    print("---")
    print(f"val_accuracy: {final_acc:.6f}")
    print(f"best_val_accuracy: {best_acc:.6f}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"epochs: {epoch}")

if __name__ == "__main__":
    main()
