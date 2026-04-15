#!/usr/bin/env python3
"""Comprehensive benchmark script for autoresearch."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ecosort.models.classifier import WasteClassifier

BATCH_SIZE = 32
NUM_CLASSES = 6
IMAGE_SIZE = 224
DATA_DIR = Path("/Users/pyan/ecosort/data/processed/ontario")

class SimpleWasteDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.samples = []
        self.root = Path(root_dir) / split
        classes = ['blue_bin', 'green_bin', 'garbage', 'hazardous', 'e_waste', 'yard_waste']
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for p in (self.root / c).glob("*.jpg") if (self.root / c).exists() else []:
                self.samples.append((p, self.class_to_idx[c]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = Image.open(img).convert('RGB')
        return self.transform(img) if self.transform else img, label

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    bx1, by1 = np.random.randint(0, IMAGE_SIZE), np.random.randint(0, IMAGE_SIZE)
    bw, bh = int(IMAGE_SIZE * np.sqrt(1-lam)), int(IMAGE_SIZE * np.sqrt(1-lam))
    bx2, by2 = min(bx1 + bw, IMAGE_SIZE), min(by1 + bh, IMAGE_SIZE)
    index = torch.randperm(x.size(0)).to(x.device)
    x_cut = x.clone()
    x_cut[:, :, bx1:bx2, by1:by2] = x[index, :, bx1:bx2, by1:by2]
    lam = 1 - (bx2-bx1)*(by2-by1)/(IMAGE_SIZE*IMAGE_SIZE)
    return x_cut, y, y[index], lam

def get_loaders(augment_level='normal'):
    if augment_level == 'strong':
        train_t = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        train_t = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    val_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return (DataLoader(SimpleWasteDataset(DATA_DIR,"train",train_t), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(SimpleWasteDataset(DATA_DIR,"val",val_t), batch_size=BATCH_SIZE))

def train_epoch(model, loader, opt, device, args):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        
        if args.mixup:
            img, y_a, y_b, lam = mixup_data(img, lbl)
            opt.zero_grad()
            loss = lam * criterion(model(img), y_a) + (1-lam) * criterion(model(img), y_b)
        elif args.cutmix:
            img, y_a, y_b, lam = cutmix_data(img, lbl)
            opt.zero_grad()
            loss = lam * criterion(model(img), y_a) + (1-lam) * criterion(model(img), y_b)
        else:
            opt.zero_grad()
            loss = criterion(model(img), lbl)
        
        loss.backward()
        opt.step()

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, lbl in loader:
            out = model(img.to(device))
            correct += (out.argmax(1).cpu() == lbl).sum().item()
            total += len(lbl)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str, default="default", choices=["default","se","eca"])
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--augment", type=str, default="normal", choices=["normal","strong"])
    parser.add_argument("--time_budget", type=int, default=60)
    parser.add_argument("--lr1", type=float, default=0.01)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--experiment", type=str, default="")
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_loader, val_loader = get_loaders(args.augment)
    model = WasteClassifier(
        num_classes=NUM_CLASSES, 
        head_type=args.head, 
        backbone=args.backbone,
        dropout=args.dropout
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    start = time.time()
    best_acc = 0
    best_epoch = 0
    
    model.freeze_backbone()
    opt = torch.optim.Adam(model.get_trainable_params(False), lr=args.lr1, weight_decay=args.weight_decay)
    
    epoch = 0
    while time.time() - start < args.time_budget:
        epoch += 1
        train_epoch(model, train_loader, opt, device, args)
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        
        if epoch == 5:
            model.unfreeze_backbone()
            opt = torch.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
        
        if time.time() - start >= args.time_budget: break
    
    final_acc = evaluate(model, val_loader, device)
    
    print(f"experiment: {args.experiment}")
    print(f"val_accuracy: {final_acc:.6f}")
    print(f"best_val_accuracy: {best_acc:.6f}")
    print(f"best_epoch: {best_epoch}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"epochs: {epoch}")
    print(f"training_seconds: {time.time() - start:.1f}")

if __name__ == "__main__": main()
