#!/usr/bin/env python3
"""Comprehensive model evaluation with per-class metrics and confusion matrix."""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from ecosort.models.classifier import WasteClassifier
from ecosort.data.dataset import WasteDataset
from ecosort.data.transforms import get_val_transforms
from ecosort.constants import CATEGORY_ID_TO_NAME, ONTARIO_CATEGORIES

# Config
NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHECKPOINT_PATH = Path("models/checkpoints/best_model.pth")
DATA_DIR = Path("data/processed/ontario")
OUTPUT_DIR = Path("docs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [CATEGORY_ID_TO_NAME[i] for i in range(NUM_CLASSES)]
DISPLAY_NAMES = {c.name: c.display for c in ONTARIO_CATEGORIES}


def load_model():
    """Load the trained model from checkpoint."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = WasteClassifier(
        num_classes=NUM_CLASSES,
        head_type="eca",
        backbone="mobilenet_v3_small",
        pretrained=False,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def evaluate_dataset(model, dataloader, device):
    """Run inference on a dataset and collect predictions."""
    all_preds = []
    all_labels = []
    all_probs = []
    total_inference_time = 0
    num_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            start = time.perf_counter()
            outputs = model(images)
            end = time.perf_counter()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            total_inference_time += (end - start) * 1000
            num_images += len(labels)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        total_inference_time,
        num_images,
    )


def generate_confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    """Generate and save confusion matrix visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=[DISPLAY_NAMES.get(n, n) for n in class_names],
            yticklabels=[DISPLAY_NAMES.get(n, n) for n in class_names],
            title="EcoSort Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )

        fig.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved to {output_path}")
    except ImportError:
        print("  matplotlib not available, skipping confusion matrix plot")
        # Save raw data instead
        cm = confusion_matrix(y_true, y_pred)
        np.savetxt(str(output_path).replace(".png", ".csv"), cm, delimiter=",", fmt="%d")
        print(f"  Confusion matrix data saved to {str(output_path).replace('.png', '.csv')}")


def generate_report(y_true, y_pred, y_probs, inference_time_ms, num_images):
    """Generate comprehensive evaluation report."""
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=[DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]) for i in range(NUM_CLASSES)],
        digits=4,
        output_dict=True,
    )

    # Overall metrics
    accuracy = (y_true == y_pred).mean()
    avg_inference = inference_time_ms / max(num_images, 1)

    # Per-class accuracy
    per_class_acc = {}
    for i in range(NUM_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[i]] = float((y_pred[mask] == i).mean())
        else:
            per_class_acc[CLASS_NAMES[i]] = 0.0

    # Top-2 and top-3 accuracy
    top2 = 0
    top3 = 0
    for i in range(len(y_true)):
        top_preds = np.argsort(y_probs[i])[::-1]
        if y_true[i] in top_preds[:2]:
            top2 += 1
        if y_true[i] in top_preds[:3]:
            top3 += 1
    top2_acc = top2 / len(y_true)
    top3_acc = top3 / len(y_true)

    # Build report
    result = {
        "model": {
            "architecture": "MobileNetV3-Small + ECA",
            "checkpoint": str(CHECKPOINT_PATH),
            "num_parameters": sum(
                p.numel()
                for p in WasteClassifier(num_classes=NUM_CLASSES, head_type="eca", pretrained=False).parameters()
            ),
        },
        "overall": {
            "accuracy": round(float(accuracy), 4),
            "top2_accuracy": round(float(top2_acc), 4),
            "top3_accuracy": round(float(top3_acc), 4),
            "total_images": num_images,
            "avg_inference_ms": round(avg_inference, 2),
            "total_inference_ms": round(inference_time_ms, 1),
        },
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return result


def main():
    print("=" * 70)
    print("ECOSORT MODEL EVALUATION")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, device = load_model()
    print(f"  Device: {device}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Create dataloaders
    transform = get_val_transforms(IMAGE_SIZE)
    splits = {}
    for split_name in ["val", "test"]:
        split_dir = DATA_DIR / split_name
        if split_dir.exists():
            splits[split_name] = DataLoader(
                WasteDataset(DATA_DIR, split=split_name, transform=transform),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            print(f"  {split_name}: {len(splits[split_name].dataset)} images")

    # Evaluate each split
    all_results = {}
    for split_name, dataloader in splits.items():
        print(f"\n--- Evaluating {split_name} set ---")
        y_pred, y_true, y_probs, inf_time, n_images = evaluate_dataset(model, dataloader, device)

        result = generate_report(y_true, y_pred, y_probs, inf_time, n_images)
        all_results[split_name] = result

        print(f"  Accuracy: {result['overall']['accuracy']:.4f} ({result['overall']['accuracy']*100:.2f}%)")
        print(f"  Top-2 Accuracy: {result['overall']['top2_accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {result['overall']['top3_accuracy']:.4f}")
        print(f"  Avg inference: {result['overall']['avg_inference_ms']:.2f} ms/image")
        print(f"\n  Per-class accuracy:")
        for cls_name, acc in result["per_class_accuracy"].items():
            display = DISPLAY_NAMES.get(cls_name, cls_name)
            bar = "█" * int(acc * 30) + "░" * (30 - int(acc * 30))
            print(f"    {display:30s} {bar} {acc:.4f}")

        # Generate confusion matrix
        generate_confusion_matrix_plot(
            y_true, y_pred, CLASS_NAMES,
            OUTPUT_DIR / f"confusion_matrix_{split_name}.png",
        )

    # Save results
    output_file = OUTPUT_DIR / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for split_name, result in all_results.items():
        print(f"\n  {split_name.upper()} SET:")
        print(f"    Overall Accuracy: {result['overall']['accuracy']*100:.2f}%")
        print(f"    Top-2 Accuracy:   {result['overall']['top2_accuracy']*100:.2f}%")
        print(f"    Top-3 Accuracy:   {result['overall']['top3_accuracy']*100:.2f}%")
        print(f"    Avg Inference:    {result['overall']['avg_inference_ms']:.2f} ms")


if __name__ == "__main__":
    main()
