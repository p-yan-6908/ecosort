#!/usr/bin/env python3
from pathlib import Path
import torch

from ecosort.models.classifier import WasteClassifier


def export_torchscript(model, output_path):
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)
    traced.save(output_path)
    print(f"Saved TorchScript model to {output_path}")


def export_onnx(model, output_path):
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, example_input, output_path, opset_version=11)
    print(f"Saved ONNX model to {output_path}")


def main():
    checkpoint_path = Path("models/checkpoints/phase2_best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path("models/checkpoints/phase1_best.pth")

    model = WasteClassifier(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    export_torchscript(model, "models/ecosort_model.pt")
    export_onnx(model, "models/ecosort_model.onnx")


if __name__ == "__main__":
    main()
