#!/usr/bin/env python3
"""Export EcoSort model to ONNX, TorchScript, and quantized formats."""

import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from ecosort.models.classifier import WasteClassifier

# Config
NUM_CLASSES = 6
HEAD_TYPE = "eca"
BACKBONE = "mobilenet_v3_small"
CHECKPOINT_PATH = Path("models/checkpoints/best_model.pth")
EXPORT_DIR = Path("models/exported")
IMAGE_SIZE = 224


def load_model():
    """Load trained model from checkpoint."""
    print("Loading model...")
    model = WasteClassifier(
        num_classes=NUM_CLASSES, head_type=HEAD_TYPE,
        backbone=BACKBONE, pretrained=False,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded from {CHECKPOINT_PATH}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def export_onnx(model):
    """Export model to ONNX format."""
    output_path = EXPORT_DIR / "ecosort.onnx"
    print("\n--- ONNX Export ---")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, dummy_input, str(output_path),
            opset_version=18,
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.2f} MB)")
    # Verify
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(output_path))
        ort_in = {sess.get_inputs()[0].name: np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)}
        ort_out = sess.run(None, ort_in)
        print(f"  ONNX Runtime verification: OK (output shape: {ort_out[0].shape})")
    except ImportError:
        print("  onnxruntime not installed, skipping verification")
    return output_path


def export_torchscript(model):
    """Export model to TorchScript format."""
    output_path = EXPORT_DIR / "ecosort.pt"
    print("\n--- TorchScript Export ---")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.2f} MB)")
    loaded = torch.jit.load(str(output_path))
    with torch.no_grad():
        out = loaded(dummy_input)
    print(f"  Verification: OK (output shape: {out.shape})")
    return output_path


def export_quantized(model):
    """Export dynamically quantized model."""
    output_path = EXPORT_DIR / "ecosort_quantized.pth"
    print("\n--- Quantized Export ---")
    torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.2f} MB)")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = quantized(dummy_input)
    print(f"  Verification: OK (output shape: {out.shape})")
    return output_path


def benchmark_inference(model, name="Model", num_runs=30):
    """Benchmark inference time."""
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        for _ in range(5): model(dummy_input)  # warmup
        times = []
        for _ in range(num_runs):
            s = time.perf_counter()
            model(dummy_input)
            times.append((time.perf_counter() - s) * 1000)
    avg = sum(times) / len(times)
    print(f"  {name}: avg={avg:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")
    return avg


def main():
    print("=" * 70)
    print("ECOSORT MODEL EXPORT")
    print("=" * 70)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    print("\n--- Original Model Benchmark ---")
    benchmark_inference(model, "Original")

    export_onnx(model)
    export_torchscript(model)
    export_quantized(model)

    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"  {'Format':<25} {'Size (MB)':<12} {'Path'}")
    print(f"  {'-'*25} {'-'*12} {'-'*30}")
    for path in sorted(EXPORT_DIR.iterdir()):
        size_mb = path.stat().st_size / (1024 * 1024)
        if "quantized" in path.name:
            fmt = "Quantized (INT8)"
        elif path.suffix == ".onnx":
            fmt = "ONNX"
        elif path.suffix == ".pt":
            fmt = "TorchScript"
        else:
            fmt = path.suffix
        print(f"  {fmt:<25} {size_mb:<12.2f} {path}")


if __name__ == "__main__":
    main()
