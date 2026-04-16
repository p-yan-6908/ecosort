# EcoSort Model Documentation

## Model Architecture

**Best Configuration (from Autoresearch):**
- **Architecture:** MobileNetV3-Small
- **Head:** ECA Attention (Efficient Channel Attention)
- **Parameters:** 1.10M
- **Model Size:** ~4 MB

### Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.28% |
| Test Accuracy | 97.32% |
| Inference Time | ~50ms (CPU) |
| Model Size | 4 MB |

---

## Training Details

### Configuration

```yaml
model:
  architecture: mobilenet_v3_small
  head_type: eca
  num_classes: 6
  dropout: 0.2
  pretrained: true

training:
  label_smoothing: 0.1
  phase1:
    epochs: 10
    learning_rate: 0.01
    freeze_backbone: true
  phase2:
    epochs: 20
    learning_rate: 0.0001
```

### Two-Phase Training

1. **Phase 1 (Frozen Backbone):**
   - Train classifier head only
   - 10 epochs, LR=0.01
   - Achieved: 95.10% validation accuracy

2. **Phase 2 (Full Fine-tuning):**
   - Unfreeze entire network
   - 17 epochs, LR=0.0001
   - Early stopping with patience=10
   - Achieved: 97.28% validation accuracy

### Dataset

| Split | Images |
|-------|--------|
| Train | 2,988 |
| Validation | 551 |
| Test | 559 |
| **Total** | **4,098** |

### Categories

1. Blue Bin (Recyclables) - ♻️
2. Green Bin (Organics) - 🌿
3. Garbage (Black Bin) - 🗑️
4. Household Hazardous - ⚠️
5. Electronic Waste - 💻
6. Yard Waste - 🍂

---

## Using the Model

### Python API

```python
from pathlib import Path
from PIL import Image
from ecosort.inference.predictor import WastePredictor

# Load model
predictor = WastePredictor(
    model_path=Path("models/checkpoints/best_model.pth"),
    num_classes=6
)

# Predict
image = Image.open("waste_image.jpg")
result = predictor.predict(image)

print(f"Class: {result['display_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### REST API

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

### Command Line

```bash
python -m ecosort.inference.predictor --image image.jpg
```

---

## Model File Locations

```
models/
├── checkpoints/
│   ├── best_model.pth      # Final trained model (97.32%)
│   ├── phase1_best.pth     # Phase 1 checkpoint
│   └── phase2_best.pth     # Phase 2 checkpoint (if exists)
└── exported/
    └── ecosort.onnx        # ONNX export (optional)
```

---

## Model Export

### Export to ONNX

```bash
python scripts/export_model.py --output models/exported/ecosort.onnx
```

### Export to TorchScript

```python
import torch
from ecosort.models.classifier import WasteClassifier

model = WasteClassifier(num_classes=6, head_type="eca")
model.load_state_dict(torch.load("models/checkpoints/best_model.pth"))
model.eval()

# TorchScript export
scripted = torch.jit.script(model)
scripted.save("models/exported/ecosort.pt")
```

---

## Model Quantization (Optional)

For smaller model size and faster inference:

```python
import torch

# Load model
model = WasteClassifier(num_classes=6, head_type="eca")
model.load_state_dict(torch.load("models/checkpoints/best_model.pth"))
model.eval()

# Dynamic quantization
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized.state_dict(), "models/exported/ecosort_quantized.pth")
```

Expected size reduction: ~2-3x smaller

---

## Autoresearch Results Summary

| Configuration | Val Acc | Params | Notes |
|---------------|---------|--------|-------|
| Baseline | 96.37% | 1.10M | Default head |
| SE Attention | 96.01% | 1.26M | Overfitting |
| **ECA + LS 0.1** | **97.28%** | **1.10M** | **BEST** |
| MobileNetV3-Large | 96.91% | 3.44M | Larger, not better |

**Key Findings:**
- Label smoothing (0.1) provides +2.72% improvement
- ECA outperforms SE with fewer parameters
- Small models generalize better with limited data

See `autoresearch_results.md` for full experiment details.
