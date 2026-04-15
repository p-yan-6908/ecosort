# autoresearch for EcoSort Model Improvement

## Benchmark

```yaml
command: python scripts/benchmark_model.py
metric_name: val_accuracy
metric_unit: "%"
direction: higher
```

## Files in Scope

```yaml
- ecosort/models/classifier.py
- ecosort/models/layers.py
```

## Off Limits

```yaml
- scripts/benchmark_model.py
- data/
- tests/
```

## Constraints

```yaml
- Model must have fewer than 50M parameters
- Inference time must be under 50ms on CPU
- Must use PyTorch and torchvision only
```

---

# Experiment Ideas

This is an experiment to have the LLM autonomously improve the EcoSort waste classification model.

## Goal

Get the highest validation accuracy within model size and inference time constraints.

## Ideas to try

1. **Attention mechanisms**: SE-Attention, CBAM, ECA-Net, CE-Attention
2. **Architecture changes**: Different backbone (EfficientNetV2-S, MobileNetV3-Large)
3. **Classifier head improvements**: Add BatchNorm, different activations, dropout rates
4. **Regularization**: Label smoothing, mixup, cutmix
5. **Multi-scale features**: Feature pyramid, spatial attention
6. **Knowledge distillation**: Train a smaller student from a larger teacher

## Key research findings to implement

Based on waste classification research (2023-2025):

1. **CE-Attention**: Channel-efficient attention improves feature extraction
2. **Multi-scale spatial features (SAFM)**: Better capture varying waste sizes
3. **Data augmentation**: Rotation, translation, noise injection
4. **Ensemble heads**: Multiple classifier heads for robustness

## Current baseline

- **Architecture**: MobileNetV3-Small
- **Parameters**: 2.5M
- **Accuracy**: ~87% (estimated on TrashNet-like dataset)
