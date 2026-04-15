"""Model Metrics API Routes.

This module provides endpoints for model performance metrics.
"""

from fastapi import APIRouter, Depends
import torch
import time

from ecosort.api.schemas import HealthResponse
from ecosort.api.dependencies import get_predictor
from ecosort.inference.predictor import WastePredictor

router = APIRouter()


@router.get(
    "/metrics",
    summary="Get model metrics",
    description="Returns model performance metrics including parameter count, "
    "device, and inference benchmark.",
)
async def get_metrics(predictor: WastePredictor = Depends(get_predictor)):
    """Get model performance metrics.
    
    Returns information about the model including:
    - Total parameters
    - Device (CPU/MPS/CUDA)
    - Model size in MB
    - Average inference time
    
    Args:
        predictor: Injected predictor instance.
    
    Returns:
        Dict with model metrics.
    """
    model = predictor.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Benchmark inference time (10 runs)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(predictor.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_inference_ms = sum(times) / len(times)
    
    return {
        "model": {
            "architecture": "MobileNetV3-Small",
            "num_classes": 6,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_mb": round(total_size_mb, 2),
        },
        "device": predictor.device,
        "performance": {
            "avg_inference_ms": round(avg_inference_ms, 2),
            "min_inference_ms": round(min(times), 2),
            "max_inference_ms": round(max(times), 2),
        },
        "categories": list(predictor.categories.keys()),
    }


@router.get(
    "/metrics/info",
    summary="Get model info",
    description="Returns basic model information without benchmarking.",
)
async def get_model_info(predictor: WastePredictor = Depends(get_predictor)):
    """Get basic model information without benchmarking.
    
    Args:
        predictor: Injected predictor instance.
    
    Returns:
        Dict with basic model info.
    """
    model = predictor.model
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "architecture": "MobileNetV3-Small",
        "num_classes": 6,
        "parameters": total_params,
        "device": predictor.device,
    }
