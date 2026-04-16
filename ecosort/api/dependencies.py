"""API Dependencies - Model Loading"""

from pathlib import Path
from ecosort.inference.predictor import WastePredictor
from ecosort.config import Config

_predictor: WastePredictor = None


def load_model(config: Config):
    """Load the trained model from checkpoint."""
    global _predictor
    
    # Try checkpoints in order of preference
    checkpoints = [
        Path(config.checkpoints_dir) / "best_model.pth",
        Path(config.checkpoints_dir) / "phase2_best.pth",
        Path(config.checkpoints_dir) / "phase1_best.pth",
    ]
    
    model_path = None
    for checkpoint in checkpoints:
        if checkpoint.exists():
            model_path = checkpoint
            break
    
    if model_path and model_path.exists():
        _predictor = WastePredictor(model_path, num_classes=config.num_classes)
        print(f"✓ Model loaded from {model_path}")
        print(f"  Model parameters: {sum(p.numel() for p in _predictor.model.parameters()):,}")
    else:
        print(f"Warning: No model checkpoint found. Checked:")
        for cp in checkpoints:
            print(f"  - {cp}")


def get_predictor() -> WastePredictor:
    """Get the loaded predictor instance."""
    if _predictor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _predictor
