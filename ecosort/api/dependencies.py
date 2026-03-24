"""API Dependencies - Model Loading"""

from pathlib import Path
from ecosort.inference.predictor import WastePredictor
from ecosort.config import Config

_predictor: WastePredictor = None


def load_model(config: Config):
    global _predictor
    model_path = Path(config.checkpoints_dir) / "phase2_best.pth"
    if not model_path.exists():
        model_path = Path(config.checkpoints_dir) / "phase1_best.pth"
    if model_path.exists():
        _predictor = WastePredictor(model_path, num_classes=config.num_classes)
    else:
        print(f"Warning: No model checkpoint found at {model_path}")


def get_predictor() -> WastePredictor:
    if _predictor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _predictor
