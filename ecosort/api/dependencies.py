"""API Dependencies - Model Loading"""

from pathlib import Path
import logging
import yaml
from ecosort.inference.predictor import WastePredictor

logger = logging.getLogger(__name__)

_predictor: WastePredictor = None


def load_model(config):
    """Load the trained model from checkpoint."""
    global _predictor
    
    # Get checkpoints directory from config
    checkpoints_dir = getattr(config, 'checkpoints_dir', 'models/checkpoints')
    
    # Try checkpoints in order of preference
    checkpoints = [
        Path(checkpoints_dir) / "best_model.pth",
        Path(checkpoints_dir) / "phase2_best.pth",
        Path(checkpoints_dir) / "phase1_best.pth",
    ]
    
    model_path = None
    for checkpoint in checkpoints:
        if checkpoint.exists():
            model_path = checkpoint
            break
    
    if model_path and model_path.exists():
        # Read head_type from config.yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        
        head_type = cfg.get("model", {}).get("head_type", "eca")
        backbone = cfg.get("model", {}).get("architecture", "mobilenet_v3_small")
        num_classes = cfg.get("model", {}).get("num_classes", 6)
        
        _predictor = WastePredictor(
            model_path, 
            num_classes=num_classes,
            head_type=head_type,
            backbone=backbone
        )
        logger.info("Model loaded from %s", model_path)
        logger.info("  Head type: %s", head_type)
        logger.info("  Backbone: %s", backbone)
        logger.info("  Model parameters: %d", sum(p.numel() for p in _predictor.model.parameters()))
    else:
        logger.warning("No model checkpoint found. Checked:")
        for cp in checkpoints:
            logger.warning("  - %s", cp)


def get_predictor() -> WastePredictor:
    """Get the loaded predictor instance."""
    if _predictor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _predictor
