"""Waste Image Predictor for Inference"""

from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from PIL import Image

from ecosort.models.classifier import WasteClassifier
from ecosort.data.transforms import get_inference_transforms
from ecosort.constants import ONTARIO_CATEGORIES, CATEGORY_ID_TO_NAME


class WastePredictor:
    """Predictor class for waste classification inference."""

    def __init__(self, model_path: Path, num_classes: int = 6, device: str = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = WasteClassifier.from_checkpoint(
            model_path, num_classes=num_classes, device=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_inference_transforms()
        self.categories = {c.id: c for c in ONTARIO_CATEGORIES}

    def predict(self, image: Image.Image) -> Dict:
        image = image.convert("RGB")
        image_np = np.array(image)
        image_tensor = self.transform(image=image_np)["image"]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, predicted = torch.max(probabilities, 0)
        predicted_id = predicted.item()

        category = self.categories[predicted_id]
        all_probs = {
            CATEGORY_ID_TO_NAME[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }

        return {
            "class_id": predicted_id,
            "class_name": category.name,
            "display_name": category.display,
            "confidence": confidence.item(),
            "icon": category.icon,
            "color": category.color,
            "description": category.description,
            "all_probabilities": all_probs,
        }

    def predict_top_k(self, image: Image.Image, k: int = 3) -> List[Dict]:
        image = image.convert("RGB")
        image_np = np.array(image)
        image_tensor = self.transform(image=image_np)["image"]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_k_probs, top_k_ids = torch.topk(probabilities, k)

        results = []
        for prob, idx in zip(top_k_probs, top_k_ids):
            category = self.categories[idx.item()]
            results.append(
                {
                    "class_name": category.name,
                    "display_name": category.display,
                    "confidence": prob.item(),
                    "icon": category.icon,
                }
            )

        return results
