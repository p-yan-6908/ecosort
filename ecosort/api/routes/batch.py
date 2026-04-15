"""Batch Prediction API Routes.

This module provides endpoints for batch processing multiple images.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io
from typing import List

from ecosort.api.schemas import PredictionResponse
from ecosort.api.dependencies import get_predictor
from ecosort.inference.predictor import WastePredictor

router = APIRouter()


@router.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    summary="Batch classify multiple waste images",
    description="Upload multiple images to classify them in a single request. "
    "Maximum 10 images per batch.",
    responses={
        200: {
            "description": "Successful batch prediction",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "class_id": 0,
                            "class_name": "blue_bin",
                            "display_name": "Blue Bin (Recyclables)",
                            "confidence": 0.95,
                            "icon": "♻️",
                            "color": "#2563EB",
                            "description": "Cardboard, paper, plastic, metal, glass",
                            "all_probabilities": {"blue_bin": 0.95, "green_bin": 0.03, "garbage": 0.01},
                        }
                    ]
                }
            },
        },
        400: {"description": "Too many files or invalid file type"},
        500: {"description": "Prediction failed"},
    },
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Image files to classify (max 10)"),
    predictor: WastePredictor = Depends(get_predictor),
) -> List[PredictionResponse]:
    """Classify multiple waste images in a batch.
    
    Accepts multiple image files and returns predictions for each.
    Maximum 10 images per request to prevent timeout.
    
    Args:
        files: List of uploaded image files (max 10).
        predictor: Injected predictor instance.
    
    Returns:
        List of PredictionResponse, one for each image.
    
    Raises:
        HTTPException: 400 for invalid files or too many, 500 for prediction errors.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)}. Maximum 10 images per batch.",
        )

    results = []
    
    for i, file in enumerate(files):
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"File {i + 1}: Unsupported type {file.content_type}. Use JPEG, PNG, or WebP.",
            )

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"File {i + 1}: Failed to read image: {str(e)}",
            )

        try:
            result = predictor.predict(image)
            results.append(PredictionResponse(**result))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"File {i + 1}: Prediction failed: {str(e)}",
            )

    return results
