"""Prediction API Routes.

This module provides endpoints for waste image classification,
including single image prediction and top-k predictions.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from ecosort.api.schemas import PredictionResponse, TopKResponse
from ecosort.api.dependencies import get_predictor
from ecosort.inference.predictor import WastePredictor

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a waste image",
    description="Upload an image to classify what type of waste it is. "
    "Returns the predicted category, confidence score, and sorting tips.",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "class_id": 0,
                        "class_name": "blue_bin",
                        "display_name": "Blue Bin (Recyclables)",
                        "confidence": 0.95,
                        "icon": "♻️",
                        "color": "#2563EB",
                        "description": "Cardboard, paper, plastic, metal, glass",
                        "all_probabilities": {
                            "blue_bin": 0.95,
                            "green_bin": 0.03,
                            "garbage": 0.01,
                            "hazardous": 0.005,
                            "e_waste": 0.003,
                            "yard_waste": 0.002,
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid file type or corrupted image"},
        500: {"description": "Prediction failed"},
    },
)
async def predict_image(
    file: UploadFile = File(..., description="Image file to classify (JPEG, PNG, or WebP)"),
    predictor: WastePredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Classify a waste image and return the predicted category.
    
    Accepts an image file and returns:
    - The predicted waste category
    - Confidence score (0-1)
    - Category information (icon, color, description)
    - Probabilities for all categories
    
    Args:
        file: Uploaded image file (JPEG, PNG, or WebP).
        predictor: Injected predictor instance.
    
    Returns:
        PredictionResponse with classification results.
    
    Raises:
        HTTPException: 400 for invalid files, 500 for prediction errors.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    try:
        result = predictor.predict(image)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/top-k",
    response_model=TopKResponse,
    summary="Get top-k predictions",
    description="Upload an image to get the top-k most likely waste categories. "
    "Useful for cases where the model is uncertain.",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {
                                "class_name": "blue_bin",
                                "display_name": "Blue Bin",
                                "confidence": 0.95,
                                "icon": "♻️",
                            },
                            {
                                "class_name": "green_bin",
                                "display_name": "Green Bin",
                                "confidence": 0.03,
                                "icon": "🌿",
                            },
                            {
                                "class_name": "garbage",
                                "display_name": "Garbage",
                                "confidence": 0.01,
                                "icon": "🗑️",
                            },
                        ]
                    }
                }
            },
        },
        400: {"description": "Invalid file type or corrupted image"},
        500: {"description": "Prediction failed"},
    },
)
async def predict_top_k(
    file: UploadFile = File(..., description="Image file to classify"),
    k: int = 3,
    predictor: WastePredictor = Depends(get_predictor),
) -> TopKResponse:
    """Get the top-k most likely predictions for a waste image.
    
    Returns the k most likely categories with their confidence scores,
    useful for understanding model uncertainty.
    
    Args:
        file: Uploaded image file (JPEG, PNG, or WebP).
        k: Number of top predictions to return (default: 3).
        predictor: Injected predictor instance.
    
    Returns:
        TopKResponse with list of top predictions.
    
    Raises:
        HTTPException: 400 for invalid files, 500 for prediction errors.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {file.content_type}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    try:
        results = predictor.predict_top_k(image, k=k)
        return TopKResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
