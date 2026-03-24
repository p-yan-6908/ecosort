"""Prediction API Routes"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from ecosort.api.schemas import PredictionResponse, TopKResponse
from ecosort.api.dependencies import get_predictor
from ecosort.inference.predictor import WastePredictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), predictor: WastePredictor = Depends(get_predictor)
):
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


@router.post("/predict/top-k", response_model=TopKResponse)
async def predict_top_k(
    file: UploadFile = File(...),
    k: int = 3,
    predictor: WastePredictor = Depends(get_predictor),
):
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
