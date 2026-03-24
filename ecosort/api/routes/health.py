"""Health and Info API Routes"""

from fastapi import APIRouter

from ecosort.api.schemas import HealthResponse, ClassesResponse, ClassInfo
from ecosort.api.dependencies import get_predictor
from ecosort.constants import ONTARIO_CATEGORIES

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", model_loaded=model_loaded, version="1.0.0")


@router.get("/classes", response_model=ClassesResponse)
async def get_classes():
    categories = [
        ClassInfo(
            id=cat.id,
            name=cat.name,
            display_name=cat.display,
            color=cat.color,
            icon=cat.icon,
            description=cat.description,
        )
        for cat in ONTARIO_CATEGORIES
    ]
    return ClassesResponse(categories=categories)
