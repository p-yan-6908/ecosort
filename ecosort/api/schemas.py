"""Pydantic Models for API Request/Response"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    class_id: int = Field(..., description="Numeric class ID")
    class_name: str = Field(..., description="Class name (e.g., 'blue_bin')")
    display_name: str = Field(..., description="Human-readable display name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    icon: str = Field(..., description="Emoji icon for the category")
    color: str = Field(..., description="Hex color code")
    description: str = Field(..., description="Category description")
    all_probabilities: Dict[str, float] = Field(
        ..., description="All class probabilities"
    )


class TopKPrediction(BaseModel):
    class_name: str
    display_name: str
    confidence: float
    icon: str


class TopKResponse(BaseModel):
    predictions: List[TopKPrediction]


class ClassInfo(BaseModel):
    id: int
    name: str
    display_name: str
    color: str
    icon: str
    description: str


class ClassesResponse(BaseModel):
    categories: List[ClassInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str
