"""
Pydantic Schemas — Chest X-ray API
=====================================
Request and response models for the FastAPI application.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Response Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    models_loaded: int = Field(..., example=4)
    device: str = Field(..., example="cuda")


class ModelListResponse(BaseModel):
    """Available models listing."""
    available_models: List[str] = Field(
        ..., example=["EfficientNet-B3", "Swin-Tiny", "SVM", "XGBoost"]
    )


class PredictionResponse(BaseModel):
    """Single image prediction response."""
    model: str = Field(..., example="EfficientNet-B3")
    prediction: str = Field(..., example="COVID-19")
    prediction_id: int = Field(..., example=1)
    confidence: float = Field(..., example=0.9542)
    all_probabilities: Dict[str, float] = Field(
        ...,
        example={
            "Normal": 0.01,
            "COVID-19": 0.95,
            "Pneumonia": 0.02,
            "Tuberculosis": 0.005,
            "Lung_Cancer": 0.005,
            "Fibrosis": 0.01,
        },
    )
    inference_time_ms: float = Field(..., example=23.4)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_images: int
    total_inference_time_ms: float


class ExplainResponse(BaseModel):
    """Explainability response with Grad-CAM overlay."""
    model: str
    prediction: str
    prediction_id: int
    confidence: float
    all_probabilities: Dict[str, float]
    gradcam_image_base64: str = Field(
        ..., description="Base64-encoded Grad-CAM overlay image (PNG)"
    )
    inference_time_ms: float


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., example="File type not supported. Use JPEG or PNG.")
