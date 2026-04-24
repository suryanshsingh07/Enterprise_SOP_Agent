"""
FastAPI Application — Chest X-ray Multi-Disease Detection API
================================================================
Production-ready API for chest X-ray classification and explainability.

Endpoints:
    GET  /health           — Health check
    GET  /models           — List available models
    POST /predict/{model}  — Single image prediction
    POST /predict/batch    — Batch prediction
    POST /explain/{model}  — Grad-CAM explainability
"""

from __future__ import annotations

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Imports from project ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.model_registry import ModelRegistry
from api.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    ModelListResponse,
    PredictionResponse,
)
from api.middleware import setup_cors, logging_middleware
from src.data.dataset_builder import CLASS_NAMES, UNIFIED_CLASSES

# ── Globals ───────────────────────────────────────────────────────────────────
registry: ModelRegistry = ModelRegistry()
DEVICE: torch.device = torch.device("cpu")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
IMG_SIZE = 224


# =============================================================================
# Lifespan (startup / shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup."""
    global DEVICE, registry

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    registry = ModelRegistry(
        checkpoint_dir="outputs/checkpoints",
        classical_dir="outputs/classical",
        device=str(DEVICE),
    )
    registry.load_all()
    logger.info(f"Models loaded: {len(registry)}")

    yield  # Application runs here

    logger.info("Shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Chest X-ray Multi-Disease Detection API",
    description=(
        "A unified API for chest X-ray classification across 6 disease classes "
        "using Classical ML, Deep CNNs, and Vision Transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
setup_cors(app)
app.middleware("http")(logging_middleware)


# =============================================================================
# Helper Functions
# =============================================================================
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image bytes to model input tensor.

    Pipeline: bytes → PIL → grayscale → CLAHE → RGB → normalize → tensor

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.

    Returns
    -------
    torch.Tensor
        Preprocessed tensor, shape (1, 3, 224, 224).
    """
    # Read image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    img_array = np.array(pil_img, dtype=np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_array)

    # Resize
    img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))

    # Gray → RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_rgb - mean) / std

    # To tensor: (H, W, C) → (1, C, H, W)
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def preprocess_for_classical(image_bytes: bytes) -> np.ndarray:
    """Preprocess for classical ML models (HOG+LBP+PCA features).

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.

    Returns
    -------
    np.ndarray
        Feature vector ready for classical model.
    """
    from src.data.preprocessing import ChestXrayPreprocessor
    from src.features.classical_features import ClassicalFeatureExtractor

    # Read and preprocess
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_array = np.array(pil_img, dtype=np.uint8)

    preprocessor = ChestXrayPreprocessor(img_size=IMG_SIZE)
    img_enhanced = preprocessor.apply_clahe(img_array)
    img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))

    # Extract features
    extractor = ClassicalFeatureExtractor()
    features_path = Path("outputs/features/feature_extractor.joblib")
    if features_path.exists():
        extractor.load(features_path)
        raw_feat = extractor.extract_single(img_resized).reshape(1, -1)
        feat = extractor.transform([img_resized])
    else:
        feat = extractor.extract_single(img_resized).reshape(1, -1)

    return feat


async def validate_upload(file: UploadFile) -> bytes:
    """Validate uploaded file type and size.

    Parameters
    ----------
    file : UploadFile
        Uploaded file.

    Returns
    -------
    bytes
        File content bytes.

    Raises
    ------
    HTTPException
        If file type or size is invalid.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not supported. Use JPEG or PNG.",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1e6:.1f} MB). Max: 10 MB.",
        )

    return content


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=len(registry),
        device=str(DEVICE),
    )


@app.get("/models", response_model=ModelListResponse, tags=["System"])
async def list_models() -> ModelListResponse:
    """List all available loaded models."""
    return ModelListResponse(available_models=registry.list_models())


@app.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict(model_name: str, file: UploadFile = File(...)) -> PredictionResponse:
    """Predict disease class from a chest X-ray image.

    Parameters
    ----------
    model_name : str
        Name of the model to use (from /models endpoint).
    file : UploadFile
        Chest X-ray image (JPEG or PNG, max 10 MB).
    """
    # Validate
    if model_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {registry.list_models()}",
        )

    content = await validate_upload(file)
    model = registry.get_model(model_name)
    model_type = registry.get_model_type(model_name)

    t0 = time.time()

    if model_type == "dl":
        tensor = preprocess_image(content).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        features = preprocess_for_classical(content)
        proba = model.predict_proba(features)[0]

    inference_ms = (time.time() - t0) * 1000

    pred_id = int(np.argmax(proba))
    pred_name = CLASS_NAMES[pred_id]
    confidence = float(proba[pred_id])

    all_probs = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return PredictionResponse(
        model=model_name,
        prediction=pred_name,
        prediction_id=pred_id,
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
        inference_time_ms=round(inference_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str = "EfficientNet-B3",
) -> BatchPredictionResponse:
    """Batch prediction on multiple images.

    Processes in batches of 8.

    Parameters
    ----------
    files : list[UploadFile]
        List of chest X-ray images.
    model_name : str
        Model name. Default 'EfficientNet-B3'.
    """
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    model = registry.get_model(model_name)
    model_type = registry.get_model_type(model_name)
    predictions = []

    t0 = time.time()
    batch_size = 8

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]

        for file in batch_files:
            content = await validate_upload(file)
            t_single = time.time()

            if model_type == "dl":
                tensor = preprocess_image(content).to(DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    proba = F.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                features = preprocess_for_classical(content)
                proba = model.predict_proba(features)[0]

            single_ms = (time.time() - t_single) * 1000

            pred_id = int(np.argmax(proba))
            predictions.append(PredictionResponse(
                model=model_name,
                prediction=CLASS_NAMES[pred_id],
                prediction_id=pred_id,
                confidence=round(float(proba[pred_id]), 4),
                all_probabilities={CLASS_NAMES[j]: round(float(p), 4) for j, p in enumerate(proba)},
                inference_time_ms=round(single_ms, 2),
            ))

    total_ms = (time.time() - t0) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(predictions),
        total_inference_time_ms=round(total_ms, 2),
    )


@app.post(
    "/explain/{model_name}",
    response_model=ExplainResponse,
    tags=["Explainability"],
)
async def explain(model_name: str, file: UploadFile = File(...)) -> ExplainResponse:
    """Generate Grad-CAM explanation for a prediction.

    Returns prediction + base64-encoded Grad-CAM overlay image.

    Parameters
    ----------
    model_name : str
        Model name (DL models only).
    file : UploadFile
        Chest X-ray image.
    """
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    if registry.get_model_type(model_name) != "dl":
        raise HTTPException(
            status_code=400,
            detail="Grad-CAM is only available for deep learning models.",
        )

    content = await validate_upload(file)
    model = registry.get_model(model_name)

    t0 = time.time()

    tensor = preprocess_image(content).to(DEVICE)

    # Prediction
    with torch.no_grad():
        logits = model(tensor)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(proba))

    # Grad-CAM
    from src.explainability.gradcam import GradCAM, get_target_layer

    model_type_str = "efficientnet" if "efficient" in model_name.lower() else "resnet"
    try:
        target_layer = get_target_layer(model, model_type_str)
    except ValueError:
        # Fallback: try to find last conv layer
        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = name
                break
        if target_layer is None:
            raise HTTPException(status_code=500, detail="Could not find target layer for Grad-CAM.")

    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(tensor, class_idx=pred_id)
    gradcam.remove_hooks()

    # Create overlay image
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    img_array = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)

    # Encode to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    inference_ms = (time.time() - t0) * 1000

    return ExplainResponse(
        model=model_name,
        prediction=CLASS_NAMES[pred_id],
        prediction_id=pred_id,
        confidence=round(float(proba[pred_id]), 4),
        all_probabilities={CLASS_NAMES[j]: round(float(p), 4) for j, p in enumerate(proba)},
        gradcam_image_base64=b64_image,
        inference_time_ms=round(inference_ms, 2),
    )


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
