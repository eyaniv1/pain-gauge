"""
Pain Gauge AI Inference Service

FastAPI microservice that loads a pre-trained pain detection model
and provides inference via HTTP API. Binds to localhost only.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from au_weights import blend_scores
from model_manager import ModelManager
from preprocessing import preprocess


manager = ModelManager()
startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    startup_time = time.time()

    # Attempt to load model on startup
    loaded = manager.load()
    if loaded:
        status = manager.get_status()
        print(f"Model loaded: {status['version']} ({status['format']})")
    else:
        print("No model found — service running without model. Place a model in models/active/")

    yield


app = FastAPI(
    title="Pain Gauge Inference Service",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Response Models ───────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    model_format: str | None = None
    uptime_s: float = 0.0


class ModelInfo(BaseModel):
    filename: str
    path: str
    format: str
    version: str
    active: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class ActivateRequest(BaseModel):
    path: str


class ActivateResponse(BaseModel):
    ok: bool
    active_model: str | None = None


class PredictRequest(BaseModel):
    image: str  # base64-encoded image
    au_data: dict | None = None  # optional AU data from face-api.js


class PredictResponse(BaseModel):
    ai_score: float
    confidence: float
    model_version: str | None = None
    inference_ms: float = 0.0
    face_detected: bool = True
    model_score: float | None = None
    au_score: float | None = None
    au_weights_applied: dict | None = None


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - startup_time if startup_time else 0.0
    status = manager.get_status()
    return HealthResponse(
        status="ok",
        model_loaded=status["loaded"],
        model_version=status["version"],
        model_format=status["format"],
        uptime_s=round(uptime, 1),
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(models=manager.list_models())


@app.post("/models/activate", response_model=ActivateResponse)
async def activate_model(req: ActivateRequest):
    loaded = manager.load(req.path)
    if not loaded:
        raise HTTPException(status_code=404, detail=f"Could not load model from: {req.path}")
    return ActivateResponse(ok=True, active_model=manager.version)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # Check model is loaded
    if not manager.loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Determine target size from model's expected input
    status = manager.get_status()
    input_shape = status.get("input_shape")
    if input_shape and len(input_shape) == 4:
        # Shape is (batch, channels, height, width)
        target_size = (input_shape[3], input_shape[2])  # (width, height)
    else:
        target_size = (224, 224)

    # Preprocess image — try with face detection, fall back to full image
    result = preprocess(req.image, target_size=target_size, require_face=False)
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Adapt channels if model expects grayscale (1 channel) but preprocessing outputs RGB (3 channels)
    image_array = result.image_array
    if input_shape and len(input_shape) == 4 and input_shape[1] == 1 and image_array.shape[1] == 3:
        # Convert RGB to grayscale: standard luminance weights
        image_array = 0.2989 * image_array[:, 0:1, :, :] + \
                      0.5870 * image_array[:, 1:2, :, :] + \
                      0.1140 * image_array[:, 2:3, :, :]

    # Run inference
    prediction = manager.predict(image_array)

    # Blend with AU data if provided
    blended = blend_scores(prediction["score"], req.au_data)

    # Lower confidence when no face was detected
    confidence = prediction["confidence"]
    if not result.face_found:
        confidence = round(confidence * 0.5, 2)

    return PredictResponse(
        ai_score=blended["final_score"],
        confidence=confidence,
        model_version=prediction["model_version"],
        inference_ms=prediction["inference_ms"],
        face_detected=result.face_found,
        model_score=blended["model_score"],
        au_score=blended["au_score"],
        au_weights_applied=blended["au_weights_applied"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
