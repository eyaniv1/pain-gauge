"""
Pain Gauge AI Inference Service

FastAPI microservice that loads a pre-trained pain detection model
and provides inference via HTTP API. Binds to localhost only.

Supports comparative models: when calibrated with a reference frame,
the model compares the current face against the calibration face to
detect changes in pain expression.
"""

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from au_weights import blend_scores
from model_manager import ModelManager
from preprocessing import preprocess


manager = ModelManager()
startup_time: float = 0.0

# ── Calibration State ─────────────────────────────────────
# Stores the reference frame and pain level from calibration.
# Single-user app, so one reference at a time is sufficient.

_calibration = {
    "reference_frame": None,   # Preprocessed grayscale array (1, 1, H, W)
    "pain_level": 0,           # Patient's reported pain at calibration (0-10)
    "calibrated": False,
    "self_score": 0.0,         # Model output when ref vs ref (baseline offset)
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    startup_time = time.time()

    loaded = manager.load()
    if loaded:
        status = manager.get_status()
        print(f"Model loaded: {status['version']} ({status['format']})")
        if manager.is_comparative():
            print("Comparative model detected — calibration will provide reference frame")
    else:
        print("No model found — service running without model. Place a model in models/active/")

    yield


app = FastAPI(
    title="Pain Gauge Inference Service",
    version="0.3.0",
    lifespan=lifespan,
)


# ── Request / Response Models ─────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    model_format: str | None = None
    comparative: bool = False
    calibrated: bool = False
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


class CalibrateRequest(BaseModel):
    image: str       # base64-encoded face image at calibration
    pain_level: int  # patient's reported pain (0-10)


class CalibrateResponse(BaseModel):
    ok: bool
    pain_level: int
    face_detected: bool


class PredictRequest(BaseModel):
    image: str  # base64-encoded image
    au_data: dict | None = None


class PredictResponse(BaseModel):
    ai_score: float
    confidence: float
    model_version: str | None = None
    inference_ms: float = 0.0
    face_detected: bool = True
    calibrated: bool = False
    model_score: float | None = None
    au_score: float | None = None
    au_weights_applied: dict | None = None


# ── Helper: get model target size ─────────────────────────

def _get_target_size():
    """Get (width, height) from model input shape."""
    status = manager.get_status()
    input_shape = status.get("input_shape")
    if input_shape and len(input_shape) == 4:
        return (input_shape[3], input_shape[2])
    return (224, 224)


def _is_grayscale_model():
    """Check if model expects grayscale input."""
    status = manager.get_status()
    input_shape = status.get("input_shape")
    if input_shape and len(input_shape) == 4:
        return input_shape[1] in (1, 2)  # 1-channel or 2-channel (comparative)
    return False


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
        comparative=manager.is_comparative(),
        calibrated=_calibration["calibrated"],
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


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(req: CalibrateRequest):
    """
    Store a reference frame for comparative inference.

    The clinician captures the patient's face at a known pain level.
    This becomes the baseline for detecting relative changes.
    """
    target_size = _get_target_size()
    use_grayscale = _is_grayscale_model()

    result = preprocess(req.image, target_size=target_size, require_face=False, grayscale=use_grayscale)
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    pain_level = max(0, min(10, req.pain_level))

    _calibration["reference_frame"] = result.image_array  # (1, 1, H, W) grayscale
    _calibration["pain_level"] = pain_level
    _calibration["calibrated"] = True

    # Compute self-score: model output when reference is compared to itself.
    # This is the model's baseline offset (not zero due to sigmoid/nonlinearities).
    # During prediction, we subtract this to get the true pain delta.
    self_score = 0.0
    if manager.loaded and manager.is_comparative():
        ref = result.image_array
        self_input = np.concatenate([ref, ref], axis=1)  # (1, 2, H, W)
        self_pred = manager.predict(self_input)
        self_score = self_pred["score"]
    _calibration["self_score"] = self_score

    print(f"Calibrated: reference frame stored at pain level {pain_level}, "
          f"face_detected={result.face_found}, shape={result.image_array.shape}, "
          f"self_score={self_score:.2f}")

    return CalibrateResponse(
        ok=True,
        pain_level=pain_level,
        face_detected=result.face_found,
    )


@app.post("/clear-calibration")
async def clear_calibration():
    """Clear the stored reference frame."""
    _calibration["reference_frame"] = None
    _calibration["pain_level"] = 0
    _calibration["calibrated"] = False
    _calibration["self_score"] = 0.0
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not manager.loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    target_size = _get_target_size()
    use_grayscale = _is_grayscale_model()

    # Preprocess the target (current) frame
    result = preprocess(req.image, target_size=target_size, require_face=False, grayscale=use_grayscale)
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    image_array = result.image_array

    # For comparative models: concatenate target + reference along channel axis
    if manager.is_comparative():
        target = image_array  # (1, 1, H, W)

        if _calibration["calibrated"] and _calibration["reference_frame"] is not None:
            reference = _calibration["reference_frame"]  # (1, 1, H, W)
        else:
            # No calibration — use zero reference (fallback)
            reference = np.zeros_like(target)

        # Combine: (1, 2, H, W) — channel 0 = target, channel 1 = reference
        image_array = np.concatenate([target, reference], axis=1)
    elif not use_grayscale and image_array.shape[1] == 3:
        # Non-comparative model that might need channel adaptation
        status = manager.get_status()
        input_shape = status.get("input_shape")
        if input_shape and len(input_shape) == 4 and input_shape[1] == 1:
            image_array = 0.2989 * image_array[:, 0:1, :, :] + \
                          0.5870 * image_array[:, 1:2, :, :] + \
                          0.1140 * image_array[:, 2:3, :, :]

    # Run inference
    prediction = manager.predict(image_array)
    raw_model_score = prediction["score"]

    # For comparative models with calibration: anchor score to calibration pain level.
    # The model compares target vs reference via feature subtraction. When the target
    # looks identical to the reference, the model still outputs a nonzero "self_score"
    # (due to sigmoid/FC nonlinearities). We subtract this offset to get the true delta,
    # then anchor to the patient's reported pain at calibration time.
    if manager.is_comparative() and _calibration["calibrated"]:
        cal_pain = _calibration["pain_level"]
        self_score = _calibration["self_score"]
        model_score = prediction["score"]

        # delta = how much the model thinks pain changed from reference
        delta = model_score - self_score
        # Scale delta to remaining range (can't exceed 10 or go below 0)
        if delta >= 0:
            remaining = 10 - cal_pain
            scaled_delta = delta * (remaining / max(10 - self_score, 0.1))
        else:
            scaled_delta = delta * (cal_pain / max(self_score, 0.1))

        anchored = cal_pain + scaled_delta
        prediction["score"] = round(max(0, min(10, anchored)), 2)

    # Blend with AU data if provided
    blended = blend_scores(prediction["score"], req.au_data)

    confidence = prediction["confidence"]
    if not result.face_found:
        confidence = round(confidence * 0.5, 2)

    return PredictResponse(
        ai_score=blended["final_score"],
        confidence=confidence,
        model_version=prediction["model_version"],
        inference_ms=prediction["inference_ms"],
        face_detected=result.face_found,
        calibrated=_calibration["calibrated"],
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
