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
from model_manager import ModelManager


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
