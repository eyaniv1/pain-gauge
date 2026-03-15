"""
Pain Gauge AI Inference Service

FastAPI microservice that loads a pre-trained pain detection model
and provides inference via HTTP API. Binds to localhost only.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import config


startup_time: float = 0.0
model_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    startup_time = time.time()
    # Model loading will be added in STORY-002
    yield


app = FastAPI(
    title="Pain Gauge Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    model_format: str | None = None
    uptime_s: float = 0.0


@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - startup_time if startup_time else 0.0
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_version=None,
        model_format=None,
        uptime_s=round(uptime, 1),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
