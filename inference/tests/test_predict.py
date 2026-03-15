"""Tests for the POST /predict endpoint."""

import base64
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.create_test_model import create_dummy_onnx_model


def _make_test_image(width=320, height=240) -> np.ndarray:
    """Create a simple test image."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 60, (130, 160, 200), -1)
    return img


def _encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@pytest.fixture
def loaded_app(tmp_path, monkeypatch):
    """Create an app with a loaded dummy model."""
    # Create dummy model
    model_path = tmp_path / "model.onnx"
    meta_path = tmp_path / "model-meta.json"

    created = create_dummy_onnx_model(str(model_path), n_classes=8)
    if not created:
        pytest.skip("onnx package not available")

    with open(meta_path, "w") as f:
        json.dump({"version": "0.1.0-test", "format": "onnx"}, f)

    monkeypatch.setattr("config.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("config.MODEL_FILE", "")

    # Re-import to get fresh app with model
    import importlib
    import main as main_mod

    main_mod.manager.__init__()
    main_mod.manager._model_dir = tmp_path
    main_mod.manager.load(str(model_path))

    return main_mod.app


@pytest.fixture
def unloaded_app(monkeypatch):
    """Create an app with no model loaded."""
    import main as main_mod

    main_mod.manager.__init__()
    main_mod.manager.loaded = False
    return main_mod.app


@pytest.mark.asyncio
async def test_predict_returns_score(loaded_app):
    img = _make_test_image()
    b64 = _encode_image(img)

    transport = ASGITransport(app=loaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"image": b64})

    assert resp.status_code == 200
    data = resp.json()
    assert "ai_score" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "inference_ms" in data
    assert 0 <= data["ai_score"] <= 10
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] == "0.1.0-test"
    assert data["inference_ms"] >= 0
    assert "face_detected" in data


@pytest.mark.asyncio
async def test_predict_with_data_uri_prefix(loaded_app):
    img = _make_test_image()
    b64 = "data:image/jpeg;base64," + _encode_image(img)

    transport = ASGITransport(app=loaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"image": b64})

    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["ai_score"] <= 10


@pytest.mark.asyncio
async def test_predict_invalid_image(loaded_app):
    transport = ASGITransport(app=loaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"image": "not-valid-base64!!!"})

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_no_model_returns_503(unloaded_app):
    img = _make_test_image()
    b64 = _encode_image(img)

    transport = ASGITransport(app=unloaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"image": b64})

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_predict_with_au_data(loaded_app):
    img = _make_test_image()
    b64 = _encode_image(img)
    au_data = {"AU4": 0.8, "AU6": 0.3, "AU7": 0.5}

    transport = ASGITransport(app=loaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"image": b64, "au_data": au_data})

    # AU data is accepted but not yet used (STORY-005)
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["ai_score"] <= 10


@pytest.mark.asyncio
async def test_predict_missing_image_field(loaded_app):
    transport = ASGITransport(app=loaded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={})

    assert resp.status_code == 422  # Pydantic validation error
