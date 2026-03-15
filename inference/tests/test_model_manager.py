"""Tests for the ModelManager."""

import json
import os
import tempfile

import numpy as np
import pytest

# Ensure inference/ is on path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_manager import ModelManager


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary model directory structure."""
    active = tmp_path / "active"
    archive = tmp_path / "archive"
    active.mkdir()
    archive.mkdir()
    return tmp_path


@pytest.fixture
def dummy_onnx_model(tmp_model_dir):
    """Create a dummy ONNX model for testing."""
    from tests.create_test_model import create_dummy_onnx_model

    model_path = tmp_model_dir / "active" / "model.onnx"
    created = create_dummy_onnx_model(str(model_path), n_classes=8)
    if not created:
        pytest.skip("onnx package not available")

    # Write meta
    meta = {"version": "0.1.0-test", "format": "onnx", "description": "Test model"}
    meta_path = tmp_model_dir / "active" / "model-meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return str(model_path)


class TestModelManagerLoad:
    def test_load_returns_false_when_no_model(self, tmp_model_dir, monkeypatch):
        monkeypatch.setattr("config.MODEL_DIR", str(tmp_model_dir / "active"))
        monkeypatch.setattr("config.MODEL_FILE", "")
        mgr = ModelManager()
        assert mgr.load() is False
        assert mgr.loaded is False

    def test_load_onnx_model(self, dummy_onnx_model, tmp_model_dir, monkeypatch):
        monkeypatch.setattr("config.MODEL_DIR", str(tmp_model_dir / "active"))
        monkeypatch.setattr("config.MODEL_FILE", "")
        mgr = ModelManager()
        assert mgr.load() is True
        assert mgr.loaded is True
        assert mgr.format == "onnx"
        assert mgr.version == "0.1.0-test"

    def test_load_explicit_path(self, dummy_onnx_model):
        mgr = ModelManager()
        assert mgr.load(dummy_onnx_model) is True
        assert mgr.loaded is True

    def test_load_nonexistent_path(self):
        mgr = ModelManager()
        assert mgr.load("/nonexistent/model.onnx") is False
        assert mgr.loaded is False

    def test_load_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "model.xyz"
        bad_file.write_text("not a model")
        mgr = ModelManager()
        assert mgr.load(str(bad_file)) is False


class TestModelManagerPredict:
    def test_predict_raises_when_no_model(self):
        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="No model loaded"):
            mgr.predict(np.zeros((1, 3, 64, 64)))

    def test_predict_returns_score_and_confidence(self, dummy_onnx_model):
        mgr = ModelManager()
        mgr.load(dummy_onnx_model)

        # Create a dummy input matching model's expected shape
        dummy_input = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = mgr.predict(dummy_input)

        assert "score" in result
        assert "confidence" in result
        assert "model_version" in result
        assert "inference_ms" in result
        assert 0 <= result["score"] <= 10
        assert 0 <= result["confidence"] <= 1
        assert result["model_version"] == "0.1.0-test"
        assert result["inference_ms"] >= 0

    def test_predict_returns_consistent_results(self, dummy_onnx_model):
        mgr = ModelManager()
        mgr.load(dummy_onnx_model)

        dummy_input = np.ones((1, 3, 64, 64), dtype=np.float32) * 0.5
        r1 = mgr.predict(dummy_input)
        r2 = mgr.predict(dummy_input)

        assert r1["score"] == r2["score"]
        assert r1["confidence"] == r2["confidence"]


class TestModelManagerStatus:
    def test_status_when_no_model(self):
        mgr = ModelManager()
        status = mgr.get_status()
        assert status["loaded"] is False
        assert status["version"] is None
        assert status["format"] is None

    def test_status_after_load(self, dummy_onnx_model):
        mgr = ModelManager()
        mgr.load(dummy_onnx_model)
        status = mgr.get_status()
        assert status["loaded"] is True
        assert status["version"] == "0.1.0-test"
        assert status["format"] == "onnx"
        assert status["load_time_s"] >= 0


class TestModelManagerList:
    def test_list_models_empty(self, tmp_model_dir, monkeypatch):
        monkeypatch.setattr("config.MODEL_DIR", str(tmp_model_dir / "active"))
        mgr = ModelManager()
        models = mgr.list_models()
        assert models == []

    def test_list_models_finds_active(self, dummy_onnx_model, tmp_model_dir, monkeypatch):
        monkeypatch.setattr("config.MODEL_DIR", str(tmp_model_dir / "active"))
        mgr = ModelManager()
        models = mgr.list_models()
        assert len(models) == 1
        assert models[0]["active"] is True
        assert models[0]["format"] == "onnx"
        assert models[0]["version"] == "0.1.0-test"
