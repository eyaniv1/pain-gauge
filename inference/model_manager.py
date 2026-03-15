"""
Model Manager

Handles loading, switching, and querying pre-trained pain detection models.
Supports both ONNX (.onnx) and PyTorch (.pt, .pth) formats.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

import config


class ModelManager:
    def __init__(self):
        self.model = None
        self.session = None  # ONNX inference session
        self.format: str | None = None
        self.version: str | None = None
        self.meta: dict = {}
        self.loaded = False
        self.load_time_s: float = 0.0
        self._model_dir = Path(config.MODEL_DIR)
        self._input_name: str | None = None
        self._input_shape: tuple | None = None

    def load(self, model_path: str | None = None) -> bool:
        """Load a model from file. Auto-detects format from extension."""
        if model_path is None:
            model_path = self._find_active_model()

        if model_path is None:
            self.loaded = False
            return False

        path = Path(model_path)
        if not path.exists():
            self.loaded = False
            return False

        start = time.time()

        try:
            if path.suffix == ".onnx":
                self._load_onnx(path)
            elif path.suffix in (".pt", ".pth"):
                self._load_pytorch(path)
            else:
                raise ValueError(f"Unsupported model format: {path.suffix}")

            self.load_time_s = time.time() - start
            self.loaded = True
            self._load_meta(path)
            return True

        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
            self.loaded = False
            return False

    def _find_active_model(self) -> str | None:
        """Find the active model in the models directory."""
        if config.MODEL_FILE:
            candidate = self._model_dir / config.MODEL_FILE
            if candidate.exists():
                return str(candidate)

        # Search for model files in active directory
        for ext in (".onnx", ".pt", ".pth"):
            candidates = list(self._model_dir.glob(f"*{ext}"))
            if candidates:
                return str(candidates[0])

        return None

    def _load_onnx(self, path: Path):
        """Load an ONNX model via ONNX Runtime."""
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self.format = "onnx"

        # Get input metadata
        input_info = self.session.get_inputs()[0]
        self._input_name = input_info.name
        self._input_shape = tuple(input_info.shape)

    def _load_pytorch(self, path: Path):
        """Load a PyTorch model."""
        import torch

        self.model = torch.load(str(path), map_location="cpu", weights_only=False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.format = "pytorch"

    def _load_meta(self, model_path: Path):
        """Load model metadata from companion JSON file."""
        meta_path = model_path.parent / "model-meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
                self.version = self.meta.get("version", "unknown")
        else:
            self.version = "unknown"
            self.meta = {"version": self.version, "format": self.format}

    def predict(self, image_array: np.ndarray) -> dict:
        """
        Run inference on a preprocessed image array.

        Args:
            image_array: Preprocessed image as numpy array, shape matching model input.

        Returns:
            dict with 'score' (0-10), 'confidence' (0-1), 'raw_output'.
        """
        if not self.loaded:
            raise RuntimeError("No model loaded")

        start = time.time()

        if self.format == "onnx":
            result = self._predict_onnx(image_array)
        elif self.format == "pytorch":
            result = self._predict_pytorch(image_array)
        else:
            raise RuntimeError(f"Unknown format: {self.format}")

        result["inference_ms"] = round((time.time() - start) * 1000, 1)
        result["model_version"] = self.version
        return result

    def _predict_onnx(self, image_array: np.ndarray) -> dict:
        """Run ONNX inference."""
        # Ensure correct dtype
        input_array = image_array.astype(np.float32)

        # Run inference
        outputs = self.session.run(None, {self._input_name: input_array})
        raw_output = outputs[0]

        return self._interpret_output(raw_output)

    def _predict_pytorch(self, image_array: np.ndarray) -> dict:
        """Run PyTorch inference."""
        import torch

        input_tensor = torch.from_numpy(image_array.astype(np.float32))

        with torch.no_grad():
            raw_output = self.model(input_tensor)

        if isinstance(raw_output, torch.Tensor):
            raw_output = raw_output.numpy()

        return self._interpret_output(raw_output)

    def is_comparative(self) -> bool:
        """Check if model is a comparative model (needs target + reference)."""
        return self.meta.get("comparative", False)

    def _interpret_output(self, raw_output: np.ndarray) -> dict:
        """
        Interpret model output into pain score and confidence.

        Handles three output formats:
        1. Comparative (40 outputs): extract PSPI from metadata-specified index
        2. Classification (N classes): softmax → map to 0-10 scale
        3. Regression (single float): clip to 0-10
        """
        output = raw_output.flatten()

        # Comparative model — extract PSPI using metadata
        if self.is_comparative() and "pspi_index" in self.meta:
            pspi_index = self.meta["pspi_index"]
            pspi_max = self.meta.get("pspi_max", 16)
            raw_pspi = float(output[pspi_index])
            pspi_clamped = max(0.0, min(pspi_max, raw_pspi))
            score = round(pspi_clamped * (10.0 / pspi_max), 2)
            confidence = 0.8
            return {"score": score, "confidence": confidence, "raw_output": output.tolist()}

        if len(output) == 1:
            # Regression model — single score
            score = float(np.clip(output[0], 0, 10))
            confidence = 0.8
            return {"score": round(score, 2), "confidence": confidence, "raw_output": output.tolist()}

        # Classification model — softmax probabilities
        if np.any(output < 0) or np.sum(output) < 0.99 or np.sum(output) > 1.01:
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
        else:
            probs = output

        if len(probs) == 8:
            pain_weights = np.array([0.0, 0.0, 1.0, 4.0, 6.0, 7.0, 8.0, 3.0])
            score = float(np.dot(probs, pain_weights))
            confidence = float(1.0 - probs[0] - probs[1])
        else:
            n_classes = len(probs)
            class_scores = np.linspace(0, 10, n_classes)
            score = float(np.dot(probs, class_scores))
            confidence = float(1.0 - np.max(probs) if np.max(probs) < 0.5 else np.max(probs))

        score = round(np.clip(score, 0, 10), 2)
        confidence = round(np.clip(confidence, 0, 1), 2)

        return {"score": score, "confidence": confidence, "raw_output": probs.tolist()}

    def list_models(self) -> list[dict]:
        """List all available models in models directory."""
        models = []
        base_dir = self._model_dir.parent  # models/

        for subdir in ("active", "archive"):
            dir_path = base_dir / subdir
            if not dir_path.exists():
                continue
            for ext in ("*.onnx", "*.pt", "*.pth"):
                for f in dir_path.glob(ext):
                    meta_path = f.parent / "model-meta.json"
                    meta = {}
                    if meta_path.exists():
                        with open(meta_path) as mf:
                            meta = json.load(mf)

                    models.append({
                        "filename": f.name,
                        "path": str(f),
                        "format": f.suffix.lstrip("."),
                        "version": meta.get("version", "unknown"),
                        "active": subdir == "active",
                    })

        return models

    def get_status(self) -> dict:
        """Get current model status."""
        return {
            "loaded": self.loaded,
            "version": self.version,
            "format": self.format,
            "load_time_s": round(self.load_time_s, 2),
            "input_shape": list(self._input_shape) if self._input_shape else None,
        }
