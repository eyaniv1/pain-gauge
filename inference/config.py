"""
Inference service configuration.

Reads settings from environment variables with sensible defaults.
"""

import os

# Server
HOST = os.getenv("INFERENCE_HOST", "127.0.0.1")
PORT = int(os.getenv("INFERENCE_PORT", "5000"))

# Model
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "active"))
MODEL_FILE = os.getenv("MODEL_FILE", "")  # Auto-detect if empty

# AU Weights
AU_WEIGHTS_PATH = os.getenv("AU_WEIGHTS_PATH", os.path.join(os.path.dirname(__file__), "au_weights.json"))

# Inference
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
PREDICT_TIMEOUT_S = float(os.getenv("PREDICT_TIMEOUT_S", "5.0"))
